// llm-on-cpu :: model/qwen3_5_int4_model.cpp (INT4/QLWC; does not modify BF16 path)
#include "model/qwen3_5_int4_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <stdexcept>

#include "hal/cpu_ops.h"
#include "hal/int4_ops.h"
#include "model/mtp_head.h"
#include "model/tokenizer_hf.h"
#include "model/vision/image_preprocess.h"
#include "weights/qlwc_store.h"
#include "common/log.h"
#include <nlohmann/json.hpp>

#if defined(LLMOC_ENABLE_AVX2)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace llmoc::model {
namespace {

float softplus(float x) {
  if (!std::isfinite(x)) return 0.f;
  if (x > 20.f) return x;
  if (x < -20.f) return std::exp(x);
  return std::log1p(std::exp(x));
}
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// decode 热路径复用缓冲，避免每层 vector 分配
struct Int4Scratch {
  std::vector<float> normed, attn_out, residual;
  std::vector<float> qg, kk, vv, qq, gate, attn_heads, kpf, vpf;
  std::vector<float> mixed, z, b, a, mixed_c, q, k, v, g, beta, core;
  std::vector<float> gproj, uproj, mid, down, last, xbuf, logits;
  static void fit(std::vector<float>& v, size_t n) {
    if (v.size() < n) v.resize(n);
  }
};
Int4Scratch& scratch() {
  static thread_local Int4Scratch s;
  return s;
}

}  // namespace

void Qwen35Int4Model::load(qlwc::QlwcStore* store, const std::string& hf_config_json_path) {
  store_ = store;
  pass_wd_ = hal::WDtype::kBF16;
  for (const auto& tm : store_->header().tensors) {
    if (tm.kind == qlwc::TensorKind::kPassthrough) {
      pass_wd_ = tm.pass_dtype == qlwc::PassDtype::kF16 ? hal::WDtype::kF16 : hal::WDtype::kBF16;
      break;
    }
  }
  std::ifstream in(hf_config_json_path);
  if (!in) throw std::runtime_error("cannot open " + hf_config_json_path);
  nlohmann::json root;
  try {
    in >> root;
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("bad JSON in " + hf_config_json_path + ": " + e.what() +
                             " (file must be UTF-8; re-download if corrupted)");
  }
  const auto& tc = root.contains("text_config") ? root["text_config"] : root;
  cfg_.hidden = tc.value("hidden_size", 2560);
  cfg_.layers = tc.value("num_hidden_layers", 32);
  cfg_.n_heads = tc.value("num_attention_heads", 16);
  cfg_.n_kv = tc.value("num_key_value_heads", 4);
  cfg_.head_dim = tc.value("head_dim", 256);
  cfg_.intermediate = tc.value("intermediate_size", 9216);
  cfg_.vocab = tc.value("vocab_size", 248320);
  cfg_.rms_eps = static_cast<float>(tc.value("rms_norm_eps", 1e-6));
  cfg_.tie_embeddings = tc.value("tie_word_embeddings", true);
  cfg_.linear_num_k = tc.value("linear_num_key_heads", 16);
  cfg_.linear_num_v = tc.value("linear_num_value_heads", 32);
  cfg_.linear_dk = tc.value("linear_key_head_dim", 128);
  cfg_.linear_dv = tc.value("linear_value_head_dim", 128);
  cfg_.conv_k = tc.value("linear_conv_kernel_dim", 4);
  if (tc.contains("rope_parameters")) {
    const auto& rp = tc["rope_parameters"];
    cfg_.rope_theta = rp.value("rope_theta", 10000000.f);
    cfg_.partial_rotary = rp.value("partial_rotary_factor", 0.25f);
    mrope_interleaved_ = rp.value("mrope_interleaved", true);
    if (rp.contains("mrope_section") && rp["mrope_section"].is_array() &&
        rp["mrope_section"].size() >= 3) {
      mrope_section_[0] = rp["mrope_section"][0].get<int>();
      mrope_section_[1] = rp["mrope_section"][1].get<int>();
      mrope_section_[2] = rp["mrope_section"][2].get<int>();
    }
  }
  cfg_.layer_types.clear();
  if (tc.contains("layer_types")) {
    for (const auto& t : tc["layer_types"]) cfg_.layer_types.push_back(t.get<std::string>());
  } else {
    for (int i = 0; i < cfg_.layers; ++i)
      cfg_.layer_types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
  }
  cfg_.image_token_id = root.value("image_token_id", 248056);
  cfg_.vision_start_id = root.value("vision_start_token_id", 248053);
  cfg_.vision_end_id = root.value("vision_end_token_id", 248054);

  // 探测权重前缀
  if (store_->has("language_model.embed_tokens.weight"))
    prefix_ = "language_model.";
  else if (store_->has("embedding.weight"))
    prefix_ = "";
  else
    throw std::runtime_error("cannot find embed_tokens in QLWC");

  meta_.hidden = cfg_.hidden;
  meta_.layers = cfg_.layers;
  meta_.vocab = cfg_.vocab;
  meta_.n_kv = cfg_.n_kv;
  meta_.head_dim = cfg_.head_dim;
  meta_.linear_num_v = cfg_.linear_num_v;
  meta_.linear_dk = cfg_.linear_dk;
  meta_.linear_dv = cfg_.linear_dv;
  meta_.conv_k = cfg_.conv_k;
  meta_.conv_dim =
      cfg_.linear_num_k * cfg_.linear_dk * 2 + cfg_.linear_num_v * cfg_.linear_dv;
  meta_.is_moe = false;
  meta_.kind = "qwen3_5_int4";

  if (store_->has("visual.patch_embed.proj.weight") && root.contains("vision_config")) {
    const auto& vc = root["vision_config"];
    vision::VisionConfig vcfg;
    vcfg.depth = vc.value("depth", 24);
    vcfg.hidden = vc.value("hidden_size", 1024);
    vcfg.intermediate = vc.value("intermediate_size", 4096);
    vcfg.num_heads = vc.value("num_heads", 16);
    vcfg.patch_size = vc.value("patch_size", 16);
    vcfg.temporal_patch_size = vc.value("temporal_patch_size", 2);
    vcfg.spatial_merge = vc.value("spatial_merge_size", 2);
    vcfg.out_hidden = vc.value("out_hidden_size", cfg_.hidden);
    vcfg.in_channels = vc.value("in_channels", 3);
    vcfg.num_position_embeddings = vc.value("num_position_embeddings", 2304);
    vision_.load(store_, vcfg);
    image_prep_.patch_size = vcfg.patch_size;
    image_prep_.temporal_patch_size = vcfg.temporal_patch_size;
    image_prep_.merge_size = vcfg.spatial_merge;

    // 从 HF preprocessor_config 读 min/max_pixels（勿再把 max 锁死成 256²）
    {
      const auto slash = hf_config_json_path.find_last_of("/\\");
      const std::string dir =
          (slash == std::string::npos) ? "." : hf_config_json_path.substr(0, slash);
      const std::string prep_path = dir + "/preprocessor_config.json";
      std::ifstream pf(prep_path);
      if (pf) {
        nlohmann::json prep = nlohmann::json::parse(pf, nullptr, false);
        if (!prep.is_discarded() && prep.contains("size")) {
          const auto& sz = prep["size"];
          image_prep_.min_pixels = sz.value("shortest_edge", image_prep_.min_pixels);
          image_prep_.max_pixels = sz.value("longest_edge", image_prep_.max_pixels);
        }
      }
      // CPU 软顶：512² 平衡识字与速度（1024² 过慢）；官方最长可达 16M
      constexpr int kCpuMaxPixels = 512 * 512;
      if (image_prep_.max_pixels > kCpuMaxPixels) image_prep_.max_pixels = kCpuMaxPixels;
      if (image_prep_.min_pixels > image_prep_.max_pixels)
        image_prep_.min_pixels = image_prep_.max_pixels;
      LOG_INFO("vision: image_prep min_pixels=%d max_pixels=%d (~%dx%d)", image_prep_.min_pixels,
               image_prep_.max_pixels,
               static_cast<int>(std::sqrt(static_cast<double>(image_prep_.max_pixels))),
               static_cast<int>(std::sqrt(static_cast<double>(image_prep_.max_pixels))));
    }
  }

  build_layer_packs();
}

void Qwen35Int4Model::build_layer_packs() {
  layers_.assign(cfg_.layers, {});
  const int nk = cfg_.linear_num_k, nv = cfg_.linear_num_v;
  const int dk = cfg_.linear_dk, dv = cfg_.linear_dv;
  const int conv_dim = nk * dk * 2 + nv * dv;
  for (int L = 0; L < cfg_.layers; ++L) {
    auto& lp = layers_[L];
    const std::string base = prefix_ + "layers." + std::to_string(L) + ".";
    lp.is_full = (cfg_.layer_types[L] == "full_attention");
    lp.ln1 = pass(base + "input_layernorm.weight");
    lp.ln2 = pass(base + "post_attention_layernorm.weight");
    lp.wgate = store_->get_int4(base + "mlp.gate_proj.weight");
    lp.wup = store_->get_int4(base + "mlp.up_proj.weight");
    lp.wdown = store_->get_int4(base + "mlp.down_proj.weight");
    if (lp.is_full) {
      lp.wq = store_->get_int4(base + "self_attn.q_proj.weight");
      lp.wk = store_->get_int4(base + "self_attn.k_proj.weight");
      lp.wv = store_->get_int4(base + "self_attn.v_proj.weight");
      lp.wo = store_->get_int4(base + "self_attn.o_proj.weight");
      lp.qn = pass(base + "self_attn.q_norm.weight");
      lp.kn = pass(base + "self_attn.k_norm.weight");
    } else {
      lp.wqkv = store_->get_int4(base + "linear_attn.in_proj_qkv.weight");
      lp.wz = store_->get_int4(base + "linear_attn.in_proj_z.weight");
      lp.wb = store_->get_int4(base + "linear_attn.in_proj_b.weight");
      lp.wa = store_->get_int4(base + "linear_attn.in_proj_a.weight");
      lp.wout = store_->get_int4(base + "linear_attn.out_proj.weight");
      lp.nrm = pass(base + "linear_attn.norm.weight");
      const uint16_t* A_log = pass(base + "linear_attn.A_log");
      const uint16_t* dt_bias = pass(base + "linear_attn.dt_bias");
      const uint16_t* conv_w = pass(base + "linear_attn.conv1d.weight");
      lp.A_log_f.resize(nv);
      lp.dt_bias_f.resize(nv);
      for (int h = 0; h < nv; ++h) {
        lp.A_log_f[h] = hal::load_w(A_log + h, pass_wd_);
        lp.dt_bias_f[h] = hal::load_w(dt_bias + h, pass_wd_);
      }
      lp.conv_w_f.resize(static_cast<size_t>(conv_dim) * cfg_.conv_k);
      for (int c = 0; c < conv_dim; ++c)
        for (int k = 0; k < cfg_.conv_k; ++k)
          lp.conv_w_f[c * cfg_.conv_k + k] =
              hal::load_w(conv_w + c * cfg_.conv_k + k, pass_wd_);
    }
  }
  const std::string emb_name =
      prefix_.empty() ? "embedding.weight" : prefix_ + "embed_tokens.weight";
  emb_is_int4_ = is_int4(emb_name);
  if (emb_is_int4_) emb_int4_ = store_->get_int4(emb_name);
  else emb_pass_ = pass(emb_name);

  lm_is_int4_ = false;
  lm_pass_ = nullptr;
  lm_int4_ = {};
  if (!cfg_.tie_embeddings) {
    std::string lm_name;
    if (store_->has(prefix_ + "lm_head.weight")) lm_name = prefix_ + "lm_head.weight";
    else if (store_->has("lm_head.weight")) lm_name = "lm_head.weight";
    else
      throw std::runtime_error(
          "tie_word_embeddings=false but lm_head.weight missing in QLWC");
    lm_is_int4_ = is_int4(lm_name);
    if (lm_is_int4_) lm_int4_ = store_->get_int4(lm_name);
    else lm_pass_ = pass(lm_name);
  } else {
    lm_is_int4_ = emb_is_int4_;
    lm_int4_ = emb_int4_;
    lm_pass_ = emb_pass_;
  }

  final_norm_ = pass(prefix_ + "norm.weight");
  LOG_INFO("Qwen35Int4: layers=%d hidden=%d heads=%d lin_v=%d tie=%d", cfg_.layers, cfg_.hidden,
           cfg_.n_heads, cfg_.linear_num_v, cfg_.tie_embeddings ? 1 : 0);
}

void Qwen35Int4Model::init_cache(SessionCache& cache, int max_seq) const {
  cache.init(cfg_.layers, max_seq, cfg_.n_kv, cfg_.head_dim, cfg_.linear_num_v, cfg_.linear_dk,
             cfg_.linear_dv, meta_.conv_dim, cfg_.conv_k);
}

bool Qwen35Int4Model::is_int4(const std::string& name) const {
  return store_->is_int4(name);
}

const uint16_t* Qwen35Int4Model::pass(const std::string& name) {
  return store_->get_pass(name).data;
}

void Qwen35Int4Model::gemm_w(const float* x, const std::string& wname, float* y, int M, int K) {
  if (is_int4(wname)) {
    hal::gemm_int4(x, store_->get_int4(wname), y);
  } else {
    (void)M;
    (void)K;
    hal::gemm_bias_free(x, pass(wname), y, M, K, pass_wd_);
  }
}

void Qwen35Int4Model::gemm_view(const float* x, const qlwc::Int4View& W, float* y) {
  hal::gemm_int4(x, W, y);
}

void Qwen35Int4Model::gemm_view_batch(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  if (n <= 1) {
    if (n == 1) hal::gemm_int4(X, W, Y);
    return;
  }
  hal::gemm_int4_batch(X, n, W, Y);
}

void Qwen35Int4Model::embed(int32_t token, float* out) {
  if (vision_n_tok_ > 0 && token == cfg_.image_token_id) {
    if (vision_cursor_ >= vision_n_tok_)
      throw std::runtime_error("image_pad count exceeds vision embeds");
    std::memcpy(out, vision_embeds_.data() + static_cast<size_t>(vision_cursor_) * cfg_.hidden,
                sizeof(float) * cfg_.hidden);
    ++vision_cursor_;
    return;
  }
  if (emb_is_int4_) {
    hal::dequant_int4_row(emb_int4_, token, out);
    return;
  }
  const uint16_t* row = emb_pass_ + static_cast<size_t>(token) * cfg_.hidden;
  for (int i = 0; i < cfg_.hidden; ++i) out[i] = hal::load_w(row + i, pass_wd_);
}

void Qwen35Int4Model::set_vision_embeds(std::vector<float> embeds, int n_tok) {
  if (n_tok < 0 || static_cast<size_t>(n_tok) * cfg_.hidden != embeds.size())
    throw std::runtime_error("vision embeds size mismatch");
  vision_embeds_ = std::move(embeds);
  vision_n_tok_ = n_tok;
  vision_cursor_ = 0;
}

void Qwen35Int4Model::clear_vision_embeds() {
  vision_embeds_.clear();
  vision_n_tok_ = 0;
  vision_cursor_ = 0;
  vision_grid_thw_.clear();
}

bool Qwen35Int4Model::encode_message_images(const std::vector<ChatMessage>& messages,
                                            std::vector<float>& embeds_out,
                                            std::vector<int>& pad_counts_out) {
  if (!vision_.ready()) return false;
  embeds_out.clear();
  pad_counts_out.clear();
  vision_grid_thw_.clear();
  for (const auto& m : messages) {
    for (const auto& im : m.images) {
      LOG_INFO("vision: preprocess image (%zu bytes)", im.bytes.size());
      auto prepared =
          vision::prepare_image_bytes(im.bytes.data(), im.bytes.size(), image_prep_);
      std::vector<float> embeds;
      LOG_INFO("vision: encode resize_grid=(%d,%d) patches=%d -> %d tokens (max_pixels=%d)",
               prepared.grid_h * image_prep_.patch_size, prepared.grid_w * image_prep_.patch_size,
               prepared.num_patches(), prepared.num_merged_tokens(), image_prep_.max_pixels);
      vision_.encode(prepared, embeds);
      pad_counts_out.push_back(prepared.num_merged_tokens());
      embeds_out.insert(embeds_out.end(), embeds.begin(), embeds.end());
      vision_grid_thw_.push_back(prepared.grid_t);
      vision_grid_thw_.push_back(prepared.grid_h);
      vision_grid_thw_.push_back(prepared.grid_w);
    }
  }
  return !pad_counts_out.empty();
}


void Qwen35Int4Model::prepare_mrope_positions(const std::vector<int32_t>& tokens, bool is_prefill) {
  const int n = static_cast<int>(tokens.size());
  cur_pos_t_.assign(n, 0);
  cur_pos_h_.assign(n, 0);
  cur_pos_w_.assign(n, 0);
  if (!is_prefill || vision_grid_thw_.empty()) {
    // decode / 纯文本：三维同位（等价 1D RoPE）
    const int base = is_prefill ? 0 : mrope_next_;
    for (int i = 0; i < n; ++i) {
      cur_pos_t_[i] = cur_pos_h_[i] = cur_pos_w_[i] = base + i;
    }
    mrope_next_ = base + n;
    return;
  }

  // Qwen3-VL get_rope_index：vision 用 (t,h,w) 网格，文本三维同号
  const int merge = image_prep_.merge_size;
  const int32_t img_pad = cfg_.image_token_id;
  int img_i = 0;
  int st = 0;
  int st_idx = 0;

  while (st < n) {
    int ed_image = n + 1;
    for (int i = st; i < n; ++i) {
      if (tokens[i] == img_pad) {
        ed_image = i;
        break;
      }
    }
    if (ed_image > n) {
      // 尾部纯文本
      for (int i = st; i < n; ++i) {
        cur_pos_t_[i] = cur_pos_h_[i] = cur_pos_w_[i] = st_idx + (i - st);
      }
      st_idx += (n - st);
      break;
    }
    // 文本前缀到首个 image_pad
    const int text_len = ed_image - st;
    for (int i = 0; i < text_len; ++i) {
      cur_pos_t_[st + i] = cur_pos_h_[st + i] = cur_pos_w_[st + i] = st_idx + i;
    }
    if (img_i * 3 + 2 >= static_cast<int>(vision_grid_thw_.size()))
      throw std::runtime_error("vision_grid_thw underflow");
    const int gt = vision_grid_thw_[img_i * 3 + 0];
    const int gh = vision_grid_thw_[img_i * 3 + 1] / merge;
    const int gw = vision_grid_thw_[img_i * 3 + 2] / merge;
    const int n_vis = gt * gh * gw;
    ++img_i;
    const int vis_base = st_idx + text_len;
    for (int t = 0; t < gt; ++t) {
      for (int hi = 0; hi < gh; ++hi) {
        for (int wi = 0; wi < gw; ++wi) {
          const int j = t * gh * gw + hi * gw + wi;
          const int idx = ed_image + j;
          if (idx >= n) throw std::runtime_error("vision token span OOB");
          cur_pos_t_[idx] = vis_base + t;  // image: t 维通常为 0
          cur_pos_h_[idx] = vis_base + hi;
          cur_pos_w_[idx] = vis_base + wi;
        }
      }
    }
    // Qwen3: t_index always 0 relative; absolute = text_len+st_idx + t(=0)
    // 上面 t 维用了 vis_base+t；与官方 t_index.flatten()+text_len+st_idx 一致（t=0..）
    st = ed_image + n_vis;
    int local_max = vis_base;
    for (int j = 0; j < n_vis; ++j) {
      local_max = std::max(local_max, cur_pos_t_[ed_image + j]);
      local_max = std::max(local_max, cur_pos_h_[ed_image + j]);
      local_max = std::max(local_max, cur_pos_w_[ed_image + j]);
    }
    st_idx = local_max + 1;
  }
  int mx = 0;
  for (int i = 0; i < n; ++i)
    mx = std::max(mx, std::max(cur_pos_t_[i], std::max(cur_pos_h_[i], cur_pos_w_[i])));
  mrope_next_ = mx + 1;
}

void Qwen35Int4Model::layer_forward(int layer, float* x, SessionCache& cache, int pos_start, int n_tok,
                                    bool is_prefill) {
  const auto& lp = layers_[layer];
  const int H = cfg_.hidden;
  const int I = cfg_.intermediate;
  auto& sc = scratch();
  const size_t nH = static_cast<size_t>(n_tok) * H;
  Int4Scratch::fit(sc.normed, nH);
  Int4Scratch::fit(sc.attn_out, nH);
  Int4Scratch::fit(sc.residual, nH);
  std::memset(sc.attn_out.data(), 0, sizeof(float) * nH);
  std::memcpy(sc.residual.data(), x, sizeof(float) * nH);

  for (int t = 0; t < n_tok; ++t)
     hal::rmsnorm(x + t * H, lp.ln1, sc.normed.data() + t * H, H, cfg_.rms_eps, pass_wd_, true);

  auto& Lkv = cache.layer(layer);

  if (lp.is_full) {
    const int nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
    const int rotary_dim = static_cast<int>(hd * cfg_.partial_rotary) / 2 * 2;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    Int4Scratch::fit(sc.qg, static_cast<size_t>(n_tok) * nh * hd * 2);
    Int4Scratch::fit(sc.kk, static_cast<size_t>(n_tok) * nkv * hd);
    Int4Scratch::fit(sc.vv, static_cast<size_t>(n_tok) * nkv * hd);
    Int4Scratch::fit(sc.qq, static_cast<size_t>(n_tok) * nh * hd);
    Int4Scratch::fit(sc.gate, static_cast<size_t>(n_tok) * nh * hd);
    Int4Scratch::fit(sc.attn_heads, static_cast<size_t>(n_tok) * nh * hd);
    if (n_tok > 1) {
      gemm_view_batch(sc.normed.data(), n_tok, lp.wq, sc.qg.data());
      gemm_view_batch(sc.normed.data(), n_tok, lp.wk, sc.kk.data());
      gemm_view_batch(sc.normed.data(), n_tok, lp.wv, sc.vv.data());
    } else {
      gemm_view(sc.normed.data(), lp.wq, sc.qg.data());
      gemm_view(sc.normed.data(), lp.wk, sc.kk.data());
      gemm_view(sc.normed.data(), lp.wv, sc.vv.data());
    }
    for (int t = 0; t < n_tok; ++t) {
      const int idx = (static_cast<int>(cur_pos_t_.size()) == n_tok) ? t : (pos_start + t);
      const int pt = (idx < static_cast<int>(cur_pos_t_.size())) ? cur_pos_t_[idx] : (pos_start + t);
      const int ph = (idx < static_cast<int>(cur_pos_h_.size())) ? cur_pos_h_[idx] : pt;
      const int pw = (idx < static_cast<int>(cur_pos_w_.size())) ? cur_pos_w_[idx] : pt;
      for (int h = 0; h < nh; ++h) {
        float* qh = sc.qq.data() + (t * nh + h) * hd;
        float* gh = sc.gate.data() + (t * nh + h) * hd;
        const float* src = sc.qg.data() + (t * nh + h) * hd * 2;
        std::memcpy(qh, src, sizeof(float) * hd);
        std::memcpy(gh, src + hd, sizeof(float) * hd);
        hal::rmsnorm(qh, lp.qn, qh, hd, cfg_.rms_eps, pass_wd_, true);
        hal::apply_mrope_freqs(qh, hd, rotary_dim, pt, ph, pw, cfg_.rope_theta, mrope_section_,
                               mrope_interleaved_);
      }
      for (int h = 0; h < nkv; ++h) {
        float* kh = sc.kk.data() + (t * nkv + h) * hd;
        float* vh = sc.vv.data() + (t * nkv + h) * hd;
        hal::rmsnorm(kh, lp.kn, kh, hd, cfg_.rms_eps, pass_wd_, true);
        hal::apply_mrope_freqs(kh, hd, rotary_dim, pt, ph, pw, cfg_.rope_theta, mrope_section_,
                               mrope_interleaved_);
        float* kdst = Lkv.k.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq + t) * hd;
        float* vdst = Lkv.v.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq + t) * hd;
        std::memcpy(kdst, kh, sizeof(float) * hd);
        std::memcpy(vdst, vh, sizeof(float) * hd);
      }
    }
    if (is_prefill && Lkv.seq == 0) {
      Int4Scratch::fit(sc.kpf, static_cast<size_t>(n_tok) * nkv * hd);
      Int4Scratch::fit(sc.vpf, static_cast<size_t>(n_tok) * nkv * hd);
      for (int t = 0; t < n_tok; ++t)
        for (int h = 0; h < nkv; ++h) {
          std::memcpy(sc.kpf.data() + (t * nkv + h) * hd,
                      Lkv.k.data() + (static_cast<size_t>(h) * cache.max_seq() + t) * hd,
                      sizeof(float) * hd);
          std::memcpy(sc.vpf.data() + (t * nkv + h) * hd,
                      Lkv.v.data() + (static_cast<size_t>(h) * cache.max_seq() + t) * hd,
                      sizeof(float) * hd);
        }
       hal::attn_prefill(sc.qq.data(), sc.kpf.data(), sc.vpf.data(), sc.attn_heads.data(), n_tok, nh,
                        nkv, hd, scale);
    } else {
      for (int t = 0; t < n_tok; ++t) {
        const int seq_len = Lkv.seq + t + 1;
          hal::attn_decode_one(sc.qq.data() + t * nh * hd, Lkv.k.data(), Lkv.v.data(),
                             sc.attn_heads.data() + t * nh * hd, nh, nkv, hd, seq_len,
                             cache.max_seq(), scale);
      }
    }
    Lkv.seq += n_tok;
    for (int t = 0; t < n_tok; ++t) {
      for (int i = 0; i < nh * hd; ++i)
        sc.attn_heads[t * nh * hd + i] *= sigmoid(sc.gate[t * nh * hd + i]);
    }
    if (n_tok > 1)
      gemm_view_batch(sc.attn_heads.data(), n_tok, lp.wo, sc.attn_out.data());
    else
      gemm_view(sc.attn_heads.data(), lp.wo, sc.attn_out.data());
  } else {
    const int nk = cfg_.linear_num_k, nv = cfg_.linear_num_v;
    const int dk = cfg_.linear_dk, dv = cfg_.linear_dv;
    const int key_dim = nk * dk, value_dim = nv * dv;
    const int conv_dim = key_dim * 2 + value_dim;
    Int4Scratch::fit(sc.mixed, static_cast<size_t>(n_tok) * conv_dim);
    Int4Scratch::fit(sc.z, static_cast<size_t>(n_tok) * value_dim);
    Int4Scratch::fit(sc.b, static_cast<size_t>(n_tok) * nv);
    Int4Scratch::fit(sc.a, static_cast<size_t>(n_tok) * nv);
    Int4Scratch::fit(sc.mixed_c, static_cast<size_t>(n_tok) * conv_dim);
    Int4Scratch::fit(sc.q, static_cast<size_t>(n_tok) * nv * dk);
    Int4Scratch::fit(sc.k, static_cast<size_t>(n_tok) * nv * dk);
    Int4Scratch::fit(sc.v, static_cast<size_t>(n_tok) * nv * dv);
    Int4Scratch::fit(sc.g, static_cast<size_t>(n_tok) * nv);
    Int4Scratch::fit(sc.beta, static_cast<size_t>(n_tok) * nv);
    Int4Scratch::fit(sc.core, static_cast<size_t>(n_tok) * value_dim);

    if (n_tok > 1) {
      gemm_view_batch(sc.normed.data(), n_tok, lp.wqkv, sc.mixed.data());
      gemm_view_batch(sc.normed.data(), n_tok, lp.wz, sc.z.data());
      gemm_view_batch(sc.normed.data(), n_tok, lp.wb, sc.b.data());
      gemm_view_batch(sc.normed.data(), n_tok, lp.wa, sc.a.data());
    } else {
      gemm_view(sc.normed.data(), lp.wqkv, sc.mixed.data());
      gemm_view(sc.normed.data(), lp.wz, sc.z.data());
      gemm_view(sc.normed.data(), lp.wb, sc.b.data());
      gemm_view(sc.normed.data(), lp.wa, sc.a.data());
    }

    auto& conv_state = Lkv.linear.conv;
    const float* cw = lp.conv_w_f.data();
    const int ck = cfg_.conv_k;
    for (int t = 0; t < n_tok; ++t) {
      const float* xin = sc.mixed.data() + t * conv_dim;
      float* xout = sc.mixed_c.data() + t * conv_dim;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (conv_dim >= 1024 && !omp_in_parallel())
#endif
      for (int c = 0; c < conv_dim; ++c) {
        float* st = &conv_state[static_cast<size_t>(c) * ck];
        const float* wk = cw + static_cast<size_t>(c) * ck;
        // depthwise conv_k=4 展开：shift + silu(dot)
        if (ck == 4) {
          st[3] = st[2];
          st[2] = st[1];
          st[1] = st[0];
          st[0] = xin[c];
          const float acc = st[0] * wk[3] + st[1] * wk[2] + st[2] * wk[1] + st[3] * wk[0];
          xout[c] = acc / (1.f + std::exp(-acc));
        } else {
          for (int k = ck - 1; k > 0; --k) st[k] = st[k - 1];
          st[0] = xin[c];
          float acc = 0.f;
          for (int k = 0; k < ck; ++k) acc += st[k] * wk[ck - 1 - k];
          xout[c] = acc / (1.f + std::exp(-acc));
        }
      }
    }
    Lkv.linear.has_state = true;

    const int rep = nv / nk;
    for (int t = 0; t < n_tok; ++t) {
      const float* m = sc.mixed_c.data() + t * conv_dim;
      for (int h = 0; h < nk; ++h) {
        for (int r = 0; r < rep; ++r) {
          const int hh = h * rep + r;
          std::memcpy(sc.q.data() + (t * nv + hh) * dk, m + h * dk, sizeof(float) * dk);
          std::memcpy(sc.k.data() + (t * nv + hh) * dk, m + key_dim + h * dk, sizeof(float) * dk);
        }
      }
      std::memcpy(sc.v.data() + t * value_dim, m + 2 * key_dim, sizeof(float) * value_dim);
      for (int h = 0; h < nv; ++h) {
        sc.beta[t * nv + h] = sigmoid(sc.b[t * nv + h]);
        float A = std::exp(lp.A_log_f[h]);
        if (!std::isfinite(A) || A > 1e4f) A = 1e4f;
        if (A < 1e-6f) A = 1e-6f;
        float sp = softplus(sc.a[t * nv + h] + lp.dt_bias_f[h]);
        if (!std::isfinite(sp)) sp = 0.f;
        sc.g[t * nv + h] = -A * sp;
      }
    }

    hal::gated_delta_recurrent(sc.q.data(), sc.k.data(), sc.v.data(), sc.g.data(), sc.beta.data(),
                               Lkv.linear.recurrent.data(), sc.core.data(), n_tok, nv, dk, dv, true);

    for (int t = 0; t < n_tok; ++t) {
      for (int h = 0; h < nv; ++h) {
        float* ch = sc.core.data() + (t * nv + h) * dv;
        float* zh = sc.z.data() + (t * nv + h) * dv;
          hal::rmsnorm_gated(ch, zh, lp.nrm, ch, dv, cfg_.rms_eps, pass_wd_);
      }
    }
    if (n_tok > 1)
      gemm_view_batch(sc.core.data(), n_tok, lp.wout, sc.attn_out.data());
    else
      gemm_view(sc.core.data(), lp.wout, sc.attn_out.data());
  }

  for (size_t i = 0; i < nH; ++i) x[i] = sc.residual[i] + sc.attn_out[i];

  std::memcpy(sc.residual.data(), x, sizeof(float) * nH);
  Int4Scratch::fit(sc.gproj, static_cast<size_t>(n_tok) * I);
  Int4Scratch::fit(sc.uproj, static_cast<size_t>(n_tok) * I);
  Int4Scratch::fit(sc.mid, static_cast<size_t>(n_tok) * I);
  Int4Scratch::fit(sc.down, static_cast<size_t>(n_tok) * H);
  for (int t = 0; t < n_tok; ++t)
    hal::rmsnorm(x + t * H, lp.ln2, sc.normed.data() + t * H, H, cfg_.rms_eps, pass_wd_, true);
  if (n_tok > 1) {
    gemm_view_batch(sc.normed.data(), n_tok, lp.wgate, sc.gproj.data());
    gemm_view_batch(sc.normed.data(), n_tok, lp.wup, sc.uproj.data());
    for (int t = 0; t < n_tok; ++t)
      hal::silu_and_mul(sc.gproj.data() + t * I, sc.uproj.data() + t * I, sc.mid.data() + t * I, I);
    gemm_view_batch(sc.mid.data(), n_tok, lp.wdown, sc.down.data());
    for (int t = 0; t < n_tok; ++t)
      for (int i = 0; i < H; ++i) x[t * H + i] = sc.residual[t * H + i] + sc.down[t * H + i];
  } else {
    gemm_view(sc.normed.data(), lp.wgate, sc.gproj.data());
    gemm_view(sc.normed.data(), lp.wup, sc.uproj.data());
    hal::silu_and_mul(sc.gproj.data(), sc.uproj.data(), sc.mid.data(), I);
    gemm_view(sc.mid.data(), lp.wdown, sc.down.data());
    for (int i = 0; i < H; ++i) x[i] = sc.residual[i] + sc.down[i];
  }
}

void Qwen35Int4Model::forward(const std::vector<int32_t>& tokens, SessionCache& cache,
                              std::vector<float>& logits, bool is_prefill) {
  // 常规路径只需最后一 token 的 logits（避免 prefill 扫 n 次词表）
  if (!store_ || tokens.empty()) throw std::runtime_error("model not ready / empty tokens");
  const int n = static_cast<int>(tokens.size());
  const int H = cfg_.hidden;
  const int V = cfg_.vocab;
  std::vector<float> x(static_cast<size_t>(n) * H);
  int pos_start = 0;
  if (!is_prefill) {
    for (int i = 0; i < cfg_.layers; ++i)
      if (cfg_.layer_types[i] == "full_attention") {
        pos_start = cache.layer(i).seq;
        break;
      }
  }
  if (is_prefill) vision_cursor_ = 0;
  for (int t = 0; t < n; ++t) embed(tokens[t], x.data() + t * H);
  prepare_mrope_positions(tokens, is_prefill);
  for (int L = 0; L < cfg_.layers; ++L) layer_forward(L, x.data(), cache, pos_start, n, is_prefill);

  prefix_hiddens_.clear();
  prefix_logits_.clear();
  std::vector<float> h(H);
  hal::rmsnorm(x.data() + (n - 1) * H, final_norm_, h.data(), H, cfg_.rms_eps, pass_wd_, true);
  for (float& v : h)
    if (!std::isfinite(v)) v = 0.f;
  last_hidden_ = h;
  logits.resize(static_cast<size_t>(V));
  if (lm_is_int4_) {
#if defined(LLMOC_ENABLE_AVX2)
    if (lm_int4_.qweight) {
      const int rb = (lm_int4_.K + 1) / 2;
      _mm_prefetch(reinterpret_cast<const char*>(lm_int4_.qweight), _MM_HINT_T0);
      if (V > 4)
        _mm_prefetch(reinterpret_cast<const char*>(lm_int4_.qweight + static_cast<size_t>(4) * rb),
                     _MM_HINT_T0);
    }
#endif
    hal::gemm_int4(h.data(), lm_int4_, logits.data());
  } else {
    hal::gemm_bias_free(h.data(), lm_pass_, logits.data(), V, H, pass_wd_);
  }
  last_logits_ = logits;
}

void Qwen35Int4Model::forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                                         std::vector<float>& logits_all, bool is_prefill) {
  if (!store_ || tokens.empty()) throw std::runtime_error("model not ready / empty tokens");
  const int n = static_cast<int>(tokens.size());
  const int H = cfg_.hidden;
  const int V = cfg_.vocab;
  static const bool kProf = [] {
    const char* e = std::getenv("LLMOC_PROFILE");
    return e && e[0] == '1';
  }();
  using Clock = std::chrono::steady_clock;
  const auto t_all0 = Clock::now();
  std::vector<float> x(static_cast<size_t>(n) * H);
  int pos_start = 0;
  if (!is_prefill) {
    for (int i = 0; i < cfg_.layers; ++i)
      if (cfg_.layer_types[i] == "full_attention") {
        pos_start = cache.layer(i).seq;
        break;
      }
  }
  if (is_prefill) vision_cursor_ = 0;
  for (int t = 0; t < n; ++t) embed(tokens[t], x.data() + t * H);

  double ms_full = 0, ms_lin = 0;
  prepare_mrope_positions(tokens, is_prefill);
  for (int L = 0; L < cfg_.layers; ++L) {
    if (kProf) {
      const auto a = Clock::now();
      layer_forward(L, x.data(), cache, pos_start, n, is_prefill);
      const auto b = Clock::now();
      const double d = std::chrono::duration<double, std::milli>(b - a).count();
      if (cfg_.layer_types[L] == "full_attention") ms_full += d;
      else ms_lin += d;
    } else {
      layer_forward(L, x.data(), cache, pos_start, n, is_prefill);
    }
  }

  Clock::time_point t_head0;
  if (kProf) t_head0 = Clock::now();
  // decode(n=1)：不零填、不全量拷贝 prefix_*，去掉二次 NaN 扫描
  logits_all.resize(static_cast<size_t>(n) * V);
  const bool keep_prefix = (n > 1);
  if (keep_prefix) prefix_hiddens_.assign(static_cast<size_t>(n) * H, 0.f);
  else {
    prefix_hiddens_.clear();
    prefix_logits_.clear();
  }
  std::vector<float> h(H);
  last_hidden_.assign(H, 0.f);
  if (n > 1 && lm_is_int4_) {
    auto& sc = scratch();
    Int4Scratch::fit(sc.last, static_cast<size_t>(n) * H);
    for (int t = 0; t < n; ++t) {
      float* ht = sc.last.data() + t * H;
      hal::rmsnorm(x.data() + t * H, final_norm_, ht, H, cfg_.rms_eps, pass_wd_, true);
      for (int i = 0; i < H; ++i)
        if (!std::isfinite(ht[i])) ht[i] = 0.f;
      if (keep_prefix)
        std::memcpy(prefix_hiddens_.data() + static_cast<size_t>(t) * H, ht, sizeof(float) * H);
    }
    last_hidden_.assign(sc.last.begin() + static_cast<size_t>(n - 1) * H, sc.last.end());
    gemm_view_batch(sc.last.data(), n, lm_int4_, logits_all.data());
  } else {
    for (int t = 0; t < n; ++t) {
      hal::rmsnorm(x.data() + t * H, final_norm_, h.data(), H, cfg_.rms_eps, pass_wd_, true);
      for (float& v : h)
        if (!std::isfinite(v)) v = 0.f;
      if (keep_prefix)
        std::memcpy(prefix_hiddens_.data() + static_cast<size_t>(t) * H, h.data(), sizeof(float) * H);
      float* dest = logits_all.data() + static_cast<size_t>(t) * V;
      if (lm_is_int4_) {
#if defined(LLMOC_ENABLE_AVX2)
        if (lm_int4_.qweight) {
          const int rb = (lm_int4_.K + 1) / 2;
          _mm_prefetch(reinterpret_cast<const char*>(lm_int4_.qweight), _MM_HINT_T0);
          if (V > 4)
            _mm_prefetch(reinterpret_cast<const char*>(lm_int4_.qweight + static_cast<size_t>(4) * rb),
                         _MM_HINT_T0);
        }
#endif
        hal::gemm_int4(h.data(), lm_int4_, dest);
      } else {
        hal::gemm_bias_free(h.data(), lm_pass_, dest, V, H, pass_wd_);
      }
      if (t == n - 1) last_hidden_ = h;
    }
  }
  if (keep_prefix) {
    for (float& v : logits_all)
      if (!std::isfinite(v)) v = -1e9f;
    prefix_logits_ = logits_all;
  }
  last_logits_.assign(logits_all.begin() + static_cast<size_t>(n - 1) * V, logits_all.end());
  if (kProf && !is_prefill && n == 1) {
    const auto t_all1 = Clock::now();
    const double ms_head = std::chrono::duration<double, std::milli>(t_all1 - t_head0).count();
    const double ms_tot = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
    std::fprintf(stderr, "[profile] layers_full=%.1f layers_linear=%.1f lm_head=%.1f total=%.1f ms\n",
                 ms_full, ms_lin, ms_head, ms_tot);
  }
}

void Qwen35Int4Model::commit_prefix_state(int pos) {
  const int H = cfg_.hidden;
  const int V = cfg_.vocab;
  if (pos < 0 || prefix_hiddens_.size() < static_cast<size_t>(pos + 1) * H) return;
  last_hidden_.assign(prefix_hiddens_.begin() + static_cast<size_t>(pos) * H,
                      prefix_hiddens_.begin() + static_cast<size_t>(pos + 1) * H);
  if (prefix_logits_.size() >= static_cast<size_t>(pos + 1) * V) {
    last_logits_.assign(prefix_logits_.begin() + static_cast<size_t>(pos) * V,
                        prefix_logits_.begin() + static_cast<size_t>(pos + 1) * V);
  }
}

bool Qwen35Int4Model::mtp_has_cb(void* ctx, const std::string& name) {
  return static_cast<Qwen35Int4Model*>(ctx)->store_->has(name);
}
void Qwen35Int4Model::mtp_gemm_cb(void* ctx, const float* x, const std::string& wname, float* y,
                                  int M, int K) {
  static_cast<Qwen35Int4Model*>(ctx)->gemm_w(x, wname, y, M, K);
}
const uint16_t* Qwen35Int4Model::mtp_pass_cb(void* ctx, const std::string& name) {
  return static_cast<Qwen35Int4Model*>(ctx)->pass(name);
}
void Qwen35Int4Model::mtp_embed_cb(void* ctx, int32_t token, float* out) {
  static_cast<Qwen35Int4Model*>(ctx)->embed(token, out);
}
float Qwen35Int4Model::mtp_embed_dot_cb(void* ctx, const float* h, int32_t token) {
  auto* self = static_cast<Qwen35Int4Model*>(ctx);
  std::vector<float> row(self->cfg_.hidden);
  self->embed(token, row.data());
  float acc = 0.f;
  for (int i = 0; i < self->cfg_.hidden; ++i) acc += h[i] * row[i];
  return acc;
}

MtpWeightAccess Qwen35Int4Model::mtp_access() {
  MtpWeightAccess wa;
  wa.hidden = cfg_.hidden;
  wa.n_heads = cfg_.n_heads;
  wa.n_kv = cfg_.n_kv;
  wa.head_dim = cfg_.head_dim;
  wa.intermediate = cfg_.intermediate;
  wa.vocab = cfg_.vocab;
  wa.rms_eps = cfg_.rms_eps;
  wa.rope_theta = cfg_.rope_theta;
  wa.partial_rotary = cfg_.partial_rotary;
  wa.rms_one_plus = true;
  wa.pass_dt = pass_wd_;
  wa.ctx = this;
  wa.has = &mtp_has_cb;
  wa.gemm = &mtp_gemm_cb;
  wa.pass = &mtp_pass_cb;
  wa.embed = &mtp_embed_cb;
  wa.embed_dot = &mtp_embed_dot_cb;
  if (!last_logits_.empty()) {
    wa.hint_logits = last_logits_.data();
    wa.hint_top_m = 256;
  }
  return wa;
}

bool Qwen35Int4Model::has_mtp() const {
  if (!store_) return false;
  return mtp_weights_present(const_cast<Qwen35Int4Model*>(this)->mtp_access());
}

bool Qwen35Int4Model::draft_propose(const std::vector<int32_t>& history, int draft_k,
                                    std::vector<int32_t>& out) {
  out.clear();
  if (last_hidden_.empty()) return false;
  return mtp_draft_propose(mtp_access(), last_hidden_, history, draft_k, out);
}

}  // namespace llmoc::model
