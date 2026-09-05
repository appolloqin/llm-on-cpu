// llm-on-cpu :: model/qwen3_5_model.cpp
#include "model/qwen3_5_model.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"
#include <nlohmann/json.hpp>
#include "common/log.h"
#include "model/mtp_head.h"
#include "weights/lwc_format.h"

#include <chrono>

namespace llmoc::model {
namespace {

float softplus(float x) {
  if (!std::isfinite(x)) return 0.f;
  if (x > 20.f) return x;
  if (x < -20.f) return std::exp(x);
  return std::log1p(std::exp(x));
}
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

void Qwen35Model::load(wt::WeightManager* wm, const std::string& hf_config_json_path) {
  wm_ = wm;
  const auto dt = wm_->header().dtype;
  wd_ = dt == lwc::Dtype::F16 ? hal::WDtype::kF16 : hal::WDtype::kBF16;
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
    cfg_.rope_theta = tc["rope_parameters"].value("rope_theta", 10000000.f);
    cfg_.partial_rotary = tc["rope_parameters"].value("partial_rotary_factor", 0.25f);
  }
  cfg_.layer_types.clear();
  if (tc.contains("layer_types")) {
    for (const auto& t : tc["layer_types"]) cfg_.layer_types.push_back(t.get<std::string>());
  } else {
    for (int i = 0; i < cfg_.layers; ++i)
      cfg_.layer_types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
  }
  // 探测权重前缀
  if (wm_->header().find("language_model.embed_tokens.weight"))
    prefix_ = "language_model.";
  else if (wm_->header().find("embedding.weight"))
    prefix_ = "";
  else
    throw std::runtime_error("cannot find embed_tokens in LWC");

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
  meta_.kind = "qwen3_5";

  // untied lm_head（Qwen3.8-27B 等）；tied 时复用 embed
  lm_head_name_.clear();
  if (!cfg_.tie_embeddings) {
    if (wm_->header().find(prefix_ + "lm_head.weight"))
      lm_head_name_ = prefix_ + "lm_head.weight";
    else if (wm_->header().find("lm_head.weight"))
      lm_head_name_ = "lm_head.weight";
    else
      throw std::runtime_error(
          "tie_word_embeddings=false but lm_head.weight missing in LWC");
  }

  // 启动自检：第一层 full attn 的 q_proj 字节数须匹配 config（防 4B 配置套 27B 权重）
  {
    const int elem = (wd_ == hal::WDtype::kF32) ? 4 : 2;
    for (int i = 0; i < cfg_.layers; ++i) {
      if (cfg_.layer_types[i] != "full_attention") continue;
      const std::string name =
          prefix_ + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
      const auto* t = wm_->header().find(name);
      if (!t) throw std::runtime_error("missing " + name);
      const uint64_t expect =
          static_cast<uint64_t>(2) * cfg_.n_heads * cfg_.head_dim * cfg_.hidden * elem;
      if (t->nbytes != expect)
        throw std::runtime_error(
            "q_proj size mismatch: " + name + " got " + std::to_string(t->nbytes) +
            " expect " + std::to_string(expect) +
            " (config.json must match this checkpoint — Qwen3.8-27B: hidden=5120 heads=24)");
      break;
    }
  }

  LOG_INFO("Qwen35Model: layers=%d hidden=%d heads=%d/%d hd=%d inter=%d lin_v=%d tie=%d lm_head=%s",
           cfg_.layers, cfg_.hidden, cfg_.n_heads, cfg_.n_kv, cfg_.head_dim, cfg_.intermediate,
           cfg_.linear_num_v, cfg_.tie_embeddings ? 1 : 0,
           lm_head_name_.empty() ? "(tied embed)" : lm_head_name_.c_str());
}

void Qwen35Model::warm_gpu_bf16_weights() {
  if (!hal::cuda::enabled() || !wm_) return;
  const bool is_f16 = (wd_ == hal::WDtype::kF16);
  int n_ok = 0, n_skip = 0;
  for (const auto& t : wm_->header().tensors) {
    if (t.shape.size() < 2) continue;
    if (t.name.find("embed") != std::string::npos || t.name.find("lm_head") != std::string::npos)
      continue;
    const int M = static_cast<int>(t.shape[0]);
    const int K = static_cast<int>(t.shape[1]);
    if (M <= 0 || K <= 0 || M >= 65536) {
      ++n_skip;
      continue;
    }
    try {
      const uint16_t* W = w(t.name);
      if (hal::cuda::prefetch_w16(W, M, K, is_f16))
        ++n_ok;
      else
        ++n_skip;
    } catch (...) {
      ++n_skip;
    }
  }
  LOG_INFO("Qwen35: warm_gpu_bf16 ok=%d skip=%d used=%.2fGiB", n_ok, n_skip,
           hal::cuda::vram_used() / double(1ull << 30));
  hal::cuda::log_status();
}

void Qwen35Model::init_cache(SessionCache& cache, int max_seq) const {
  std::vector<uint8_t> need_kv(static_cast<size_t>(cfg_.layers), 1);
  for (int i = 0; i < cfg_.layers; ++i)
    need_kv[static_cast<size_t>(i)] = (cfg_.layer_types[i] == "full_attention") ? 1 : 0;
  cache.init(cfg_.layers, max_seq, cfg_.n_kv, cfg_.head_dim, cfg_.linear_num_v, cfg_.linear_dk,
             cfg_.linear_dv, meta_.conv_dim, cfg_.conv_k, &need_kv);
}

const uint16_t* Qwen35Model::w(const std::string& name) {
  const auto& bytes = wm_->get(name);
  return reinterpret_cast<const uint16_t*>(bytes.data());
}

void Qwen35Model::embed(int32_t token, float* out) {
  const std::string name =
      prefix_.empty() ? "embedding.weight" : prefix_ + "embed_tokens.weight";
  const uint16_t* emb = w(name);
  const uint16_t* row = emb + static_cast<size_t>(token) * cfg_.hidden;
  for (int i = 0; i < cfg_.hidden; ++i) out[i] = hal::load_w(row + i, wd_);
}

void Qwen35Model::layer_forward(int layer, float* x, SessionCache& cache, int pos_start, int n_tok,
                                bool is_prefill) {
  const std::string base = prefix_ + "layers." + std::to_string(layer) + ".";
  const int H = cfg_.hidden;
  const int I = cfg_.intermediate;
  std::vector<float> normed(static_cast<size_t>(n_tok) * H);
  std::vector<float> attn_out(static_cast<size_t>(n_tok) * H, 0.f);
  std::vector<float> residual(static_cast<size_t>(n_tok) * H);
  std::memcpy(residual.data(), x, sizeof(float) * n_tok * H);

  const uint16_t* ln1 = w(base + "input_layernorm.weight");
  for (int t = 0; t < n_tok; ++t)
    hal::rmsnorm(x + t * H, ln1, normed.data() + t * H, H, cfg_.rms_eps, wd_, true);

  const std::string& ltype = cfg_.layer_types[layer];
  auto& Lkv = cache.layer(layer);

  if (ltype == "full_attention") {
    const int nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
    const int rotary_dim = static_cast<int>(hd * cfg_.partial_rotary) / 2 * 2;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    // q_proj: 2*nh*hd (q + gate)
    std::vector<float> qg(static_cast<size_t>(n_tok) * nh * hd * 2);
    std::vector<float> kk(static_cast<size_t>(n_tok) * nkv * hd);
    std::vector<float> vv(static_cast<size_t>(n_tok) * nkv * hd);
    const uint16_t* Wq = w(base + "self_attn.q_proj.weight");
    const uint16_t* Wk = w(base + "self_attn.k_proj.weight");
    const uint16_t* Wv = w(base + "self_attn.v_proj.weight");
    const uint16_t* qn = w(base + "self_attn.q_norm.weight");
    const uint16_t* kn = w(base + "self_attn.k_norm.weight");
    for (int t = 0; t < n_tok; ++t) {
      hal::gemm_bias_free(normed.data() + t * H, Wq, qg.data() + t * nh * hd * 2, nh * hd * 2, H, wd_);
      hal::gemm_bias_free(normed.data() + t * H, Wk, kk.data() + t * nkv * hd, nkv * hd, H, wd_);
      hal::gemm_bias_free(normed.data() + t * H, Wv, vv.data() + t * nkv * hd, nkv * hd, H, wd_);
    }
    // split q/gate, apply norms + rope, write KV cache
    std::vector<float> qq(static_cast<size_t>(n_tok) * nh * hd);
    std::vector<float> gate(static_cast<size_t>(n_tok) * nh * hd);
    for (int t = 0; t < n_tok; ++t) {
      const int pos = pos_start + t;
      for (int h = 0; h < nh; ++h) {
        float* qh = qq.data() + (t * nh + h) * hd;
        float* gh = gate.data() + (t * nh + h) * hd;
        const float* src = qg.data() + (t * nh + h) * hd * 2;
        std::memcpy(qh, src, sizeof(float) * hd);
        std::memcpy(gh, src + hd, sizeof(float) * hd);
        hal::rmsnorm(qh, qn, qh, hd, cfg_.rms_eps, wd_, true);
        hal::apply_rope_freqs(qh, hd, rotary_dim, pos, cfg_.rope_theta);
      }
      for (int h = 0; h < nkv; ++h) {
        float* kh = kk.data() + (t * nkv + h) * hd;
        float* vh = vv.data() + (t * nkv + h) * hd;
        hal::rmsnorm(kh, kn, kh, hd, cfg_.rms_eps, wd_, true);
        hal::apply_rope_freqs(kh, hd, rotary_dim, pos, cfg_.rope_theta);
        // store as [n_kv, seq, hd] for decode helper
        float* kdst = Lkv.k.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq + t) * hd;
        float* vdst = Lkv.v.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq + t) * hd;
        std::memcpy(kdst, kh, sizeof(float) * hd);
        std::memcpy(vdst, vh, sizeof(float) * hd);
      }
    }
    std::vector<float> attn_heads(static_cast<size_t>(n_tok) * nh * hd);
    if (is_prefill && Lkv.seq == 0) {
      // build contiguous k/v [seq,nkv,hd] for prefill
      std::vector<float> kpf(static_cast<size_t>(n_tok) * nkv * hd);
      std::vector<float> vpf(static_cast<size_t>(n_tok) * nkv * hd);
      for (int t = 0; t < n_tok; ++t)
        for (int h = 0; h < nkv; ++h) {
          std::memcpy(kpf.data() + (t * nkv + h) * hd,
                      Lkv.k.data() + (static_cast<size_t>(h) * cache.max_seq() + t) * hd,
                      sizeof(float) * hd);
          std::memcpy(vpf.data() + (t * nkv + h) * hd,
                      Lkv.v.data() + (static_cast<size_t>(h) * cache.max_seq() + t) * hd,
                      sizeof(float) * hd);
        }
      if (!hal::cuda::try_attn_prefill(qq.data(), kpf.data(), vpf.data(), attn_heads.data(), n_tok,
                                       nh, nkv, hd, scale)) {
        hal::attn_prefill(qq.data(), kpf.data(), vpf.data(), attn_heads.data(), n_tok, nh, nkv, hd,
                          scale);
      }
    } else {
      for (int t = 0; t < n_tok; ++t) {
        const int seq_len = Lkv.seq + t + 1;
        hal::attn_decode_one(qq.data() + t * nh * hd, Lkv.k.data(), Lkv.v.data(),
                             attn_heads.data() + t * nh * hd, nh, nkv, hd, seq_len,
                             cache.max_seq(), scale);
      }
    }
    Lkv.seq += n_tok;
    // gate + o_proj
    const uint16_t* Wo = w(base + "self_attn.o_proj.weight");
    for (int t = 0; t < n_tok; ++t) {
      for (int i = 0; i < nh * hd; ++i)
        attn_heads[t * nh * hd + i] *= sigmoid(gate[t * nh * hd + i]);
      hal::gemm_bias_free(attn_heads.data() + t * nh * hd, Wo, attn_out.data() + t * H, H, nh * hd, wd_);
    }
  } else {
    // Gated DeltaNet linear attention
    const int nk = cfg_.linear_num_k, nv = cfg_.linear_num_v;
    const int dk = cfg_.linear_dk, dv = cfg_.linear_dv;
    const int key_dim = nk * dk, value_dim = nv * dv;
    const int conv_dim = key_dim * 2 + value_dim;
    std::vector<float> mixed(static_cast<size_t>(n_tok) * conv_dim);
    std::vector<float> z(static_cast<size_t>(n_tok) * value_dim);
    std::vector<float> b(static_cast<size_t>(n_tok) * nv);
    std::vector<float> a(static_cast<size_t>(n_tok) * nv);
    const uint16_t* Wqkv = w(base + "linear_attn.in_proj_qkv.weight");
    const uint16_t* Wz = w(base + "linear_attn.in_proj_z.weight");
    const uint16_t* Wb = w(base + "linear_attn.in_proj_b.weight");
    const uint16_t* Wa = w(base + "linear_attn.in_proj_a.weight");
    const uint16_t* A_log = w(base + "linear_attn.A_log");
    const uint16_t* dt_bias = w(base + "linear_attn.dt_bias");
    const uint16_t* conv_w = w(base + "linear_attn.conv1d.weight");  // [C,1,K]
    const uint16_t* nrm = w(base + "linear_attn.norm.weight");
    const uint16_t* Wout = w(base + "linear_attn.out_proj.weight");

    for (int t = 0; t < n_tok; ++t) {
      const float* xt = normed.data() + t * H;
      hal::gemm_bias_free(xt, Wqkv, mixed.data() + t * conv_dim, conv_dim, H, wd_);
      hal::gemm_bias_free(xt, Wz, z.data() + t * value_dim, value_dim, H, wd_);
      hal::gemm_bias_free(xt, Wb, b.data() + t * nv, nv, H, wd_);
      hal::gemm_bias_free(xt, Wa, a.data() + t * nv, nv, H, wd_);
    }

    // causal depthwise conv1d + silu, with state
    std::vector<float> mixed_c(static_cast<size_t>(n_tok) * conv_dim);
    auto& conv_state = Lkv.linear.conv;
    for (int t = 0; t < n_tok; ++t) {
      for (int c = 0; c < conv_dim; ++c) {
        // shift state
        for (int k = cfg_.conv_k - 1; k > 0; --k)
          conv_state[c * cfg_.conv_k + k] = conv_state[c * cfg_.conv_k + k - 1];
        conv_state[c * cfg_.conv_k] = mixed[t * conv_dim + c];
        double acc = 0.0;
        for (int k = 0; k < cfg_.conv_k; ++k) {
          // weight layout [C,1,K] -> index c*K + k (k=0 is current)
          const float wk = hal::load_w(conv_w + c * cfg_.conv_k + (cfg_.conv_k - 1 - k), wd_);
          acc += static_cast<double>(conv_state[c * cfg_.conv_k + k]) * wk;
        }
        const float y = static_cast<float>(acc);
        mixed_c[t * conv_dim + c] = y / (1.f + std::exp(-y));  // silu
      }
    }
    Lkv.linear.has_state = true;

    // split q,k,v
    std::vector<float> q(static_cast<size_t>(n_tok) * nv * dk);  // after repeat
    std::vector<float> k(static_cast<size_t>(n_tok) * nv * dk);
    std::vector<float> v(static_cast<size_t>(n_tok) * nv * dv);
    std::vector<float> g(static_cast<size_t>(n_tok) * nv);
    std::vector<float> beta(static_cast<size_t>(n_tok) * nv);
    const int rep = nv / nk;
    for (int t = 0; t < n_tok; ++t) {
      const float* m = mixed_c.data() + t * conv_dim;
      for (int h = 0; h < nk; ++h) {
        for (int r = 0; r < rep; ++r) {
          const int hh = h * rep + r;
          std::memcpy(q.data() + (t * nv + hh) * dk, m + h * dk, sizeof(float) * dk);
          std::memcpy(k.data() + (t * nv + hh) * dk, m + key_dim + h * dk, sizeof(float) * dk);
        }
      }
      std::memcpy(v.data() + t * value_dim, m + 2 * key_dim, sizeof(float) * value_dim);
      for (int h = 0; h < nv; ++h) {
        beta[t * nv + h] = sigmoid(b[t * nv + h]);
        float A = std::exp(hal::load_w(A_log + h, wd_));
        if (!std::isfinite(A) || A > 1e4f) A = 1e4f;
        if (A < 1e-6f) A = 1e-6f;
        float sp = softplus(a[t * nv + h] + hal::load_w(dt_bias + h, wd_));
        if (!std::isfinite(sp)) sp = 0.f;
        g[t * nv + h] = -A * sp;
      }
    }

    std::vector<float> core(static_cast<size_t>(n_tok) * value_dim);
    hal::gated_delta_recurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
                               Lkv.linear.recurrent.data(), core.data(), n_tok, nv, dk, dv, true);

    for (int t = 0; t < n_tok; ++t) {
      for (int h = 0; h < nv; ++h) {
        float* ch = core.data() + (t * nv + h) * dv;
        float* zh = z.data() + (t * nv + h) * dv;
        hal::rmsnorm_gated(ch, zh, nrm, ch, dv, cfg_.rms_eps, wd_);
      }
      hal::gemm_bias_free(core.data() + t * value_dim, Wout, attn_out.data() + t * H, H, value_dim, wd_);
    }
  }

  // residual
  for (size_t i = 0; i < residual.size(); ++i) x[i] = residual[i] + attn_out[i];

  // MLP
  std::memcpy(residual.data(), x, sizeof(float) * n_tok * H);
  const uint16_t* ln2 = w(base + "post_attention_layernorm.weight");
  const uint16_t* Wg = w(base + "mlp.gate_proj.weight");
  const uint16_t* Wu = w(base + "mlp.up_proj.weight");
  const uint16_t* Wd = w(base + "mlp.down_proj.weight");
  std::vector<float> gproj(I), uproj(I), mid(I), down(H);
  for (int t = 0; t < n_tok; ++t) {
    hal::rmsnorm(x + t * H, ln2, normed.data() + t * H, H, cfg_.rms_eps, wd_, true);
    hal::gemm_bias_free(normed.data() + t * H, Wg, gproj.data(), I, H, wd_);
    hal::gemm_bias_free(normed.data() + t * H, Wu, uproj.data(), I, H, wd_);
    hal::silu_and_mul(gproj.data(), uproj.data(), mid.data(), I);
    hal::gemm_bias_free(mid.data(), Wd, down.data(), H, I, wd_);
    for (int i = 0; i < H; ++i) x[t * H + i] = residual[t * H + i] + down[i];
  }
}

void Qwen35Model::forward(const std::vector<int32_t>& tokens, SessionCache& cache,
                          std::vector<float>& logits, bool is_prefill) {
  std::vector<float> all;
  forward_all_logits(tokens, cache, all, is_prefill);
  const int V = cfg_.vocab;
  const int n = static_cast<int>(tokens.size());
  logits.assign(all.begin() + static_cast<size_t>(n - 1) * V, all.end());
}

void Qwen35Model::forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                                     std::vector<float>& logits_all, bool is_prefill) {
  if (!wm_ || tokens.empty()) throw std::runtime_error("model not ready / empty tokens");
  const int n = static_cast<int>(tokens.size());
  const int H = cfg_.hidden;
  const int V = cfg_.vocab;
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  double ms_full = 0, ms_lin = 0;
  std::vector<float> x(static_cast<size_t>(n) * H);
  int pos_start = 0;
  if (!is_prefill) {
    for (int i = 0; i < cfg_.layers; ++i)
      if (cfg_.layer_types[i] == "full_attention") {
        pos_start = cache.layer(i).seq;
        break;
      }
  }
  for (int t = 0; t < n; ++t) embed(tokens[t], x.data() + t * H);

  for (int L = 0; L < cfg_.layers; ++L) {
    const auto tl = Clock::now();
    layer_forward(L, x.data(), cache, pos_start, n, is_prefill);
    if (llmoc::log::profile_enabled()) {
      const double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - tl).count();
      if (cfg_.layer_types[L] == "full_attention") ms_full += ms;
      else ms_lin += ms;
    }
  }

  const auto th0 = Clock::now();
  const uint16_t* fn = w(prefix_ + "norm.weight");
  const std::string emb_name =
      prefix_.empty() ? "embedding.weight" : prefix_ + "embed_tokens.weight";
  const uint16_t* emb = w(emb_name);
  const uint16_t* lm = lm_head_name_.empty() ? emb : w(lm_head_name_);
  logits_all.assign(static_cast<size_t>(n) * V, 0.f);
  prefix_hiddens_.assign(static_cast<size_t>(n) * H, 0.f);
  std::vector<float> h(H);
  last_hidden_.assign(H, 0.f);
  for (int t = 0; t < n; ++t) {
    hal::rmsnorm(x.data() + t * H, fn, h.data(), H, cfg_.rms_eps, wd_, true);
    std::memcpy(prefix_hiddens_.data() + static_cast<size_t>(t) * H, h.data(),
                sizeof(float) * H);
    float* dest = logits_all.data() + static_cast<size_t>(t) * V;
    // HF Linear: y = x @ W^T，本仓库 gemm 约定 W 为 [out, in] row-major
    hal::gemm_bias_free(h.data(), lm, dest, V, H, wd_);
    if (t == n - 1) last_hidden_ = h;
  }
  prefix_logits_ = logits_all;
  last_logits_.assign(logits_all.begin() + static_cast<size_t>(n - 1) * V, logits_all.end());

  if (llmoc::log::profile_enabled()) {
    const double ms_head =
        std::chrono::duration<double, std::milli>(Clock::now() - th0).count();
    const double ms_all =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    LOG_INFO(
        "profile forward: n_tok=%d prefill=%d full_layers=%.1fms linear_layers=%.1fms lm_head=%.1fms "
        "total=%.1fms",
        n, is_prefill ? 1 : 0, ms_full, ms_lin, ms_head, ms_all);
  }
}

void Qwen35Model::commit_prefix_state(int pos) {
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

bool Qwen35Model::mtp_has_cb(void* ctx, const std::string& name) {
  return static_cast<Qwen35Model*>(ctx)->wm_->header().find(name) != nullptr;
}
void Qwen35Model::mtp_gemm_cb(void* ctx, const float* x, const std::string& wname, float* y, int M,
                              int K) {
  auto* self = static_cast<Qwen35Model*>(ctx);
  hal::gemm_bias_free(x, self->w(wname), y, M, K, self->wd_);
}
const uint16_t* Qwen35Model::mtp_pass_cb(void* ctx, const std::string& name) {
  return static_cast<Qwen35Model*>(ctx)->w(name);
}
void Qwen35Model::mtp_embed_cb(void* ctx, int32_t token, float* out) {
  static_cast<Qwen35Model*>(ctx)->embed(token, out);
}
float Qwen35Model::mtp_embed_dot_cb(void* ctx, const float* h, int32_t token) {
  auto* self = static_cast<Qwen35Model*>(ctx);
  std::vector<float> row(self->cfg_.hidden);
  self->embed(token, row.data());
  float acc = 0.f;
  for (int i = 0; i < self->cfg_.hidden; ++i) acc += h[i] * row[i];
  return acc;
}

MtpWeightAccess Qwen35Model::mtp_access() {
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
  wa.pass_dt = wd_;
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

bool Qwen35Model::has_mtp() const {
  if (!wm_) return false;
  return mtp_weights_present(const_cast<Qwen35Model*>(this)->mtp_access());
}

bool Qwen35Model::draft_propose(const std::vector<int32_t>& history, int draft_k,
                                std::vector<int32_t>& out, int32_t pin_first) {
  out.clear();
  if (last_hidden_.empty()) return false;
  return mtp_draft_propose(mtp_access(), last_hidden_, history, draft_k, out, pin_first);
}

}  // namespace llmoc::model
