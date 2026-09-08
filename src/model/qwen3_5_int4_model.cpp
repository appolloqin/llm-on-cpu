// llm-on-cpu :: model/qwen3_5_int4_model.cpp (INT4/QLWC; does not modify BF16 path)
#include "model/qwen3_5_int4_model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"
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

// MoE experts: GPU fused path by default when CUDA on.
// LLMOC_MOE_GPU_EXPERTS=0 → force host OpenMP; =1 → force GPU (same as default).
bool moe_gpu_experts_enabled() {
  static const int mode = [] {
    const char* e = std::getenv("LLMOC_MOE_GPU_EXPERTS");
    if (!e || !e[0]) return 1;  // default on
    if (e[0] == '0' && e[1] == '\0') return 0;
    return 1;
  }();
  return mode != 0;
}

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
  cfg_.intermediate = tc.value("intermediate_size", 0);
  cfg_.vocab = tc.value("vocab_size", 248320);
  cfg_.rms_eps = static_cast<float>(tc.value("rms_norm_eps", 1e-6));
  cfg_.tie_embeddings = tc.value("tie_word_embeddings", true);
  cfg_.linear_num_k = tc.value("linear_num_key_heads", 16);
  cfg_.linear_num_v = tc.value("linear_num_value_heads", 32);
  cfg_.linear_dk = tc.value("linear_key_head_dim", 128);
  cfg_.linear_dv = tc.value("linear_value_head_dim", 128);
  cfg_.conv_k = tc.value("linear_conv_kernel_dim", 4);
  cfg_.n_experts = tc.value("num_experts", tc.value("n_routed_experts", 0));
  cfg_.topk = tc.value("num_experts_per_tok", 0);
  cfg_.moe_intermediate = tc.value("moe_intermediate_size", 0);
  cfg_.shared_expert_intermediate =
      tc.value("shared_expert_intermediate_size", cfg_.moe_intermediate);
  cfg_.first_k_dense = tc.value("first_k_dense_replace", 0);
  cfg_.is_moe = cfg_.n_experts > 0;
  if (cfg_.is_moe) {
    if (cfg_.topk <= 0) cfg_.topk = 8;
    if (cfg_.moe_intermediate <= 0) cfg_.moe_intermediate = 512;
    if (cfg_.shared_expert_intermediate <= 0)
      cfg_.shared_expert_intermediate = cfg_.moe_intermediate;
    if (cfg_.intermediate <= 0) cfg_.intermediate = cfg_.moe_intermediate;
  } else if (cfg_.intermediate <= 0) {
    cfg_.intermediate = 9216;
  }
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
  meta_.is_moe = cfg_.is_moe;
  meta_.kind = cfg_.is_moe ? "qwen3_5_moe_int4" : "qwen3_5_int4";

  // Weight-based MoE detect (config may omit num_experts on some AWQ dumps).
  if (!cfg_.is_moe) {
    const std::string probe0 = prefix_ + "layers.0.mlp.experts.0.gate_proj.weight";
    const std::string probe_gate = prefix_ + "layers.0.mlp.gate.weight";
    if (store_->has(probe0) || store_->has(probe_gate)) {
      cfg_.is_moe = true;
      meta_.is_moe = true;
      meta_.kind = "qwen3_5_moe_int4";
      if (cfg_.n_experts <= 0) {
        int max_e = 0;
        const std::string pref = prefix_ + "layers.0.mlp.experts.";
        for (const auto& tm : store_->header().tensors) {
          if (tm.name.rfind(pref, 0) != 0) continue;
          const auto rest = tm.name.substr(pref.size());
          const auto dot = rest.find('.');
          if (dot == std::string::npos) continue;
          try {
            max_e = std::max(max_e, std::stoi(rest.substr(0, dot)) + 1);
          } catch (...) {
          }
        }
        cfg_.n_experts = max_e;
      }
      if (cfg_.topk <= 0) cfg_.topk = 8;
      if (cfg_.moe_intermediate <= 0) cfg_.moe_intermediate = 512;
      if (cfg_.shared_expert_intermediate <= 0)
        cfg_.shared_expert_intermediate = cfg_.moe_intermediate;
      if (cfg_.intermediate <= 0) cfg_.intermediate = cfg_.moe_intermediate;
    }
  }

  // AutoAWQ (zero_point:true) must be imported as gptq_asym. awq_sym drops qzeros → sticky garbage.
  {
    const auto sch = store_->header().scheme;
    nlohmann::json qc = nlohmann::json::object();
    if (root.contains("quantization_config")) qc = root["quantization_config"];
    else if (root.contains("text_config") && root["text_config"].contains("quantization_config"))
      qc = root["text_config"]["quantization_config"];
    const bool hf_zp = qc.value("zero_point", false);
    const std::string qmethod = qc.value("quant_method", "");
    size_t int4_n = 0, zeros_n = 0;
    for (const auto& tm : store_->header().tensors) {
      if (tm.kind != qlwc::TensorKind::kInt4) continue;
      ++int4_n;
      if (tm.zeros_nbytes > 0) ++zeros_n;
    }
    LOG_INFO("Qwen35Int4 qlwc: scheme=%s int4=%zu with_zeros=%zu hf_quant=%s zero_point=%d",
             sch == qlwc::Scheme::kAwqSym ? "awq_sym" : "gptq_asym", int4_n, zeros_n, qmethod.c_str(),
             hf_zp ? 1 : 0);
    if (cfg_.is_moe && hf_zp && sch == qlwc::Scheme::kAwqSym) {
      throw std::runtime_error(
          "MoE QLWC scheme=awq_sym but HF quantization_config.zero_point=true — qzeros were "
          "dropped at import, decode collapses (ici/endah/…). Re-import with current "
          "tools/import_awq_hf_qlwc.mjs (expect scheme=gptq), then: node tools/qlwc_info.mjs "
          "<file.qlwc>");
    }
    if (cfg_.is_moe && sch == qlwc::Scheme::kGptqAsym && zeros_n == 0 && int4_n > 0) {
      throw std::runtime_error(
          "MoE QLWC scheme=gptq_asym but no zeros blobs — corrupt import; re-run import_awq_hf_qlwc");
    }
  }

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

  if (cfg_.is_moe && !allow_moe_) {
    throw std::runtime_error(
        "QLWC/HF looks like MoE; use Qwen36MoeInt4Model (dense Qwen35Int4Model refuses MoE to keep boundaries)");
  }
  build_layer_packs();
}

void Qwen35Int4Model::build_layer_packs() {
  layers_.assign(cfg_.layers, {});
  // 仅 layer_stream（有 streamer_）延迟装层；QlwcStore::lazy 仍须在此 fill（ensure 按需读盘）。
  if (streamer_) {
    build_global_packs();
    LOG_INFO("Qwen35Int4: layers=%d hidden=%d heads=%d lin_v=%d tie=%d (layer_stream deferred pack)",
             cfg_.layers, cfg_.hidden, cfg_.n_heads, cfg_.linear_num_v,
             cfg_.tie_embeddings ? 1 : 0);
    return;
  }
  for (int L = 0; L < cfg_.layers; ++L) {
    fill_layer_pack(L);
    if ((L + 1) % 10 == 0 || L + 1 == cfg_.layers) {
      LOG_INFO("Qwen35Int4: fill_layer_pack %d/%d lazy_store=%d", L + 1, cfg_.layers,
               store_->lazy() ? 1 : 0);
    }
  }
  build_global_packs();
  LOG_INFO("Qwen35Int4: layers=%d hidden=%d heads=%d lin_v=%d tie=%d moe=%d experts=%d topk=%d",
           cfg_.layers, cfg_.hidden, cfg_.n_heads, cfg_.linear_num_v, cfg_.tie_embeddings ? 1 : 0,
           cfg_.is_moe ? 1 : 0, cfg_.n_experts, cfg_.topk);
  if (cfg_.is_moe) {
    const auto sch = store_->header().scheme;
    const char* sch_s = sch == qlwc::Scheme::kAwqSym ? "awq_sym_zp7" : "gptq_asym";
    LOG_INFO(
        "Qwen35Int4 MoE path: gpu_experts=%d stream_act=0 qlwc_scheme=%s; "
        "LLMOC_MOE_GPU_EXPERTS=0 forces host experts",
        moe_gpu_experts_enabled() ? 1 : 0, sch_s);
    if (sch == qlwc::Scheme::kAwqSym) {
      LOG_INFO(
          "Qwen35Int4 MoE note: AutoAWQ with zero_point:true must be imported as gptq_asym "
          "(re-run import_awq_hf_qlwc); awq_sym drops qzeros and collapses decode");
    }
  }
}

void Qwen35Int4Model::build_global_packs() {
  const std::string emb_name =
      prefix_.empty() ? "embedding.weight" : prefix_ + "embed_tokens.weight";
  if (store_->lazy()) store_->ensure(emb_name);
  emb_is_int4_ = is_int4(emb_name);
  if (emb_is_int4_) {
    emb_int4_ = store_->get_int4(emb_name);
  } else {
    const auto pv = pass_view(emb_name);
    emb_pass_ = pv.data;
    emb_dt_ = pass_to_wd(pv.dtype);
  }

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
    if (store_->lazy()) store_->ensure(lm_name);
    lm_is_int4_ = is_int4(lm_name);
    if (lm_is_int4_) {
      lm_int4_ = store_->get_int4(lm_name);
    } else {
      const auto pv = pass_view(lm_name);
      lm_pass_ = pv.data;
      lm_dt_ = pass_to_wd(pv.dtype);
    }
  } else {
    lm_is_int4_ = emb_is_int4_;
    lm_int4_ = emb_int4_;
    lm_pass_ = emb_pass_;
    lm_dt_ = emb_dt_;
  }

  const std::string fn = prefix_ + "norm.weight";
  {
    const auto pv = pass_view(fn);
    final_norm_ = pv.data;
    final_norm_dt_ = pass_to_wd(pv.dtype);
  }
  LOG_INFO("Qwen35Int4 pass dtypes: emb=%s lm=%s final_norm=%s catalog_first=%s",
           emb_is_int4_ ? "int4" : (emb_dt_ == hal::WDtype::kF16 ? "f16" : "bf16"),
           lm_is_int4_ ? "int4" : (lm_dt_ == hal::WDtype::kF16 ? "f16" : "bf16"),
           final_norm_dt_ == hal::WDtype::kF16 ? "f16" : "bf16",
           pass_wd_ == hal::WDtype::kF16 ? "f16" : "bf16");
}

void Qwen35Int4Model::fill_layer_pack(int L) {
  auto& lp = layers_[L];
  const int nk = cfg_.linear_num_k, nv = cfg_.linear_num_v;
  const int dk = cfg_.linear_dk, dv = cfg_.linear_dv;
  const int conv_dim = nk * dk * 2 + nv * dv;
  const std::string base = prefix_ + "layers." + std::to_string(L) + ".";
  lp = {};
  lp.is_full = (cfg_.layer_types[L] == "full_attention");
  {
    const auto p1 = pass_view(base + "input_layernorm.weight");
    const auto p2 = pass_view(base + "post_attention_layernorm.weight");
    lp.ln1 = p1.data;
    lp.ln2 = p2.data;
    lp.ln_dt = pass_to_wd(p1.dtype);
  }
  if (lp.is_full) {
    const int nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
    lp.wq = load_opt_w(base + "self_attn.q_proj.weight", nh * hd * 2, cfg_.hidden);
    lp.wk = load_opt_w(base + "self_attn.k_proj.weight", nkv * hd, cfg_.hidden);
    lp.wv = load_opt_w(base + "self_attn.v_proj.weight", nkv * hd, cfg_.hidden);
    lp.wo = load_opt_w(base + "self_attn.o_proj.weight", cfg_.hidden, nh * hd);
    const auto pq = pass_view(base + "self_attn.q_norm.weight");
    const auto pk = pass_view(base + "self_attn.k_norm.weight");
    lp.qn = pq.data;
    lp.kn = pk.data;
    lp.qk_norm_dt = pass_to_wd(pq.dtype);
  } else {
    const int value_dim = nv * dv;
    const int conv_dim_w = nk * dk * 2 + nv * dv;
    lp.wqkv = load_opt_w(base + "linear_attn.in_proj_qkv.weight", conv_dim_w, cfg_.hidden);
    lp.wz = load_opt_w(base + "linear_attn.in_proj_z.weight", value_dim, cfg_.hidden);
    // cyankiwi AWQ: in_proj_a/b 常在 ignore 中保持 BF16；out_proj 仅部分层量化
    lp.wb = load_opt_w(base + "linear_attn.in_proj_b.weight", nv, cfg_.hidden);
    lp.wa = load_opt_w(base + "linear_attn.in_proj_a.weight", nv, cfg_.hidden);
    lp.wout = load_opt_w(base + "linear_attn.out_proj.weight", cfg_.hidden, value_dim);
    const auto pn = pass_view(base + "linear_attn.norm.weight");
    lp.nrm = pn.data;
    lp.nrm_dt = pass_to_wd(pn.dtype);
    {
      const std::string a_name = base + "linear_attn.A_log";
      const std::string d_name = base + "linear_attn.dt_bias";
      const std::string c_name = base + "linear_attn.conv1d.weight";
      if (store_->lazy()) {
        store_->ensure(a_name);
        store_->ensure(d_name);
        store_->ensure(c_name);
      }
      const auto pa = store_->get_pass(a_name);
      const auto pd = store_->get_pass(d_name);
      const auto pc = store_->get_pass(c_name);
      const auto dta = pa.dtype == qlwc::PassDtype::kF16 ?hal::WDtype::kF16 :hal::WDtype::kBF16;
      const auto dtd = pd.dtype == qlwc::PassDtype::kF16 ?hal::WDtype::kF16 :hal::WDtype::kBF16;
      const auto dtc = pc.dtype == qlwc::PassDtype::kF16 ?hal::WDtype::kF16 :hal::WDtype::kBF16;
      lp.A_log_f.resize(nv);
      lp.dt_bias_f.resize(nv);
      for (int h = 0; h < nv; ++h) {
        lp.A_log_f[h] = hal::load_w(pa.data + h, dta);
        lp.dt_bias_f[h] = hal::load_w(pd.data + h, dtd);
      }
      lp.conv_w_f.resize(static_cast<size_t>(conv_dim) * cfg_.conv_k);
      for (int c = 0; c < conv_dim; ++c)
        for (int k = 0; k < cfg_.conv_k; ++k)
          lp.conv_w_f[c * cfg_.conv_k + k] =
            hal::load_w(pc.data + c * cfg_.conv_k + k, dtc);
    }
  }

  const bool has_dense = store_->has(base + "mlp.gate_proj.weight");
  const bool has_router = store_->has(base + "mlp.gate.weight");
  const bool has_expert0 = store_->has(base + "mlp.experts.0.gate_proj.weight");
  const bool use_dense =
      has_dense && (L < cfg_.first_k_dense || (!has_router && !has_expert0));
  if (use_dense) {
    if (store_->lazy()) {
      store_->ensure(base + "mlp.gate_proj.weight");
      store_->ensure(base + "mlp.up_proj.weight");
      store_->ensure(base + "mlp.down_proj.weight");
    }
    lp.wgate = store_->get_int4(base + "mlp.gate_proj.weight");
    lp.wup = store_->get_int4(base + "mlp.up_proj.weight");
    lp.wdown = store_->get_int4(base + "mlp.down_proj.weight");
    return;
  }

  if (!has_expert0) {
    if (store_->has(base + "mlp.experts.gate_up_proj") ||
        store_->has(base + "mlp.experts.gate_up_proj.weight")) {
      throw std::runtime_error(
          "QLWC has fused mlp.experts.gate_up_proj (3D); re-import AWQ with per-expert "
          "2D gate/up/down_proj: " +
          base);
    }
    throw std::runtime_error("MoE layer missing mlp.experts.0.gate_proj.weight: " + base);
  }
  if (!has_router) throw std::runtime_error("MoE layer missing mlp.gate.weight: " + base);
  lp.is_moe = true;
  const int E = cfg_.n_experts > 0 ? cfg_.n_experts : 1;
  const int Is = cfg_.shared_expert_intermediate > 0 ? cfg_.shared_expert_intermediate
                                                     : cfg_.moe_intermediate;
  lp.router = load_opt_w(base + "mlp.gate.weight", E, cfg_.hidden);
  const std::string sg = base + "mlp.shared_expert.gate_proj.weight";
  const std::string su = base + "mlp.shared_expert.up_proj.weight";
  const std::string sd = base + "mlp.shared_expert.down_proj.weight";
  const std::string sgg = base + "mlp.shared_expert_gate.weight";
  if (store_->has(sg) && store_->has(su) && store_->has(sd)) {
    lp.shared_gate = load_opt_w(sg, Is, cfg_.hidden);
    lp.shared_up = load_opt_w(su, Is, cfg_.hidden);
    lp.shared_down = load_opt_w(sd, cfg_.hidden, Is);
  }
  if (store_->has(sgg)) lp.shared_expert_gate = load_opt_w(sgg, 1, cfg_.hidden);
}

void Qwen35Int4Model::enable_layer_stream(wt::ILayerStreamLoader* loader) {
  streamer_ = loader;
  stream_window_ = 2;
  if (streamer_) {
    auto st = streamer_->stats();
    (void)st;
    LOG_INFO("Qwen35Int4: layer_stream enabled");
  }
}

void Qwen35Int4Model::init_cache(SessionCache& cache, int max_seq) const {
  std::vector<uint8_t> need_kv(static_cast<size_t>(cfg_.layers), 1);
  for (int i = 0; i < cfg_.layers; ++i)
    need_kv[static_cast<size_t>(i)] = (cfg_.layer_types[i] == "full_attention") ? 1 : 0;
  cache.init(cfg_.layers, max_seq, cfg_.n_kv, cfg_.head_dim, cfg_.linear_num_v, cfg_.linear_dk,
             cfg_.linear_dv, meta_.conv_dim, cfg_.conv_k, &need_kv);
}

bool Qwen35Int4Model::is_int4(const std::string& name) const {
  return store_->is_int4(name);
}

const uint16_t* Qwen35Int4Model::pass(const std::string& name) {
  return pass_view(name).data;
}

qlwc::PassView Qwen35Int4Model::pass_view(const std::string& name) {
  if (store_->lazy()) store_->ensure(name);
  return store_->get_pass(name);
}

Qwen35Int4Model::OptW Qwen35Int4Model::load_opt_w(const std::string& name, int M, int K) {
  OptW o;
  o.M = M;
  o.K = K;
  if (store_->lazy()) store_->ensure(name);
  if (is_int4(name)) {
    o.is_int4 = true;
    o.i4 = store_->get_int4(name);
  } else {
    o.is_int4 = false;
    const auto pv = store_->get_pass(name);
    o.pass = pv.data;
    o.dt = pv.dtype == qlwc::PassDtype::kF16 ? hal::WDtype::kF16 : hal::WDtype::kBF16;
  }
  return o;
}

void Qwen35Int4Model::gemm_w(const float* x, const std::string& wname, float* y, int M, int K) {
  if (is_int4(wname)) {
    if (store_->lazy()) store_->ensure(wname);
    const qlwc::Int4View W = store_->get_int4(wname);
    const bool allow_gpu = !cfg_.is_moe || moe_gpu_experts_enabled();
    if (allow_gpu && hal::cuda::enabled() && hal::cuda::try_gemm_int4(x, W, y)) return;
    hal::gemm_int4(x, W, y);
  } else {
    (void)M;
    (void)K;
    const auto pv = pass_view(wname);
    hal::gemm_bias_free(x, pv.data, y, M, K, pass_to_wd(pv.dtype));
  }
}

void Qwen35Int4Model::gemm_view(const float* x, const qlwc::Int4View& W, float* y) {
  if (hal::cuda::enabled() && hal::cuda::try_gemm_int4(x, W, y)) return;
  hal::gemm_int4(x, W, y);
}

void Qwen35Int4Model::gemm_view_batch(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  if (n <= 0) return;
  if (hal::cuda::enabled() && hal::cuda::try_gemm_int4_batch(X, n, W, Y)) return;
  if (n <= 1) {
    if (n == 1) hal::gemm_int4(X, W, Y);
    return;
  }
  hal::gemm_int4_batch(X, n, W, Y);
}

void Qwen35Int4Model::gemm_opt(const float* x, const OptW& W, float* y) {
  if (W.is_int4) {
    gemm_view(x, W.i4, y);
    return;
  }
  if (!W.pass) throw std::runtime_error("gemm_opt: missing passthrough weight");
  if (hal::cuda::enabled() && hal::cuda::try_gemm_w16(x, W.pass, y, W.M, W.K, W.dt == hal::WDtype::kF16)) {
    return;
  }
  hal::gemm_bias_free(x, W.pass, y, W.M, W.K, W.dt);
}

void Qwen35Int4Model::gemm_opt_batch(const float* X, int n, const OptW& W, float* Y) {
  if (n <= 0) return;
  if (W.is_int4) {
    gemm_view_batch(X, n, W.i4, Y);
    return;
  }
  if (!W.pass) throw std::runtime_error("gemm_opt_batch: missing passthrough weight");
  if (n == 1) {
    gemm_opt(X, W, Y);
    return;
  }
  // Weight-stationary batch (serial gemm_opt re-reads full BF16 out_proj every token).
  if (hal::cuda::enabled() && hal::cuda::try_gemm_w16_batch(X, n, W.pass, Y, W.M, W.K, W.dt == hal::WDtype::kF16)) {
    return;
  }
  hal::gemm_bias_free_batch(X, n, W.pass, Y, W.M, W.K, W.dt);
}

void Qwen35Int4Model::warm_gpu_int4_weights(int* out_ok, int* out_fail) {
  if (out_ok) *out_ok = 0;
  if (out_fail) *out_fail = 0;
  if (!hal::cuda::enabled()) return;
  int n_ok = 0, n_fail = 0;
  auto try_one = [&](const qlwc::Int4View& W, bool pin = false) {
    if (!W.qweight || W.M <= 0 || W.K <= 0) return;
    const bool ok = pin ?hal::cuda::pin_int4_weight(W) :hal::cuda::prefetch_int4_weight(W);
    if (ok)
      ++n_ok;
    else
      ++n_fail;
  };
  for (const auto& lp : layers_) {
    auto try_pass = [&](const OptW& W) {
      if (W.is_int4) {
        try_one(W.i4, /*pin=*/true);
        return;
      }
      if (!W.pass || W.M <= 0 || W.K <= 0) return;
      if (hal::cuda::prefetch_w16(W.pass, W.M, W.K, W.dt == hal::WDtype::kF16))
        ++n_ok;
      else
        ++n_fail;
    };
    if (lp.is_full) {
      try_pass(lp.wq);
      try_pass(lp.wk);
      try_pass(lp.wv);
      try_pass(lp.wo);
    } else {
      try_pass(lp.wqkv);
      try_pass(lp.wz);
      try_pass(lp.wb);
      try_pass(lp.wa);
      try_pass(lp.wout);
    }
    if (!lp.is_moe) {
      try_one(lp.wgate, true);
      try_one(lp.wup, true);
      try_one(lp.wdown, true);
    } else {
      if (lp.router.is_int4) try_one(lp.router.i4, true);
      if (lp.shared_gate.is_int4) try_one(lp.shared_gate.i4, true);
      if (lp.shared_up.is_int4) try_one(lp.shared_up.i4, true);
      if (lp.shared_down.is_int4) try_one(lp.shared_down.i4, true);
    }
  }
  // lm_head resident: INT4 → ensure_int4_resident; BF16 pass → packed W16 (no FP32 inflate)
  LOG_INFO("Qwen35Int4: warm_gpu_int4 ok=%d fail=%d used=%.2fGiB / budget=%.2fGiB (lm_int4=%d lm_pass=%d tie=%d)",
           n_ok, n_fail, hal::cuda::vram_used() / double(1ull << 30),
           hal::cuda::vram_budget() / double(1ull << 30), lm_is_int4_ ? 1 : 0,
           lm_pass_ ? 1 : 0, cfg_.tie_embeddings ? 1 : 0);
  if (lm_is_int4_) {
    try_one(lm_int4_);
  } else if (lm_pass_) {
    const size_t mbytes = static_cast<size_t>(cfg_.vocab) * cfg_.hidden * 2;
    if (hal::cuda::prefetch_w16(lm_pass_, cfg_.vocab, cfg_.hidden,
                                lm_dt_ == hal::WDtype::kF16)) {
      LOG_INFO("Qwen35Int4: lm_head W16-pack prefetch OK (%.2fGiB)", mbytes / double(1ull << 30));
    } else {
      LOG_INFO("Qwen35Int4: lm_head W16-pack prefetch FAILED (%.2fGiB needed; used=%.2fGiB budget=%.2fGiB)",
               mbytes / double(1ull << 30), hal::cuda::vram_used() / double(1ull << 30),
               hal::cuda::vram_budget() / double(1ull << 30));
      ++n_fail;
    }
  }
  // embed (tied 时跟 lm 同)
  if (emb_is_int4_ && emb_int4_.qweight && emb_int4_.qweight != lm_int4_.qweight) try_one(emb_int4_);
  hal::cuda::log_status();
  if (out_ok) *out_ok = n_ok;
  if (out_fail) *out_fail = n_fail;
}

size_t Qwen35Int4Model::resident_workspace_bytes() const {
  int n_lin = 0;
  for (const auto& lp : layers_) {
    if (!lp.is_full) ++n_lin;
  }
  const int nv = cfg_.linear_num_v;
  const int dk = cfg_.linear_dk;
  const int dv = cfg_.linear_dv;
  const size_t gdn_states =
      static_cast<size_t>(n_lin) * sizeof(float) * static_cast<size_t>(nv) * dk * dv;
  const size_t gdn_io =
      sizeof(float) * (static_cast<size_t>(nv) * dk * 2 + static_cast<size_t>(nv) * dv +
                       static_cast<size_t>(nv) * 2 + static_cast<size_t>(nv) * dv);
  const size_t act = sizeof(float) * static_cast<size_t>(cfg_.hidden) * 32;
  const size_t margin = 256ull << 20;  // 256 MiB slack
  return gdn_states + gdn_io + act + margin;
}

void Qwen35Int4Model::enable_resident_gpu(bool on) { resident_gpu_ = on; }

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
  for (int i = 0; i < cfg_.hidden; ++i) out[i] = hal::load_w(row + i, emb_dt_);
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

void Qwen35Int4Model::moe_ffn_token(int /*layer*/, const float* /*normed*/, float* /*down_acc*/) {
  throw std::runtime_error("moe_ffn_token: MoE requires Qwen36MoeInt4Model");
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
     hal::rmsnorm(x + t * H, lp.ln1, sc.normed.data() + t * H, H, cfg_.rms_eps, lp.ln_dt, true);

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
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wq, sc.qg.data());
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wk, sc.kk.data());
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wv, sc.vv.data());
    } else if (lp.wq.is_int4 && lp.wk.is_int4 && lp.wv.is_int4) {
      const qlwc::Int4View* ws3[3] = {&lp.wq.i4, &lp.wk.i4, &lp.wv.i4};
      float* ys3[3] = {sc.qg.data(), sc.kk.data(), sc.vv.data()};
      if (!hal::cuda::try_gemm_int4_multi(sc.normed.data(), ws3, ys3, 3)) {
        gemm_opt(sc.normed.data(), lp.wq, sc.qg.data());
        gemm_opt(sc.normed.data(), lp.wk, sc.kk.data());
        gemm_opt(sc.normed.data(), lp.wv, sc.vv.data());
      }
    } else {
      gemm_opt(sc.normed.data(), lp.wq, sc.qg.data());
      gemm_opt(sc.normed.data(), lp.wk, sc.kk.data());
      gemm_opt(sc.normed.data(), lp.wv, sc.vv.data());
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
        hal::rmsnorm(qh, lp.qn, qh, hd, cfg_.rms_eps, lp.qk_norm_dt, true);
        hal::apply_mrope_freqs(qh, hd, rotary_dim, pt, ph, pw, cfg_.rope_theta, mrope_section_,
                               mrope_interleaved_);
      }
      for (int h = 0; h < nkv; ++h) {
        float* kh = sc.kk.data() + (t * nkv + h) * hd;
        float* vh = sc.vv.data() + (t * nkv + h) * hd;
        hal::rmsnorm(kh, lp.kn, kh, hd, cfg_.rms_eps, lp.qk_norm_dt, true);
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
      if (!hal::cuda::try_attn_prefill(sc.qq.data(), sc.kpf.data(), sc.vpf.data(),
                                       sc.attn_heads.data(), n_tok, nh, nkv, hd, scale)) {
        hal::attn_prefill(sc.qq.data(), sc.kpf.data(), sc.vpf.data(), sc.attn_heads.data(), n_tok,
                          nh, nkv, hd, scale);
      }
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
      gemm_opt_batch(sc.attn_heads.data(), n_tok, lp.wo, sc.attn_out.data());
    else
      gemm_opt(sc.attn_heads.data(), lp.wo, sc.attn_out.data());
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
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wqkv, sc.mixed.data());
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wz, sc.z.data());
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wb, sc.b.data());
      gemm_opt_batch(sc.normed.data(), n_tok, lp.wa, sc.a.data());
    } else if (lp.wqkv.is_int4 && lp.wz.is_int4 && lp.wb.is_int4 && lp.wa.is_int4) {
      const qlwc::Int4View* ws4[4] = {&lp.wqkv.i4, &lp.wz.i4, &lp.wb.i4, &lp.wa.i4};
      float* ys4[4] = {sc.mixed.data(), sc.z.data(), sc.b.data(), sc.a.data()};
      if (!hal::cuda::try_gemm_int4_multi(sc.normed.data(), ws4, ys4, 4)) {
        gemm_opt(sc.normed.data(), lp.wqkv, sc.mixed.data());
        gemm_opt(sc.normed.data(), lp.wz, sc.z.data());
        gemm_opt(sc.normed.data(), lp.wb, sc.b.data());
        gemm_opt(sc.normed.data(), lp.wa, sc.a.data());
      }
    } else {
      gemm_opt(sc.normed.data(), lp.wqkv, sc.mixed.data());
      gemm_opt(sc.normed.data(), lp.wz, sc.z.data());
      gemm_opt(sc.normed.data(), lp.wb, sc.b.data());
      gemm_opt(sc.normed.data(), lp.wa, sc.a.data());
    }

    auto& conv_state = Lkv.linear.conv;
    const float* cw = lp.conv_w_f.data();
    const int ck = cfg_.conv_k;
    // Parallelize over channels (state is per-c); tokens stay serial per channel.
    // Old: omp fork inside for(t) → ~T forks/layer and kills prefill.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (conv_dim >= 256 && n_tok >= 1 && !omp_in_parallel())
#endif
    for (int c = 0; c < conv_dim; ++c) {
      float* st = &conv_state[static_cast<size_t>(c) * ck];
      const float* wk = cw + static_cast<size_t>(c) * ck;
      for (int t = 0; t < n_tok; ++t) {
        const float xin = sc.mixed[static_cast<size_t>(t) * conv_dim + c];
        float* xout = &sc.mixed_c[static_cast<size_t>(t) * conv_dim + c];
        if (ck == 4) {
          st[3] = st[2];
          st[2] = st[1];
          st[1] = st[0];
          st[0] = xin;
          const float acc = st[0] * wk[3] + st[1] * wk[2] + st[2] * wk[1] + st[3] * wk[0];
          *xout = acc / (1.f + std::exp(-acc));
        } else {
          for (int k = ck - 1; k > 0; --k) st[k] = st[k - 1];
          st[0] = xin;
          float acc = 0.f;
          for (int k = 0; k < ck; ++k) acc += st[k] * wk[ck - 1 - k];
          *xout = acc / (1.f + std::exp(-acc));
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

    bool gdn_ok = false;
    if (resident_gpu_) {
      // Decode (n=1) and long prefill: keep GDN state on device. Prefill used to force
      // host gated_delta_recurrent for n>1 → ~0.1 layer/s at T~1k on pure_gpu/AWQ.
      // Multi-token windows (MTP verify): sync host←device first so a mid-window GPU fail
      // can safely fall back to CPU from the pre-window state.
      if (n_tok > 1)
        hal::cuda::flush_gdn_state_to_host(Lkv.linear.recurrent.data(), nv, dk, dv);
      gdn_ok = true;
      for (int t = 0; t < n_tok; ++t) {
        if (!hal::cuda::try_gated_delta_gpu(
                sc.q.data() + static_cast<size_t>(t) * nv * dk,
                sc.k.data() + static_cast<size_t>(t) * nv * dk,
                sc.v.data() + static_cast<size_t>(t) * nv * dv, sc.g.data() + t * nv,
                sc.beta.data() + t * nv, Lkv.linear.recurrent.data(),
                sc.core.data() + static_cast<size_t>(t) * nv * dv, nv, dk, dv)) {
          gdn_ok = false;
          break;
        }
      }
      if (!gdn_ok) {
        // Drop partial device window; recompute from host (synced above when n_tok>1).
        if (n_tok == 1)
          hal::cuda::flush_gdn_state_to_host(Lkv.linear.recurrent.data(), nv, dk, dv);
        else
          hal::cuda::invalidate_gdn_state(Lkv.linear.recurrent.data());
        if (is_prefill && !Lkv.linear.has_state)
          std::fill(Lkv.linear.recurrent.begin(), Lkv.linear.recurrent.end(), 0.f);
        hal::gated_delta_recurrent(sc.q.data(), sc.k.data(), sc.v.data(), sc.g.data(),
                                   sc.beta.data(), Lkv.linear.recurrent.data(), sc.core.data(),
                                   n_tok, nv, dk, dv, true);
        gdn_ok = true;
      }
    }
    if (!gdn_ok) {
      hal::gated_delta_recurrent(sc.q.data(), sc.k.data(), sc.v.data(), sc.g.data(), sc.beta.data(),
                                 Lkv.linear.recurrent.data(), sc.core.data(), n_tok, nv, dk, dv,
                                 true);
    }

    for (int t = 0; t < n_tok; ++t) {
      for (int h = 0; h < nv; ++h) {
        float* ch = sc.core.data() + (t * nv + h) * dv;
        float* zh = sc.z.data() + (t * nv + h) * dv;
          hal::rmsnorm_gated(ch, zh, lp.nrm, ch, dv, cfg_.rms_eps, lp.nrm_dt);
      }
    }
    // Fuse wout + MLP on device (linear decode): skip host residual/mlp tail.
    if (n_tok == 1 && !lp.is_moe && resident_gpu_ && lp.wout.is_int4 && lp.ln2 && hal::cuda::try_out_mlp_resident(sc.residual.data(), sc.core.data(), lp.wout.i4, lp.ln2,
                                        lp.wgate, lp.wup, lp.wdown, x, H, I, cfg_.rms_eps,
                                        lp.ln_dt == hal::WDtype::kF16)) {
      return;
    }
    if (n_tok > 1)
      gemm_opt_batch(sc.core.data(), n_tok, lp.wout, sc.attn_out.data());
    else
      gemm_opt(sc.core.data(), lp.wout, sc.attn_out.data());
  }

  for (size_t i = 0; i < nH; ++i) x[i] = sc.residual[i] + sc.attn_out[i];

  std::memcpy(sc.residual.data(), x, sizeof(float) * nH);
  if (lp.is_moe) {
    Int4Scratch::fit(sc.normed, static_cast<size_t>(n_tok) * H);
    Int4Scratch::fit(sc.down, static_cast<size_t>(n_tok) * H);
    for (int t = 0; t < n_tok; ++t) {
     hal::rmsnorm(x + t * H, lp.ln2, sc.normed.data() + t * H, H, cfg_.rms_eps, lp.ln_dt, true);
      moe_ffn_token(layer, sc.normed.data() + t * H, sc.down.data() + t * H);
      for (int i = 0; i < H; ++i) x[t * H + i] = sc.residual[t * H + i] + sc.down[t * H + i];
    }
    return;
  }
  Int4Scratch::fit(sc.gproj, static_cast<size_t>(n_tok) * I);
  Int4Scratch::fit(sc.uproj, static_cast<size_t>(n_tok) * I);
  Int4Scratch::fit(sc.mid, static_cast<size_t>(n_tok) * I);
  Int4Scratch::fit(sc.down, static_cast<size_t>(n_tok) * H);
  if (n_tok == 1 && resident_gpu_ && lp.ln2 && hal::cuda::try_mlp_decode_resident(x, lp.ln2, lp.wgate, lp.wup, lp.wdown, x, H, I,
                                         cfg_.rms_eps, lp.ln_dt == hal::WDtype::kF16)) {
    // x already = residual + down
  } else if (n_tok > 1) {
    for (int t = 0; t < n_tok; ++t)
      hal::rmsnorm(x + t * H, lp.ln2, sc.normed.data() + t * H, H, cfg_.rms_eps, lp.ln_dt, true);
    gemm_view_batch(sc.normed.data(), n_tok, lp.wgate, sc.gproj.data());
    gemm_view_batch(sc.normed.data(), n_tok, lp.wup, sc.uproj.data());
    for (int t = 0; t < n_tok; ++t)
      hal::silu_and_mul(sc.gproj.data() + t * I, sc.uproj.data() + t * I, sc.mid.data() + t * I, I);
    gemm_view_batch(sc.mid.data(), n_tok, lp.wdown, sc.down.data());
    for (int t = 0; t < n_tok; ++t)
      for (int i = 0; i < H; ++i) x[t * H + i] = sc.residual[t * H + i] + sc.down[t * H + i];
  } else {
    for (int t = 0; t < n_tok; ++t)
      hal::rmsnorm(x + t * H, lp.ln2, sc.normed.data() + t * H, H, cfg_.rms_eps, lp.ln_dt, true);
    const qlwc::Int4View* ws2[2] = {&lp.wgate, &lp.wup};
    float* ys2[2] = {sc.gproj.data(), sc.uproj.data()};
    if (!hal::cuda::try_gemm_int4_multi(sc.normed.data(), ws2, ys2, 2)) {
      gemm_view(sc.normed.data(), lp.wgate, sc.gproj.data());
      gemm_view(sc.normed.data(), lp.wup, sc.uproj.data());
    }
    hal::silu_and_mul(sc.gproj.data(), sc.uproj.data(), sc.mid.data(), I);
    gemm_view(sc.mid.data(), lp.wdown, sc.down.data());
    for (int i = 0; i < H; ++i) x[i] = sc.residual[i] + sc.down[i];
  }
}

bool Qwen35Int4Model::layer_forward_linear_act(int layer, SessionCache& cache) {
  if (!resident_gpu_ || !hal::cuda::decode_act_valid()) return false;
  const auto& lp = layers_[layer];
  if (lp.is_moe || lp.is_full || !lp.ln1 || !lp.ln2) return false;

  const int H = cfg_.hidden;
  const int I = cfg_.intermediate;
  const int nk = cfg_.linear_num_k, nv = cfg_.linear_num_v;
  const int dk = cfg_.linear_dk, dv = cfg_.linear_dv;
  const int key_dim = nk * dk, value_dim = nv * dv;
  const int conv_dim = key_dim * 2 + value_dim;
  auto& sc = scratch();
  auto& Lkv = cache.layer(layer);

  const bool ln_f16 = lp.ln_dt == hal::WDtype::kF16;
  if (lp.wqkv.is_int4 && lp.wz.is_int4) {
    const qlwc::Int4View* wb_i4 = lp.wb.is_int4 ? &lp.wb.i4 : nullptr;
    const uint16_t* wb_pass = lp.wb.is_int4 ? nullptr : lp.wb.pass;
    const qlwc::Int4View* wa_i4 = lp.wa.is_int4 ? &lp.wa.i4 : nullptr;
    const uint16_t* wa_pass = lp.wa.is_int4 ? nullptr : lp.wa.pass;
    const qlwc::Int4View* wout_i4 = lp.wout.is_int4 ? &lp.wout.i4 : nullptr;
    const uint16_t* wout_pass = lp.wout.is_int4 ? nullptr : lp.wout.pass;
    if (hal::cuda::try_linear_decode_on_act(
            lp.ln1, lp.wqkv.i4, lp.wz.i4, wb_i4, wb_pass, lp.wb.dt == hal::WDtype::kF16, wa_i4,
            wa_pass, lp.wa.dt == hal::WDtype::kF16, lp.conv_w_f.data(), Lkv.linear.conv.data(),
            cfg_.conv_k, lp.A_log_f.data(), lp.dt_bias_f.data(), Lkv.linear.recurrent.data(), lp.nrm,
            wout_i4, wout_pass, lp.wout.dt == hal::WDtype::kF16, lp.ln2, lp.wgate, lp.wup, lp.wdown,
            nk, nv, dk, dv, I, cfg_.rms_eps, ln_f16, ln_f16)) {
      Lkv.linear.has_state = true;
      return true;
    }
  }

  // Fallback to host sandwich: pull device-owned conv/GDN state back if present.
  hal::cuda::flush_conv_state_to_host(Lkv.linear.conv.data(), conv_dim, cfg_.conv_k);
  hal::cuda::flush_gdn_state_to_host(Lkv.linear.recurrent.data(), nv, dk, dv);
  Int4Scratch::fit(sc.mixed, static_cast<size_t>(conv_dim));
  Int4Scratch::fit(sc.z, static_cast<size_t>(value_dim));
  Int4Scratch::fit(sc.b, static_cast<size_t>(nv));
  Int4Scratch::fit(sc.a, static_cast<size_t>(nv));
  Int4Scratch::fit(sc.mixed_c, static_cast<size_t>(conv_dim));
  Int4Scratch::fit(sc.q, static_cast<size_t>(nv) * dk);
  Int4Scratch::fit(sc.k, static_cast<size_t>(nv) * dk);
  Int4Scratch::fit(sc.v, static_cast<size_t>(nv) * dv);
  Int4Scratch::fit(sc.g, static_cast<size_t>(nv));
  Int4Scratch::fit(sc.beta, static_cast<size_t>(nv));
  Int4Scratch::fit(sc.core, static_cast<size_t>(value_dim));
  Int4Scratch::fit(sc.normed, static_cast<size_t>(H));

  bool in_ok = false;
  if (lp.wqkv.is_int4 && lp.wz.is_int4 && lp.wb.is_int4 && lp.wa.is_int4) {
    const qlwc::Int4View* ws4[4] = {&lp.wqkv.i4, &lp.wz.i4, &lp.wb.i4, &lp.wa.i4};
    float* ys4[4] = {sc.mixed.data(), sc.z.data(), sc.b.data(), sc.a.data()};
    in_ok = hal::cuda::try_rmsnorm_gemm_multi_from_act(lp.ln1, ws4, ys4, 4, cfg_.rms_eps, ln_f16);
  } else if (lp.wqkv.is_int4 && lp.wz.is_int4) {
    const qlwc::Int4View* ws2[2] = {&lp.wqkv.i4, &lp.wz.i4};
    float* ys2[2] = {sc.mixed.data(), sc.z.data()};
    if (hal::cuda::try_rmsnorm_gemm_multi_from_act(lp.ln1, ws2, ys2, 2, cfg_.rms_eps, ln_f16)) {
      if (!hal::cuda::decode_act_sync_to_host(sc.normed.data(), H)) return false;
      hal::rmsnorm(sc.normed.data(), lp.ln1, sc.normed.data(), H, cfg_.rms_eps, lp.ln_dt, true);
      gemm_opt(sc.normed.data(), lp.wb, sc.b.data());
      gemm_opt(sc.normed.data(), lp.wa, sc.a.data());
      in_ok = true;
    }
  }
  if (!in_ok) return false;

  auto& conv_state = Lkv.linear.conv;
  const float* cw = lp.conv_w_f.data();
  const int ck = cfg_.conv_k;
  const float* xin = sc.mixed.data();
  float* xout = sc.mixed_c.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (conv_dim >= 1024 && !omp_in_parallel())
#endif
  for (int c = 0; c < conv_dim; ++c) {
    float* st = &conv_state[static_cast<size_t>(c) * ck];
    const float* wk = cw + static_cast<size_t>(c) * ck;
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
  Lkv.linear.has_state = true;

  const int rep = nv / nk;
  const float* mc = sc.mixed_c.data();
  for (int h = 0; h < nk; ++h) {
    for (int r = 0; r < rep; ++r) {
      const int hh = h * rep + r;
      std::memcpy(sc.q.data() + hh * dk, mc + h * dk, sizeof(float) * dk);
      std::memcpy(sc.k.data() + hh * dk, mc + key_dim + h * dk, sizeof(float) * dk);
    }
  }
  std::memcpy(sc.v.data(), mc + 2 * key_dim, sizeof(float) * value_dim);
  for (int h = 0; h < nv; ++h) {
    sc.beta[h] = sigmoid(sc.b[h]);
    float A = std::exp(lp.A_log_f[h]);
    if (!std::isfinite(A) || A > 1e4f) A = 1e4f;
    if (A < 1e-6f) A = 1e-6f;
    float sp = softplus(sc.a[h] + lp.dt_bias_f[h]);
    if (!std::isfinite(sp)) sp = 0.f;
    sc.g[h] = -A * sp;
  }

  if (!hal::cuda::try_gated_delta_gpu(sc.q.data(), sc.k.data(), sc.v.data(), sc.g.data(),
                                      sc.beta.data(), Lkv.linear.recurrent.data(), sc.core.data(),
                                      nv, dk, dv)) {
    hal::cuda::flush_gdn_state_to_host(Lkv.linear.recurrent.data(), nv, dk, dv);
    hal::gated_delta_recurrent(sc.q.data(), sc.k.data(), sc.v.data(), sc.g.data(), sc.beta.data(),
                               Lkv.linear.recurrent.data(), sc.core.data(), 1, nv, dk, dv, true);
  }

  for (int h = 0; h < nv; ++h) {
    float* ch = sc.core.data() + h * dv;
    float* zh = sc.z.data() + h * dv;
    hal::rmsnorm_gated(ch, zh, lp.nrm, ch, dv, cfg_.rms_eps, lp.nrm_dt);
  }

  const qlwc::Int4View* wout_i4 = lp.wout.is_int4 ? &lp.wout.i4 : nullptr;
  const uint16_t* wout_pass = lp.wout.is_int4 ? nullptr : lp.wout.pass;
  if (!wout_i4 && !wout_pass) return false;
  return hal::cuda::try_ffn_on_act(sc.core.data(), value_dim, wout_i4, wout_pass,
                                   lp.wout.dt == hal::WDtype::kF16, lp.ln2, lp.wgate, lp.wup,
                                   lp.wdown, I, cfg_.rms_eps, ln_f16);
}

bool Qwen35Int4Model::layer_forward_full_act(int layer, SessionCache& cache, int pos_start) {
  if (!resident_gpu_ || !hal::cuda::decode_act_valid()) return false;
  const auto& lp = layers_[layer];
  if (lp.is_moe) return false;
  if (!lp.is_full || !lp.ln1 || !lp.ln2) return false;
  hal::cuda::note_full_attn_try();

  const int I = cfg_.intermediate;
  const int nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
  const int rotary_dim = static_cast<int>(hd * cfg_.partial_rotary) / 2 * 2;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const bool ln_f16 = lp.ln_dt == hal::WDtype::kF16;
  auto& sc = scratch();
  auto& Lkv = cache.layer(layer);

  Int4Scratch::fit(sc.qg, static_cast<size_t>(nh) * hd * 2);
  Int4Scratch::fit(sc.kk, static_cast<size_t>(nkv) * hd);
  Int4Scratch::fit(sc.vv, static_cast<size_t>(nkv) * hd);
  Int4Scratch::fit(sc.qq, static_cast<size_t>(nh) * hd);
  Int4Scratch::fit(sc.gate, static_cast<size_t>(nh) * hd);
  Int4Scratch::fit(sc.attn_heads, static_cast<size_t>(nh) * hd);

  // Residual stays on device: only Q/K/V projections D2H.
  if (!(lp.wq.is_int4 && lp.wk.is_int4 && lp.wv.is_int4)) return false;
  const qlwc::Int4View* ws3[3] = {&lp.wq.i4, &lp.wk.i4, &lp.wv.i4};
  float* ys3[3] = {sc.qg.data(), sc.kk.data(), sc.vv.data()};
  if (!hal::cuda::try_rmsnorm_gemm_multi_from_act(lp.ln1, ws3, ys3, 3, cfg_.rms_eps, ln_f16))
    return false;

  const int idx = (static_cast<int>(cur_pos_t_.size()) == 1) ? 0 : pos_start;
  const int pt = (idx < static_cast<int>(cur_pos_t_.size())) ? cur_pos_t_[idx] : pos_start;
  const int ph = (idx < static_cast<int>(cur_pos_h_.size())) ? cur_pos_h_[idx] : pt;
  const int pw = (idx < static_cast<int>(cur_pos_w_.size())) ? cur_pos_w_[idx] : pt;

  for (int h = 0; h < nh; ++h) {
    float* qh = sc.qq.data() + h * hd;
    float* gh = sc.gate.data() + h * hd;
    const float* src = sc.qg.data() + h * hd * 2;
    std::memcpy(qh, src, sizeof(float) * hd);
    std::memcpy(gh, src + hd, sizeof(float) * hd);
    hal::rmsnorm(qh, lp.qn, qh, hd, cfg_.rms_eps, lp.qk_norm_dt, true);
    hal::apply_mrope_freqs(qh, hd, rotary_dim, pt, ph, pw, cfg_.rope_theta, mrope_section_,
                           mrope_interleaved_);
  }
  for (int h = 0; h < nkv; ++h) {
    float* kh = sc.kk.data() + h * hd;
    float* vh = sc.vv.data() + h * hd;
    hal::rmsnorm(kh, lp.kn, kh, hd, cfg_.rms_eps, lp.qk_norm_dt, true);
    hal::apply_mrope_freqs(kh, hd, rotary_dim, pt, ph, pw, cfg_.rope_theta, mrope_section_,
                           mrope_interleaved_);
    float* kdst = Lkv.k.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq) * hd;
    float* vdst = Lkv.v.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq) * hd;
    std::memcpy(kdst, kh, sizeof(float) * hd);
    std::memcpy(vdst, vh, sizeof(float) * hd);
  }

  const int seq_len = Lkv.seq + 1;
  const int kv_pos = Lkv.seq;
  hal::attn_decode_one(sc.qq.data(), Lkv.k.data(), Lkv.v.data(), sc.attn_heads.data(), nh, nkv, hd,
                       seq_len, cache.max_seq(), scale);

  for (int i = 0; i < nh * hd; ++i) sc.attn_heads[i] *= sigmoid(sc.gate[i]);

  // o_proj + residual add + MLP on device act (no residual PCIe).
  if (!lp.wo.is_int4) return false;
  if (!hal::cuda::try_ffn_on_act(sc.attn_heads.data(), nh * hd, &lp.wo.i4, nullptr, false, lp.ln2,
                                 lp.wgate, lp.wup, lp.wdown, I, cfg_.rms_eps, ln_f16)) {
    // Roll back KV slot so host fallback can rewrite the same position.
    (void)kv_pos;
    return false;
  }
  Lkv.seq += 1;
  hal::cuda::note_full_attn_ok();
  return true;
}

void Qwen35Int4Model::forward_to_hidden(const std::vector<int32_t>& tokens, SessionCache& cache,
                                        bool is_prefill, float* h_out, double* ms_lin,
                                        double* ms_full) {
  if (!store_ || tokens.empty()) throw std::runtime_error("model not ready / empty tokens");
  const int n = static_cast<int>(tokens.size());
  const int H = cfg_.hidden;
  using Clock = std::chrono::steady_clock;

  auto& sc = scratch();
  Int4Scratch::fit(sc.xbuf, static_cast<size_t>(n) * H);
  float* x = sc.xbuf.data();

  int pos_start = 0;
  if (!is_prefill) {
    for (int i = 0; i < cfg_.layers; ++i)
      if (cfg_.layer_types[i] == "full_attention") {
        pos_start = cache.layer(i).seq;
        break;
      }
  }
  if (is_prefill) vision_cursor_ = 0;
  for (int t = 0; t < n; ++t) embed(tokens[t], x + t * H);
  prepare_mrope_positions(tokens, is_prefill);

  // MoE skips Path-A act stream: fused MLP unavailable; sync sandwich + GPU lm_head
  // caused sticky garbage (e.g. wall of "Ò") and wasted PCIe every layer.
  const bool stream_act = !cfg_.is_moe && resident_gpu_ && !is_prefill && n == 1 && hal::cuda::resident_gpu_enabled() && hal::cuda::decode_act_begin(x, H);

  double lin = 0, full = 0;
  const bool time_layers = (ms_lin && ms_full);
  const bool prog_prefill = is_prefill && n >= 64;
  const auto t_pf0 = prog_prefill ? Clock::now() : Clock::time_point{};
  if (prog_prefill) {
    LOG_INFO("prefill begin: tok=%d layers=%d (progress every 8 layers)", n, cfg_.layers);
  }
  if (is_prefill) on_prefill_begin();
  for (int L = 0; L < cfg_.layers; ++L) {
    if (streamer_) {
      if (L + 1 < cfg_.layers) streamer_->prefetch_layer(L + 1);
      streamer_->pin_layer(L);
      fill_layer_pack(L);
    }
    if (is_prefill) on_prefill_layer(L, n);
    const bool is_full = (cfg_.layer_types[L] == "full_attention");
    auto run = [&]() {
      if (stream_act && !is_full) {
        if (layer_forward_linear_act(L, cache)) return;
        if (!hal::cuda::decode_act_sync_to_host(x, H)) {
         hal::cuda::decode_act_invalidate();
          layer_forward(L, x, cache, pos_start, n, is_prefill);
          return;
        }
        layer_forward(L, x, cache, pos_start, n, is_prefill);
       hal::cuda::decode_act_load_from_host(x, H);
        return;
      }
      if (stream_act && is_full) {
        if (layer_forward_full_act(L, cache, pos_start)) return;
        if (!hal::cuda::decode_act_sync_to_host(x, H)) {
          hal::cuda::decode_act_invalidate();
          layer_forward(L, x, cache, pos_start, n, is_prefill);
          return;
        }
        layer_forward(L, x, cache, pos_start, n, is_prefill);
        hal::cuda::decode_act_load_from_host(x, H);
        return;
      }
      layer_forward(L, x, cache, pos_start, n, is_prefill);
    };
    if (time_layers) {
      const auto a = Clock::now();
      run();
      const auto b = Clock::now();
      const double d = std::chrono::duration<double, std::milli>(b - a).count();
      if (is_full)
        full += d;
      else
        lin += d;
    } else {
      run();
    }
    if (is_prefill) on_prefill_layer_done(L);
    if (streamer_ && L + 1 >= stream_window_) streamer_->release_layer(L + 1 - stream_window_);
    if (prog_prefill && ((L + 1) % 8 == 0 || L + 1 == cfg_.layers)) {
      const double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t_pf0).count();
      LOG_INFO("prefill layer %d/%d (%.1fs, ~%.1f layer/s)", L + 1, cfg_.layers, ms / 1000.0,
               (L + 1) / (ms > 0 ? ms / 1000.0 : 1.0));
    }
  }
  if (ms_lin) *ms_lin = lin;
  if (ms_full) *ms_full = full;

  if (stream_act && hal::cuda::decode_act_valid()) {
    // Sync residual for last_hidden_; lm_head may re-norm from device act.
    if (!hal::cuda::decode_act_sync_to_host(h_out, H)) {
     hal::rmsnorm(x, final_norm_, h_out, H, cfg_.rms_eps, final_norm_dt_, true);
     hal::cuda::decode_act_invalidate();
    } else {
      std::vector<float> tmp(static_cast<size_t>(H));
      std::memcpy(tmp.data(), h_out, sizeof(float) * static_cast<size_t>(H));
     hal::rmsnorm(tmp.data(), final_norm_, h_out, H, cfg_.rms_eps, final_norm_dt_, true);
    }
  } else {
   hal::cuda::decode_act_invalidate();
   hal::rmsnorm(x + (n - 1) * H, final_norm_, h_out, H, cfg_.rms_eps, final_norm_dt_, true);
  }
  for (int i = 0; i < H; ++i)
    if (!std::isfinite(h_out[i])) h_out[i] = 0.f;
  last_hidden_.assign(h_out, h_out + H);
}

void Qwen35Int4Model::forward(const std::vector<int32_t>& tokens, SessionCache& cache,
                              std::vector<float>& logits, bool is_prefill) {
  const int H = cfg_.hidden;
  const int V = cfg_.vocab;
  static const bool kProf = [] {
    const char* e = std::getenv("LLMOC_PROFILE");
    return e && e[0] == '1';
  }();
  // One-shot decode profile when resident_gpu (no env needed).
  static int kAutoProfLeft = resident_gpu_ ? 2 : 0;
  const bool do_prof = kProf || (!is_prefill && kAutoProfLeft > 0);
  using Clock = std::chrono::steady_clock;
  const auto t0 = do_prof ? Clock::now() : Clock::time_point{};

  std::vector<float> h(H);
  double ms_lin = 0, ms_full = 0;
  forward_to_hidden(tokens, cache, is_prefill, h.data(), do_prof ? &ms_lin : nullptr,
                    do_prof ? &ms_full : nullptr);

  Clock::time_point t_head0;
  if (do_prof) t_head0 = Clock::now();

  prefix_hiddens_.clear();
  prefix_logits_.clear();
  logits.resize(static_cast<size_t>(V));
  bool lm_from_act = false;
  // MoE: force host lm_head (GPU vocab GEMV previously produced sticky garbage tokens).
  if (!cfg_.is_moe && hal::cuda::enabled() && hal::cuda::decode_act_valid()) {
    const bool fn_f16 = final_norm_dt_ == hal::WDtype::kF16;
    const bool lm_f16 = lm_dt_ == hal::WDtype::kF16;
    if (lm_is_int4_) {
      lm_from_act = hal::cuda::try_lm_head_int4_from_act(final_norm_, lm_int4_, logits.data(),
                                                         cfg_.rms_eps, fn_f16);
    } else if (lm_pass_) {
      lm_from_act = hal::cuda::try_lm_head_w16_from_act(final_norm_, lm_pass_, V, logits.data(),
                                                        cfg_.rms_eps, fn_f16, lm_f16);
    }
  }
  if (!lm_from_act) {
    if (lm_is_int4_) {
      bool gpu_ok = false;
      if (!cfg_.is_moe && hal::cuda::enabled()) {
        gpu_ok = hal::cuda::try_gemm_int4(h.data(), lm_int4_, logits.data());
      }
      if (!gpu_ok) {
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
      }
    } else {
      if (!cfg_.is_moe && hal::cuda::enabled() && lm_pass_ && hal::cuda::try_gemm_w16(h.data(), lm_pass_, logits.data(), V, H,
                                  lm_dt_ == hal::WDtype::kF16)) {
        /* GPU resident W16 cublas SGEMM */
      } else {
      hal::gemm_bias_free(h.data(), lm_pass_, logits.data(), V, H, lm_dt_,
                            /*allow_gpu=*/!cfg_.is_moe);
      }
    }
  }
  hal::cuda::decode_act_invalidate();
  last_logits_ = logits;

  if (do_prof) {
    const auto t1 = Clock::now();
    const double ms_head = std::chrono::duration<double, std::milli>(t1 - t_head0).count();
    const double ms_tot = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INFO("profile decode: lin=%.1fms full=%.1fms lm_head=%.1fms total=%.1fms (%.2f tok/s)",
             ms_lin, ms_full, ms_head, ms_tot, ms_tot > 0 ? 1000.0 / ms_tot : 0.0);
    if (kAutoProfLeft > 0) --kAutoProfLeft;
  }
}

bool Qwen35Int4Model::forward_decode_greedy(const std::vector<int32_t>& tokens,
                                             SessionCache& cache, int32_t& out_token) {
  if (tokens.size() != 1 || !lm_is_int4_) return false;
  const int H = cfg_.hidden;
  auto& sc = scratch();
  Int4Scratch::fit(sc.last, static_cast<size_t>(H));
  forward_to_hidden(tokens, cache, false, sc.last.data());
  // 先尝试 GPU resident INT4 (JIT gemv_int4 M=248320 ~3ms); 失败回退 CPU AVX2 24ms
  // MoE: skip GPU lm_head (sticky garbage risk).
  if (!cfg_.is_moe && hal::cuda::enabled()) {
    Int4Scratch::fit(sc.logits, lm_int4_.M);
    if (hal::cuda::try_gemm_int4(sc.last.data(), lm_int4_, sc.logits.data())) {
      const float* lp = sc.logits.data();
      float best_val = lp[0];
      int best = 0;
      for (int i = 1; i < lm_int4_.M; ++i) {
        if (lp[i] > best_val) { best_val = lp[i]; best = i; }
      }
      out_token = best;
      last_logits_.assign(1, best_val);
      prefix_hiddens_.clear();
      prefix_logits_.clear();
      return true;
    } else if (lm_pass_) {
      if (hal::cuda::try_gemm_w16(sc.last.data(), lm_pass_, sc.logits.data(), cfg_.vocab, H,
                                  lm_dt_ == hal::WDtype::kF16)) {
        const float* lp = sc.logits.data();
        float best_val = lp[0];
        int best = 0;
        for (int i = 1; i < cfg_.vocab; ++i) {
          if (lp[i] > best_val) { best_val = lp[i]; best = i; }
        }
        out_token = best;        last_logits_.assign(1, best_val);
        prefix_hiddens_.clear();
        prefix_logits_.clear();
        return true;      }
    }
  }
#if defined(LLMOC_ENABLE_AVX2)
  if (lm_int4_.qweight)
    _mm_prefetch(reinterpret_cast<const char*>(lm_int4_.qweight), _MM_HINT_T0);
#endif
  const auto ar = hal::gemm_int4_argmax(sc.last.data(), lm_int4_);
  out_token = ar.index;
  last_logits_.assign(1, ar.value);
  prefix_hiddens_.clear();
  prefix_logits_.clear();
  return true;
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
  if (is_prefill) on_prefill_begin();
  for (int L = 0; L < cfg_.layers; ++L) {
    if (streamer_) {
      if (L + 1 < cfg_.layers) streamer_->prefetch_layer(L + 1);
      streamer_->pin_layer(L);
      fill_layer_pack(L);
    }
    if (is_prefill) on_prefill_layer(L, n);
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
    if (is_prefill) on_prefill_layer_done(L);
    if (streamer_ && L + 1 >= stream_window_) streamer_->release_layer(L + 1 - stream_window_);
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
      hal::rmsnorm(x.data() + t * H, final_norm_, ht, H, cfg_.rms_eps, final_norm_dt_, true);
      for (int i = 0; i < H; ++i)
        if (!std::isfinite(ht[i])) ht[i] = 0.f;
      if (keep_prefix)
        std::memcpy(prefix_hiddens_.data() + static_cast<size_t>(t) * H, ht, sizeof(float) * H);
    }
    last_hidden_.assign(sc.last.begin() + static_cast<size_t>(n - 1) * H, sc.last.end());
    gemm_view_batch(sc.last.data(), n, lm_int4_, logits_all.data());
  } else {
    for (int t = 0; t < n; ++t) {
      hal::rmsnorm(x.data() + t * H, final_norm_, h.data(), H, cfg_.rms_eps, final_norm_dt_, true);
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
        hal::gemm_bias_free(h.data(), lm_pass_, dest, V, H, lm_dt_,
                            /*allow_gpu=*/!cfg_.is_moe);
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

void Qwen35Int4Model::prepare_speculative_snapshot(SessionCache& cache) {
  if (!resident_gpu_) return;
  const int nv = cfg_.linear_num_v, dk = cfg_.linear_dk, dv = cfg_.linear_dv;
  for (int i = 0; i < cache.n_layers(); ++i) {
    if (i >= static_cast<int>(cfg_.layer_types.size()) || cfg_.layer_types[i] != "linear_attention")
      continue;
    auto& lin = cache.layer(i).linear;
    if (!lin.recurrent.empty())
      hal::cuda::flush_gdn_state_to_host(lin.recurrent.data(), nv, dk, dv);
  }
}

void Qwen35Int4Model::apply_speculative_restore(SessionCache& cache) {
  if (!resident_gpu_) return;
  for (int i = 0; i < cache.n_layers(); ++i) {
    if (i >= static_cast<int>(cfg_.layer_types.size()) || cfg_.layer_types[i] != "linear_attention")
      continue;
    auto& lin = cache.layer(i).linear;
    if (!lin.recurrent.empty()) hal::cuda::invalidate_gdn_state(lin.recurrent.data());
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
                                    std::vector<int32_t>& out, int32_t pin_first) {
  out.clear();
  if (last_hidden_.empty()) return false;
  return mtp_draft_propose(mtp_access(), last_hidden_, history, draft_k, out, pin_first);
}

}  // namespace llmoc::model
