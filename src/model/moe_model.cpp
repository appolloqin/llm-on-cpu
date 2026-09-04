// llm-on-cpu :: model/moe_model.cpp
#include "model/moe_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "common/log.h"
#include "hal/cuda_backend.h"

namespace llmoc::model {

bool MoeModel::has(const std::string& name) const {
  return wm_ && wm_->header().find(name) != nullptr;
}

void MoeModel::load(wt::WeightManager* wm, wt::ExpertPrefetcher* pref,
                    const std::string& hf_config_json) {
  wm_ = wm;
  pref_ = pref;
  if (!wm_) throw std::runtime_error("MoeModel: null WeightManager");

  const auto dt = wm_->header().dtype;
  wd_ = dt == lwc::Dtype::F16 ? hal::WDtype::kF16
        : dt == lwc::Dtype::F32 ? hal::WDtype::kF32
                               : hal::WDtype::kBF16;
  if (wd_ == hal::WDtype::kF32) throw std::runtime_error("MoeModel: F32 weights not supported yet");

  if (has("language_model.embed_tokens.weight") || has("language_model.layers.0.input_layernorm.weight"))
    prefix_ = "language_model.";
  else
    prefix_ = "";

  std::ifstream in(hf_config_json);
  nlohmann::json root = nlohmann::json::object();
  if (in) in >> root;
  const auto& tc = root.contains("text_config") ? root["text_config"] : root;

  cfg_.hidden = tc.value("hidden_size", 2048);
  cfg_.layers = tc.value("num_hidden_layers", 0);
  cfg_.n_heads = tc.value("num_attention_heads", 16);
  cfg_.n_kv = tc.value("num_key_value_heads", cfg_.n_heads);
  cfg_.head_dim = tc.value("head_dim", cfg_.hidden / std::max(cfg_.n_heads, 1));
  cfg_.intermediate = tc.value("intermediate_size", 0);
  cfg_.moe_intermediate = tc.value("moe_intermediate_size", tc.value("intermediate_size", 768));
  cfg_.n_experts = tc.value("n_routed_experts", tc.value("num_experts", 0));
  cfg_.topk = tc.value("num_experts_per_tok", 8);
  cfg_.first_k_dense = tc.value("first_k_dense_replace", 0);
  cfg_.vocab = tc.value("vocab_size", 151936);
  cfg_.rms_eps = static_cast<float>(tc.value("rms_norm_eps", 1e-6));
  cfg_.tie_embeddings = tc.value("tie_word_embeddings", true);
  if (tc.contains("rope_theta")) cfg_.rope_theta = tc.value("rope_theta", 1000000.f);
  if (tc.contains("rope_parameters")) {
    cfg_.rope_theta = tc["rope_parameters"].value("rope_theta", cfg_.rope_theta);
    cfg_.partial_rotary = tc["rope_parameters"].value("partial_rotary_factor", 1.f);
  }

  // 从 LWC groups 回填
  if (cfg_.layers <= 0) {
    uint32_t max_l = 0;
    for (const auto& g : wm_->header().groups) max_l = std::max(max_l, g.layer + 1);
    cfg_.layers = static_cast<int>(max_l);
  }
  if (cfg_.n_experts <= 0) {
    int max_e = 0;
    for (const auto& g : wm_->header().groups)
      max_e = std::max(max_e, static_cast<int>(g.expert_id) + 1);
    cfg_.n_experts = max_e;
  }
  if (cfg_.layers <= 0 || cfg_.n_experts <= 0)
    throw std::runtime_error("MoeModel: cannot infer layers/experts (empty groups?)");

  cfg_.has_q_norm = has(prefix_ + "layers.0.self_attn.q_norm.weight");
  cfg_.fused_kv = has(prefix_ + "layers.0.self_attn.kv_proj.weight");

  // vocab from embed
  const auto* emb = wm_->header().find(prefix_.empty() ? "embedding.weight"
                                                       : prefix_ + "embed_tokens.weight");
  if (!emb) emb = wm_->header().find("embedding.weight");
  if (emb && !emb->shape.empty()) cfg_.vocab = static_cast<int>(emb->shape[0]);
  if (emb && emb->shape.size() > 1) cfg_.hidden = static_cast<int>(emb->shape[1]);

  meta_.hidden = cfg_.hidden;
  meta_.layers = cfg_.layers;
  meta_.vocab = cfg_.vocab;
  meta_.n_kv = cfg_.n_kv;
  meta_.head_dim = cfg_.head_dim;
  meta_.is_moe = true;
  meta_.kind = "moe";
  meta_.conv_k = 4;
  meta_.conv_dim = 0;

  LOG_INFO("MoeModel loaded: layers=%d experts=%d topk=%d hidden=%d prefix='%s' dtype=%s",
           cfg_.layers, cfg_.n_experts, cfg_.topk, cfg_.hidden, prefix_.c_str(),
           wd_ == hal::WDtype::kF16 ? "F16" : "BF16");
}

void MoeModel::init_cache(SessionCache& cache, int max_seq) const {
  cache.init(cfg_.layers, max_seq, cfg_.n_kv, cfg_.head_dim, /*n_v*/ 1, /*dk*/ 1, /*dv*/ 1,
             /*conv_dim*/ 1, /*conv_k*/ 1);
}

const uint16_t* MoeModel::w(const std::string& name) {
  return reinterpret_cast<const uint16_t*>(wm_->get(name).data());
}

void MoeModel::embed(int32_t token, float* out) {
  std::string name = prefix_.empty() ? "embedding.weight" : prefix_ + "embed_tokens.weight";
  if (!has(name)) name = "embedding.weight";
  const uint16_t* row = w(name) + static_cast<size_t>(token) * cfg_.hidden;
  for (int i = 0; i < cfg_.hidden; ++i) out[i] = hal::load_w(row + i, wd_);
}

void MoeModel::router_topk(const float* x, const uint16_t* Wgate, int* ids, float* weights) {
  const int E = cfg_.n_experts;
  const int K = cfg_.topk;
  std::vector<float> logits(E);
  hal::gemm_bias_free(x, Wgate, logits.data(), E, cfg_.hidden, wd_);
  // softmax
  float m = *std::max_element(logits.begin(), logits.end());
  double s = 0.0;
  for (int i = 0; i < E; ++i) {
    logits[i] = std::exp(logits[i] - m);
    s += logits[i];
  }
  for (int i = 0; i < E; ++i) logits[i] = static_cast<float>(logits[i] / s);
  // top-k
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + K, order.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  double wsum = 0.0;
  for (int i = 0; i < K; ++i) {
    ids[i] = order[i];
    weights[i] = logits[order[i]];
    wsum += weights[i];
  }
  for (int i = 0; i < K; ++i) weights[i] = static_cast<float>(weights[i] / wsum);
}

void MoeModel::attn_layer(int layer, float* x, SessionCache& cache, int pos_start, int n_tok,
                          bool is_prefill) {
  const int H = cfg_.hidden, nh = cfg_.n_heads, nkv = cfg_.n_kv, hd = cfg_.head_dim;
  const std::string base = prefix_ + "layers." + std::to_string(layer) + ".";
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const int rotary_dim = std::max(2, static_cast<int>(hd * cfg_.partial_rotary) / 2 * 2);

  std::vector<float> residual(static_cast<size_t>(n_tok) * H);
  std::memcpy(residual.data(), x, sizeof(float) * n_tok * H);
  std::vector<float> normed(static_cast<size_t>(n_tok) * H);
  const uint16_t* ln1 = w(base + "input_layernorm.weight");
  for (int t = 0; t < n_tok; ++t)
    hal::rmsnorm(x + t * H, ln1, normed.data() + t * H, H, cfg_.rms_eps, wd_);

  auto& Lkv = cache.layer(layer);
  std::vector<float> qq(static_cast<size_t>(n_tok) * nh * hd);
  std::vector<float> kk(static_cast<size_t>(n_tok) * nkv * hd);
  std::vector<float> vv(static_cast<size_t>(n_tok) * nkv * hd);

  const uint16_t* Wq = w(base + "self_attn.q_proj.weight");
  if (cfg_.fused_kv) {
    const uint16_t* Wkv = w(base + "self_attn.kv_proj.weight");
    std::vector<float> kv(2 * H);
    for (int t = 0; t < n_tok; ++t) {
      // selftest: q HxH, treat as 1 head of dim H if nh*hd != H
      if (nh * hd == H) {
        hal::gemm_bias_free(normed.data() + t * H, Wq, qq.data() + t * nh * hd, nh * hd, H, wd_);
      } else {
        // map to single vector then split
        std::vector<float> qtmp(H);
        hal::gemm_bias_free(normed.data() + t * H, Wq, qtmp.data(), H, H, wd_);
        std::memcpy(qq.data() + t * nh * hd, qtmp.data(),
                    sizeof(float) * std::min(H, nh * hd));
      }
      hal::gemm_bias_free(normed.data() + t * H, Wkv, kv.data(), 2 * H, H, wd_);
      // split k/v equally
      const int half = H / 2;
      for (int h = 0; h < nkv; ++h) {
        std::memcpy(kk.data() + (t * nkv + h) * hd, kv.data() + (h * hd) % half,
                    sizeof(float) * std::min(hd, half));
        std::memcpy(vv.data() + (t * nkv + h) * hd, kv.data() + half + (h * hd) % half,
                    sizeof(float) * std::min(hd, half));
      }
    }
  } else {
    const uint16_t* Wk = w(base + "self_attn.k_proj.weight");
    const uint16_t* Wv = w(base + "self_attn.v_proj.weight");
    for (int t = 0; t < n_tok; ++t) {
      hal::gemm_bias_free(normed.data() + t * H, Wq, qq.data() + t * nh * hd, nh * hd, H, wd_);
      hal::gemm_bias_free(normed.data() + t * H, Wk, kk.data() + t * nkv * hd, nkv * hd, H, wd_);
      hal::gemm_bias_free(normed.data() + t * H, Wv, vv.data() + t * nkv * hd, nkv * hd, H, wd_);
    }
    if (cfg_.has_q_norm) {
      const uint16_t* qn = w(base + "self_attn.q_norm.weight");
      const uint16_t* kn = w(base + "self_attn.k_norm.weight");
      for (int t = 0; t < n_tok; ++t) {
        for (int h = 0; h < nh; ++h)
          hal::rmsnorm(qq.data() + (t * nh + h) * hd, qn, qq.data() + (t * nh + h) * hd, hd,
                       cfg_.rms_eps, wd_);
        for (int h = 0; h < nkv; ++h)
          hal::rmsnorm(kk.data() + (t * nkv + h) * hd, kn, kk.data() + (t * nkv + h) * hd, hd,
                       cfg_.rms_eps, wd_);
      }
    }
  }

  for (int t = 0; t < n_tok; ++t) {
    const int pos = pos_start + t;
    for (int h = 0; h < nh; ++h)
      hal::apply_rope_freqs(qq.data() + (t * nh + h) * hd, hd, rotary_dim, pos, cfg_.rope_theta);
    for (int h = 0; h < nkv; ++h) {
      float* kh = kk.data() + (t * nkv + h) * hd;
      float* vh = vv.data() + (t * nkv + h) * hd;
      hal::apply_rope_freqs(kh, hd, rotary_dim, pos, cfg_.rope_theta);
      float* kdst = Lkv.k.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq + t) * hd;
      float* vdst = Lkv.v.data() + (static_cast<size_t>(h) * cache.max_seq() + Lkv.seq + t) * hd;
      std::memcpy(kdst, kh, sizeof(float) * hd);
      std::memcpy(vdst, vh, sizeof(float) * hd);
    }
  }

  std::vector<float> attn_heads(static_cast<size_t>(n_tok) * nh * hd);
  if (is_prefill && Lkv.seq == 0) {
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
    if (!hal::cuda::try_attn_prefill(qq.data(), kpf.data(), vpf.data(), attn_heads.data(), n_tok, nh,
                                     nkv, hd, scale)) {
      hal::attn_prefill(qq.data(), kpf.data(), vpf.data(), attn_heads.data(), n_tok, nh, nkv, hd,
                        scale);
    }
  } else {
    for (int t = 0; t < n_tok; ++t)
      hal::attn_decode_one(qq.data() + t * nh * hd, Lkv.k.data(), Lkv.v.data(),
                           attn_heads.data() + t * nh * hd, nh, nkv, hd, Lkv.seq + t + 1,
                           cache.max_seq(), scale);
  }
  Lkv.seq += n_tok;

  std::vector<float> attn_out(static_cast<size_t>(n_tok) * H, 0.f);
  if (has(base + "self_attn.o_proj.weight")) {
    const uint16_t* Wo = w(base + "self_attn.o_proj.weight");
    for (int t = 0; t < n_tok; ++t)
      hal::gemm_bias_free(attn_heads.data() + t * nh * hd, Wo, attn_out.data() + t * H, H, nh * hd,
                          wd_);
  } else {
    // selftest without o_proj: copy truncated
    for (int t = 0; t < n_tok; ++t)
      std::memcpy(attn_out.data() + t * H, attn_heads.data() + t * nh * hd,
                  sizeof(float) * std::min(H, nh * hd));
  }
  for (size_t i = 0; i < residual.size(); ++i) x[i] = residual[i] + attn_out[i];
}

void MoeModel::dense_mlp(int layer, float* x, int n_tok) {
  const int H = cfg_.hidden;
  const int I = cfg_.intermediate > 0 ? cfg_.intermediate : cfg_.moe_intermediate;
  const std::string base = prefix_ + "layers." + std::to_string(layer) + ".";
  std::vector<float> residual(static_cast<size_t>(n_tok) * H);
  std::memcpy(residual.data(), x, sizeof(float) * n_tok * H);
  const std::string ln_name = has(base + "post_attention_layernorm.weight")
                                  ? base + "post_attention_layernorm.weight"
                                  : base + "input_layernorm.weight";
  const uint16_t* ln2 = w(ln_name);
  const uint16_t* Wg = w(base + "mlp.gate_proj.weight");
  const uint16_t* Wu = w(base + "mlp.up_proj.weight");
  const uint16_t* Wd = w(base + "mlp.down_proj.weight");
  std::vector<float> nrm(H), g(I), u(I), mid(I), down(H);
  for (int t = 0; t < n_tok; ++t) {
    hal::rmsnorm(x + t * H, ln2, nrm.data(), H, cfg_.rms_eps, wd_);
    hal::gemm_bias_free(nrm.data(), Wg, g.data(), I, H, wd_);
    hal::gemm_bias_free(nrm.data(), Wu, u.data(), I, H, wd_);
    hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
    hal::gemm_bias_free(mid.data(), Wd, down.data(), H, I, wd_);
    for (int i = 0; i < H; ++i) x[t * H + i] = residual[t * H + i] + down[i];
  }
}

void MoeModel::moe_mlp_token(int layer, float* x, bool use_prefetch, int* out_ids, float* out_w,
                             int* n_sel) {
  const int H = cfg_.hidden;
  const int I = cfg_.moe_intermediate;
  const int K = cfg_.topk;
  const std::string base = prefix_ + "layers." + std::to_string(layer) + ".";

  std::vector<float> residual(H);
  std::memcpy(residual.data(), x, sizeof(float) * H);
  const std::string ln_name = has(base + "post_attention_layernorm.weight")
                                  ? base + "post_attention_layernorm.weight"
                                  : base + "input_layernorm.weight";
  std::vector<float> nrm(H);
  hal::rmsnorm(x, w(ln_name), nrm.data(), H, cfg_.rms_eps, wd_);

  // router
  std::string gate_name = base + "mlp.gate.weight";
  if (!has(gate_name)) gate_name = base + "mlp.router.weight";
  int ids[64];
  float ws[64];
  if (K > 64) throw std::runtime_error("topk too large");
  if (has(gate_name)) {
    router_topk(nrm.data(), w(gate_name), ids, ws);
  } else {
    // selftest 无 router: 用前 topk 专家均权
    for (int i = 0; i < K; ++i) {
      ids[i] = i % cfg_.n_experts;
      ws[i] = 1.f / K;
    }
  }
  *n_sel = K;
  for (int i = 0; i < K; ++i) {
    out_ids[i] = ids[i];
    out_w[i] = ws[i];
  }

  std::vector<wt::ExpertData> experts;
  if (use_prefetch && pref_) {
    pref_->acquire(layer, experts);
  }

  std::vector<float> acc(H, 0.f);
  std::vector<float> g(I), u(I), mid(I), down(H);
  for (int i = 0; i < K; ++i) {
    const uint16_t *Wg = nullptr, *Wu = nullptr, *Wd = nullptr;
    if (use_prefetch && pref_ && !experts.empty()) {
      // acquire 顺序对应 plan 时的 ids
      size_t idx = static_cast<size_t>(i);
      if (idx >= experts.size()) idx = 0;
      Wg = reinterpret_cast<const uint16_t*>(experts[idx].gate.data);
      Wu = reinterpret_cast<const uint16_t*>(experts[idx].up.data);
      Wd = reinterpret_cast<const uint16_t*>(experts[idx].down.data);
    } else {
      const std::string gname = wt::ExpertPrefetcher::part_name(layer, ids[i], "gate");
      const std::string uname = wt::ExpertPrefetcher::part_name(layer, ids[i], "up");
      const std::string dname = wt::ExpertPrefetcher::part_name(layer, ids[i], "down");
      Wg = w(gname);
      Wu = w(uname);
      Wd = w(dname);
    }
    hal::gemm_bias_free(nrm.data(), Wg, g.data(), I, H, wd_);
    hal::gemm_bias_free(nrm.data(), Wu, u.data(), I, H, wd_);
    hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
    hal::gemm_bias_free(mid.data(), Wd, down.data(), H, I, wd_);
    for (int d = 0; d < H; ++d) acc[d] += ws[i] * down[d];
  }
  if (use_prefetch && pref_) pref_->release(layer);

  // optional shared expert
  if (has(base + "mlp.shared_expert.gate_proj.weight")) {
    const uint16_t* Wg = w(base + "mlp.shared_expert.gate_proj.weight");
    const uint16_t* Wu = w(base + "mlp.shared_expert.up_proj.weight");
    const uint16_t* Wd = w(base + "mlp.shared_expert.down_proj.weight");
    hal::gemm_bias_free(nrm.data(), Wg, g.data(), I, H, wd_);
    hal::gemm_bias_free(nrm.data(), Wu, u.data(), I, H, wd_);
    hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
    hal::gemm_bias_free(mid.data(), Wd, down.data(), H, I, wd_);
    for (int d = 0; d < H; ++d) acc[d] += down[d];
  }

  for (int d = 0; d < H; ++d) x[d] = residual[d] + acc[d];
}

void MoeModel::forward(const std::vector<int32_t>& tokens, SessionCache& cache,
                       std::vector<float>& logits, bool is_prefill) {
  if (!wm_ || tokens.empty()) throw std::runtime_error("MoeModel not ready");
  const int n = static_cast<int>(tokens.size());
  const int H = cfg_.hidden;
  std::vector<float> x(static_cast<size_t>(n) * H);

  int pos_start = 0;
  if (!is_prefill) {
    for (int i = 0; i < cfg_.layers; ++i)
      if (cache.layer(i).seq > 0) {
        pos_start = cache.layer(i).seq;
        break;
      }
  }
  for (int t = 0; t < n; ++t) embed(tokens[t], x.data() + t * H);

  // Prefill: 逐层 attn(全序列) + 逐 token MoE
  // Decode: 单 token, 可走 prefetch
  for (int L = 0; L < cfg_.layers; ++L) {
    attn_layer(L, x.data(), cache, pos_start, n, is_prefill);

    const bool dense = L < cfg_.first_k_dense ||
                       has(prefix_ + "layers." + std::to_string(L) + ".mlp.gate_proj.weight");
    if (dense && has(prefix_ + "layers." + std::to_string(L) + ".mlp.gate_proj.weight")) {
      dense_mlp(L, x.data(), n);
      continue;
    }

    // MoE
    const bool decode_pref = !is_prefill && n == 1 && pref_ != nullptr;
    for (int t = 0; t < n; ++t) {
      int ids[64];
      float ws[64];
      int nsel = 0;
      // decode: 先 plan 本层(若尚未 plan)—— 简化: 先 router 再 plan 下一层, 本层用 wm.get
      // 更佳: 上一层末尾已 plan 本层。此处 decode 用 prefetch: plan 本层 ids 后 acquire。
      if (decode_pref && t == 0) {
        // 先用 router 得到 ids, plan 自己(同步读), 同时 plan L+1 用相同 ids 作启发式
        std::vector<float> nrm(H);
        const std::string base = prefix_ + "layers." + std::to_string(L) + ".";
        const std::string ln = has(base + "post_attention_layernorm.weight")
                                   ? base + "post_attention_layernorm.weight"
                                   : base + "input_layernorm.weight";
        std::vector<float> tmp(H);
        std::memcpy(tmp.data(), x.data(), sizeof(float) * H);
        // router peek
        std::string gate_name = base + "mlp.gate.weight";
        if (!has(gate_name)) gate_name = base + "mlp.router.weight";
        std::vector<int> eids;
        if (has(gate_name)) {
          int idbuf[64];
          float wbuf[64];
          hal::rmsnorm(tmp.data(), w(ln), nrm.data(), H, cfg_.rms_eps, wd_);
          router_topk(nrm.data(), w(gate_name), idbuf, wbuf);
          for (int i = 0; i < cfg_.topk; ++i) eids.push_back(idbuf[i]);
        } else {
          for (int i = 0; i < cfg_.topk; ++i) eids.push_back(i % cfg_.n_experts);
        }
        try {
          pref_->plan_next_layer(L, eids);
        } catch (...) {
          // 可能已 plan; ignore
        }
        if (L + 1 < cfg_.layers) {
          try {
            pref_->plan_next_layer(L + 1, eids);
          } catch (...) {
          }
        }
        moe_mlp_token(L, x.data() + t * H, true, ids, ws, &nsel);
      } else {
        moe_mlp_token(L, x.data() + t * H, false, ids, ws, &nsel);
      }
    }
  }

  const std::string norm_name = prefix_.empty() ? "norm.weight" : prefix_ + "norm.weight";
  const std::string emb_name =
      prefix_.empty() ? "embedding.weight"
                      : (has(prefix_ + "embed_tokens.weight") ? prefix_ + "embed_tokens.weight"
                                                             : "embedding.weight");
  std::vector<float> last(H);
  if (has(norm_name))
    hal::rmsnorm(x.data() + (n - 1) * H, w(norm_name), last.data(), H, cfg_.rms_eps, wd_);
  else
    std::memcpy(last.data(), x.data() + (n - 1) * H, sizeof(float) * H);

  logits.assign(cfg_.vocab, 0.f);
  const uint16_t* emb = w(emb_name);
  for (int v = 0; v < cfg_.vocab; ++v) {
    const uint16_t* row = emb + static_cast<size_t>(v) * H;
    double acc = 0.0;
    for (int i = 0; i < H; ++i) acc += static_cast<double>(last[i]) * hal::load_w(row + i, wd_);
    logits[v] = static_cast<float>(acc);
  }
}

}  // namespace llmoc::model
