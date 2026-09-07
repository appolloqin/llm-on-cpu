// llm-on-cpu :: model/qwen3_6_moe_int4_model.cpp
// MoE FFN + AutoAWQ/GPTQ load guards. Do NOT change qlwc::kLocalAwqSymZero here.
#include "model/qwen3_6_moe_int4_model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.h"
#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"
#include "hal/int4_ops.h"
#include "weights/qlwc_format.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace llmoc::model {
namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

bool moe_gpu_experts_enabled() {
  static const int mode = [] {
    const char* e = std::getenv("LLMOC_MOE_GPU_EXPERTS");
    if (!e || !e[0]) return 1;
    if (e[0] == '0' && e[1] == '\0') return 0;
    return 1;
  }();
  return mode != 0;
}

}  // namespace

void Qwen36MoeInt4Model::load(qlwc::QlwcStore* store, const std::string& hf_config_json_path) {
  allow_moe_ = true;
  Qwen35Int4Model::load(store, hf_config_json_path);
  if (!cfg_.is_moe) {
    throw std::runtime_error(std::string(int4_class_name()) + ": config/weights are not MoE");
  }
  meta_.kind = int4_kind();
  const auto sch = store_->header().scheme;
  // AutoAWQ zero_point:true must be gptq_asym. Never "fix" by flipping global awq_sym zp.
  if (sch == qlwc::Scheme::kAwqSym) {
    throw std::runtime_error(
        std::string(int4_class_name()) +
        ": QLWC scheme=awq_sym is for local dense quant (zp=" +
        std::to_string(qlwc::kLocalAwqSymZero) +
        "). AutoAWQ MoE must be re-imported as gptq_asym via import_awq_hf_qlwc.mjs.");
  }
  LOG_INFO("%s ready: experts=%d topk=%d scheme=gptq_asym local_awq_zp=%d kind=%s",
           int4_class_name(), cfg_.n_experts, cfg_.topk, qlwc::kLocalAwqSymZero, int4_kind());
}

void Qwen36MoeInt4Model::moe_ffn_token(int layer, const float* normed, float* down_acc) {
  const auto& lp = layers_[layer];
  const int H = cfg_.hidden;
  const int E = cfg_.n_experts;
  const int K = cfg_.topk;
  const int I = cfg_.moe_intermediate > 0 ? cfg_.moe_intermediate : cfg_.intermediate;
  if (E <= 0 || K <= 0) throw std::runtime_error("moe_ffn: invalid experts/topk");
  if (K > 64) throw std::runtime_error("moe_ffn: topk > 64");

  std::vector<float> logits(static_cast<size_t>(E));
  gemm_opt(normed, lp.router, logits.data());
  float m = *std::max_element(logits.begin(), logits.end());
  double s = 0.0;
  for (int i = 0; i < E; ++i) {
    logits[i] = std::exp(logits[i] - m);
    s += logits[i];
  }
  for (int i = 0; i < E; ++i) logits[i] = static_cast<float>(logits[i] / s);

  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + K, order.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  double wsum = 0.0;
  for (int i = 0; i < K; ++i) wsum += logits[order[i]];
  if (wsum < 1e-12) wsum = 1.0;

  std::fill(down_acc, down_acc + H, 0.f);
  const std::string base = prefix_ + "layers." + std::to_string(layer) + ".mlp.experts.";
  const bool gpu_exp = moe_gpu_experts_enabled() && hal::cuda::enabled();

  // Ensure expert tensors on this thread before any OpenMP region (store ensure not parallel-safe).
  qlwc::Int4View gates[64], ups[64], downs[64];
  hal::cuda::MoeExpertInt4 exp_views[64];
  bool all_int4 = true;
  for (int i = 0; i < K; ++i) {
    const int e = order[static_cast<size_t>(i)];
    const std::string eb = base + std::to_string(e) + ".";
    if (store_->lazy()) {
      store_->ensure(eb + "gate_proj.weight");
      store_->ensure(eb + "up_proj.weight");
      store_->ensure(eb + "down_proj.weight");
    }
  }
  for (int i = 0; i < K; ++i) {
    const int e = order[static_cast<size_t>(i)];
    const std::string eb = base + std::to_string(e) + ".";
    const std::string gn = eb + "gate_proj.weight";
    const std::string un = eb + "up_proj.weight";
    const std::string dn = eb + "down_proj.weight";
    if (!is_int4(gn) || !is_int4(un) || !is_int4(dn)) {
      all_int4 = false;
      continue;
    }
    gates[i] = store_->get_int4(gn);
    ups[i] = store_->get_int4(un);
    downs[i] = store_->get_int4(dn);
    // Shape must match MoE intermediate (import/config mismatch → host fallback).
    if (gates[i].M != I || gates[i].K != H || ups[i].M != I || ups[i].K != H ||
        downs[i].M != H || downs[i].K != I) {
      all_int4 = false;
      continue;
    }
    exp_views[i].gate = &gates[i];
    exp_views[i].up = &ups[i];
    exp_views[i].down = &downs[i];
    exp_views[i].weight = static_cast<float>(logits[static_cast<size_t>(e)] / wsum);
  }

  // Device fused MoE: one H2D(x) + top-k SwiGLU (+ INT4 shared) + one D2H.
  if (gpu_exp && all_int4) {
    const qlwc::Int4View* sg = nullptr;
    const qlwc::Int4View* su = nullptr;
    const qlwc::Int4View* sd = nullptr;
    float shared_scale = 0.f;
    const bool shared_i4 =
        lp.shared_gate.is_int4 && lp.shared_up.is_int4 && lp.shared_down.is_int4 && 
        lp.shared_gate.i4.M == I && lp.shared_up.i4.M == I && lp.shared_down.i4.M == H;
    if (shared_i4) {
      shared_scale = 1.f;
      if (lp.shared_expert_gate.pass || lp.shared_expert_gate.is_int4) {
        float gate_logit = 0.f;
        gemm_opt(normed, lp.shared_expert_gate, &gate_logit);
        shared_scale = sigmoid(gate_logit);
      }
      sg = &lp.shared_gate.i4;
      su = &lp.shared_up.i4;
      sd = &lp.shared_down.i4;
    }
    if (hal::cuda::try_moe_ffn_int4(normed, H, I, exp_views, K, sg, su, sd, shared_scale,
                                    down_acc)) {
      // Shared was BF16/F16 pass (common in AutoAWQ ignore lists) — add via gemm_opt.
      if (!shared_i4 && (lp.shared_gate.pass || lp.shared_gate.is_int4)) {
        const int Is = lp.shared_gate.M > 0 ? lp.shared_gate.M : I;
        std::vector<float> g(static_cast<size_t>(Is)), u(static_cast<size_t>(Is)),
            mid(static_cast<size_t>(Is)), down(static_cast<size_t>(H));
        gemm_opt(normed, lp.shared_gate, g.data());
        gemm_opt(normed, lp.shared_up, u.data());
        hal::silu_and_mul(g.data(), u.data(), mid.data(), Is);
        gemm_opt(mid.data(), lp.shared_down, down.data());
        float scale = 1.f;
        if (lp.shared_expert_gate.pass || lp.shared_expert_gate.is_int4) {
          float gate_logit = 0.f;
          gemm_opt(normed, lp.shared_expert_gate, &gate_logit);
          scale = sigmoid(gate_logit);
        }
        for (int d = 0; d < H; ++d) down_acc[d] += scale * down[static_cast<size_t>(d)];
      }
      return;
    }
    static std::atomic<int> moe_fuse_fail_logs{0};
    if (moe_fuse_fail_logs.fetch_add(1) < 3) {
      LOG_WARN("MoE fused GPU FFN failed (layer=%d); falling back to per-GEMV", layer);
    }
  }

  if (!gpu_exp) {
#if defined(_OPENMP)
#pragma omp parallel
    {
      std::vector<float> g(static_cast<size_t>(I)), u(static_cast<size_t>(I)),
          mid(static_cast<size_t>(I)), down(static_cast<size_t>(H));
      std::vector<float> local(static_cast<size_t>(H), 0.f);
#pragma omp for schedule(static)
      for (int i = 0; i < K; ++i) {
        const int e = order[static_cast<size_t>(i)];
        const float ww = static_cast<float>(logits[static_cast<size_t>(e)] / wsum);
        const std::string eb = base + std::to_string(e) + ".";
        gemm_w(normed, eb + "gate_proj.weight", g.data(), I, H);
        gemm_w(normed, eb + "up_proj.weight", u.data(), I, H);
        hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
        gemm_w(mid.data(), eb + "down_proj.weight", down.data(), H, I);
        for (int d = 0; d < H; ++d) local[static_cast<size_t>(d)] += ww * down[static_cast<size_t>(d)];
      }
#pragma omp critical
      for (int d = 0; d < H; ++d) down_acc[d] += local[static_cast<size_t>(d)];
    }
#else
    std::vector<float> g(static_cast<size_t>(I)), u(static_cast<size_t>(I)),
        mid(static_cast<size_t>(I)), down(static_cast<size_t>(H));
    for (int i = 0; i < K; ++i) {
      const int e = order[static_cast<size_t>(i)];
      const float ww = static_cast<float>(logits[static_cast<size_t>(e)] / wsum);
      const std::string eb = base + std::to_string(e) + ".";
      gemm_w(normed, eb + "gate_proj.weight", g.data(), I, H);
      gemm_w(normed, eb + "up_proj.weight", u.data(), I, H);
      hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
      gemm_w(mid.data(), eb + "down_proj.weight", down.data(), H, I);
      for (int d = 0; d < H; ++d) down_acc[d] += ww * down[static_cast<size_t>(d)];
    }
#endif
  } else {
    // Fallback: per-GEMV GPU (sticky x) if fused MoE failed.
    std::vector<float> g(static_cast<size_t>(I)), u(static_cast<size_t>(I)),
        mid(static_cast<size_t>(I)), down(static_cast<size_t>(H));
    for (int i = 0; i < K; ++i) {
      const int e = order[static_cast<size_t>(i)];
      const float ww = static_cast<float>(logits[static_cast<size_t>(e)] / wsum);
      const std::string eb = base + std::to_string(e) + ".";
      gemm_w(normed, eb + "gate_proj.weight", g.data(), I, H);
      gemm_w(normed, eb + "up_proj.weight", u.data(), I, H);
     hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
      gemm_w(mid.data(), eb + "down_proj.weight", down.data(), H, I);
      for (int d = 0; d < H; ++d) down_acc[d] += ww * down[static_cast<size_t>(d)];
    }
  }

  if (lp.shared_gate.pass || lp.shared_gate.is_int4) {
    const int Is = lp.shared_gate.M > 0 ? lp.shared_gate.M : I;
    std::vector<float> g(static_cast<size_t>(Is)), u(static_cast<size_t>(Is)),
        mid(static_cast<size_t>(Is)), down(static_cast<size_t>(H));
    if (!gpu_exp && lp.shared_gate.is_int4 && lp.shared_up.is_int4 && lp.shared_down.is_int4) {
     hal::gemm_int4(normed, lp.shared_gate.i4, g.data());
     hal::gemm_int4(normed, lp.shared_up.i4, u.data());
     hal::silu_and_mul(g.data(), u.data(), mid.data(), Is);
     hal::gemm_int4(mid.data(), lp.shared_down.i4, down.data());
    } else {
      gemm_opt(normed, lp.shared_gate, g.data());
      gemm_opt(normed, lp.shared_up, u.data());
     hal::silu_and_mul(g.data(), u.data(), mid.data(), Is);
      gemm_opt(mid.data(), lp.shared_down, down.data());
    }
    float scale = 1.f;
    if (lp.shared_expert_gate.pass || lp.shared_expert_gate.is_int4) {
      float gate_logit = 0.f;
      gemm_opt(normed, lp.shared_expert_gate, &gate_logit);
      scale = sigmoid(gate_logit);
    }
    for (int d = 0; d < H; ++d) down_acc[d] += scale * down[static_cast<size_t>(d)];
  }
}


}  // namespace llmoc::model
