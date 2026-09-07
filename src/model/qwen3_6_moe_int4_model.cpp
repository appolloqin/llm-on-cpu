// llm-on-cpu :: model/qwen3_6_moe_int4_model.cpp
// MoE FFN + AutoAWQ/GPTQ load guards. Do NOT change qlwc::kLocalAwqSymZero here.
#include "model/qwen3_6_moe_int4_model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <future>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.h"
#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"
#include "hal/int4_ops.h"
#include "moe/qlwc_expert_bytes.h"
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

moe::MoeBackend parse_backend(const std::string& s) {
  if (s == "cpu") return moe::MoeBackend::kCpu;
  if (s == "offload") return moe::MoeBackend::kOffload;
  return moe::MoeBackend::kHybrid;
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

void Qwen36MoeInt4Model::init_moe_offload(const MoeOffloadRuntimeConfig& cfg) {
  moe_rt_ = cfg;
  if (!store_ || !cfg_.is_moe) return;

  const auto& hdr = store_->header();
  moe::HostBanksConfig hcfg;
  hcfg.num_layers = cfg_.layers;
  hcfg.num_experts = cfg_.n_experts;
  hcfg.hidden = cfg_.hidden;
  hcfg.intermediate = cfg_.moe_intermediate > 0 ? cfg_.moe_intermediate : cfg_.intermediate;
  hcfg.group_size = static_cast<int>(hdr.group_size ? hdr.group_size : 128);
  hcfg.has_zeros = (hdr.scheme == qlwc::Scheme::kGptqAsym);
  hcfg.scheme = hdr.scheme;
  hcfg.awq_zp = qlwc::awq_zero_point(hdr.scheme);
  hcfg.name_prefix = prefix_;
  hcfg.host_pin = cfg.host_pin;
  if (cfg.dram_hot_gb > 0) {
    const size_t hot = static_cast<size_t>(cfg.dram_hot_gb * (1ull << 30));
    const size_t already = store_->loaded_bytes();
    const size_t margin = 2ull << 30;  // OS + working set headroom
    if (hot > already + margin) {
      hcfg.dram_budget_bytes = hot - already - margin;
    } else {
      hcfg.dram_budget_bytes = hot / 2;
    }
    LOG_INFO("moe host_banks dram budget=%.2fGiB (dram_hot=%.2f store_now=%.2f margin=2.0)",
             hcfg.dram_budget_bytes / double(1ull << 30), cfg.dram_hot_gb,
             already / double(1ull << 30));
  }

  const size_t per_exp_host =
      moe::expert_storage_bytes(hcfg.hidden, hcfg.intermediate, hcfg.group_size, hcfg.has_zeros);
  LOG_INFO("moe host_banks: filling experts from QLWC (layers=%d E=%d ~%.2fMiB/expert)…",
           hcfg.num_layers, hcfg.num_experts, per_exp_host / double(1 << 20));

  moe_banks_ = std::make_unique<moe::QlwcExpertHostBanks>();
  moe_banks_->configure(hcfg);
  moe_banks_->allocate();
  moe_banks_->fill_from_qlwc(*store_);
  moe_banks_->pin();

  const int E = cfg_.n_experts;
  int slots = cfg.cache_slots;
  const size_t per_exp = moe::expert_device_bytes(hcfg.hidden, hcfg.intermediate, hcfg.group_size,
                                                   hcfg.has_zeros);
  if (slots <= 0 && hal::cuda::enabled()) {
    // Auto: aim for >= 2E (prefill dbuf) plus decode headroom from remaining budget.
    const size_t bud = hal::cuda::vram_budget();
    const size_t used =hal::cuda::vram_used();
    const size_t free_b = bud > used ? bud - used : 0;
    // MoE-first: take up to ~40% of budget for slots (attn pin happens after reserve).
    const size_t moe_cap = (std::max)(free_b, bud / 2);
    slots = per_exp > 0 ? static_cast<int>(moe_cap / per_exp) : (2 * E);
    if (cfg.prefill_overlap) slots = (std::max)(slots, 2 * E);
    slots = (std::min)(slots, (std::max)(2 * E, E * 4));  // sane upper for unit tests / small VRAM
  }
  if (slots <= 0) slots = (std::max)(2 * E, E);  // CPU-only still needs bookkeeping
  if (cfg.prefill_overlap && slots < 2 * E) slots = 2 * E;

  moe::OffloadCacheConfig ocfg;
  ocfg.num_layers = cfg_.layers;
  ocfg.num_experts = E;
  ocfg.cache_size = slots;
  ocfg.prefill_overlap = cfg.prefill_overlap && slots >= 2 * E;
  ocfg.backend = parse_backend(cfg.backend);
  ocfg.hybrid_fetch_frac = cfg.hybrid_fetch_frac;
  if (!hal::cuda::enabled() || !moe_gpu_experts_enabled()) {
    ocfg.backend = moe::MoeBackend::kCpu;
  }
#if !defined(_OPENMP)
  if (ocfg.backend == moe::MoeBackend::kHybrid) ocfg.backend = moe::MoeBackend::kOffload;
#endif

  moe_cache_ = std::make_unique<moe::OffloadMoeCache>();
  moe_cache_->configure(ocfg, moe_banks_.get());
  moe_slot_reserve_bytes_ = static_cast<size_t>(slots) * per_exp;

  const char* be =
      ocfg.backend == moe::MoeBackend::kCpu
          ? "cpu"
          : (ocfg.backend == moe::MoeBackend::kHybrid ? "hybrid" : "offload");
  LOG_INFO("moe_slots=%d reserve=%.2fGiB backend=%s prefill_overlap=%d hybrid_fetch_frac=%.2f",
           slots, moe_slot_reserve_bytes_ / double(1ull << 30), be, ocfg.prefill_overlap ? 1 : 0,
           ocfg.hybrid_fetch_frac);
}

void Qwen36MoeInt4Model::on_prefill_begin() {
  if (moe_cache_ && moe_cache_->ready()) moe_cache_->begin_prefill();
}

void Qwen36MoeInt4Model::on_prefill_layer(int layer, int /*n_tok*/) {
  if (!moe_cache_ || !moe_cache_->ready() || !moe_cache_->prefill_active()) return;
  if (!layers_[static_cast<size_t>(layer)].is_moe) return;
  moe_cache_->prefetch_prefill_layer(layer);
  if (layer + 1 < cfg_.layers && layers_[static_cast<size_t>(layer + 1)].is_moe) {
    moe_cache_->prefetch_prefill_layer(layer + 1);
  }
  moe_cache_->wait_prefill_layer(layer);
}

void Qwen36MoeInt4Model::on_prefill_layer_done(int layer) {
  if (!moe_cache_ || !moe_cache_->ready() || !moe_cache_->prefill_active()) return;
  moe_cache_->release_prefill_layer(layer);
}

void Qwen36MoeInt4Model::moe_cpu_experts(int layer, const float* normed, float* down_acc,
                                         const std::vector<int>& experts,
                                         const std::vector<float>& logits, double wsum) {
  const int H = cfg_.hidden;
  const int I = cfg_.moe_intermediate > 0 ? cfg_.moe_intermediate : cfg_.intermediate;
  if (experts.empty()) return;

  auto run_one = [&](int e, float* local) {
    const float ww = static_cast<float>(logits[static_cast<size_t>(e)] / wsum);
    std::vector<float> g(static_cast<size_t>(I)), u(static_cast<size_t>(I)),
        mid(static_cast<size_t>(I)), down(static_cast<size_t>(H));
    if (moe_banks_ && moe_banks_->has_layer(layer)) {
      hal::gemm_int4(normed, moe_banks_->gate(layer, e), g.data());
      hal::gemm_int4(normed, moe_banks_->up(layer, e), u.data());
      hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
      hal::gemm_int4(mid.data(), moe_banks_->down(layer, e), down.data());
    } else {
      const std::string eb = prefix_ + "layers." + std::to_string(layer) + ".mlp.experts." +
                             std::to_string(e) + ".";
      if (store_->lazy()) {
        store_->ensure(eb + "gate_proj.weight");
        store_->ensure(eb + "up_proj.weight");
        store_->ensure(eb + "down_proj.weight");
      }
      gemm_w(normed, eb + "gate_proj.weight", g.data(), I, H);
      gemm_w(normed, eb + "up_proj.weight", u.data(), I, H);
      hal::silu_and_mul(g.data(), u.data(), mid.data(), I);
      gemm_w(mid.data(), eb + "down_proj.weight", down.data(), H, I);
    }
    for (int d = 0; d < H; ++d) local[d] += ww * down[static_cast<size_t>(d)];
  };

#if defined(_OPENMP)
#pragma omp parallel
  {
    std::vector<float> local(static_cast<size_t>(H), 0.f);
#pragma omp for schedule(static)
    for (int i = 0; i < static_cast<int>(experts.size()); ++i) {
      run_one(experts[static_cast<size_t>(i)], local.data());
    }
#pragma omp critical
    for (int d = 0; d < H; ++d) down_acc[d] += local[static_cast<size_t>(d)];
  }
#else
  std::vector<float> local(static_cast<size_t>(H), 0.f);
  for (int e : experts) run_one(e, local.data());
  for (int d = 0; d < H; ++d) down_acc[d] += local[static_cast<size_t>(d)];
#endif
}

void Qwen36MoeInt4Model::moe_ffn_token_offload(int layer, const float* normed, float* down_acc,
                                               const std::vector<int>& order,
                                               const std::vector<float>& logits, double wsum,
                                               int K) {
  const auto& lp = layers_[layer];
  const int H = cfg_.hidden;
  const int I = cfg_.moe_intermediate > 0 ? cfg_.moe_intermediate : cfg_.intermediate;

  std::vector<int> topk(static_cast<size_t>(K));
  for (int i = 0; i < K; ++i) topk[static_cast<size_t>(i)] = order[static_cast<size_t>(i)];

  const auto be = moe_cache_->config().backend;
  if (be == moe::MoeBackend::kHybrid) {
    moe_cache_->ensure_experts_hybrid(layer, topk.data(), K);
  } else if (be == moe::MoeBackend::kCpu) {
    moe_cache_->ensure_experts(layer, topk.data(), K);
  } else {
    moe_cache_->ensure_experts(layer, topk.data(), K);
  }

  // Overlap CPU overflow with PCIe copy when hybrid.
  std::future<void> cpu_fut;
  std::vector<float> cpu_acc;
  const auto& overflow = moe_cache_->overflow_experts();
  if (!overflow.empty()) {
    cpu_acc.assign(static_cast<size_t>(H), 0.f);
    cpu_fut = std::async(std::launch::async, [&] {
      moe_cpu_experts(layer, normed, cpu_acc.data(), overflow, logits, wsum);
    });
  }

  const int ncopy = moe_cache_->copy_missing(layer);
  if (ncopy > 0 && moe_cache_->last_fetch_count() > 0) {
    static std::atomic<int> hybrid_logs{0};
    if (be == moe::MoeBackend::kHybrid && hybrid_logs.fetch_add(1) < 3) {
      LOG_INFO("hybrid fetch=%d copy_bytes=%.2fMiB overflow=%zu layer=%d",
               moe_cache_->last_fetch_count(),
               moe_cache_->last_copy_bytes() / double(1 << 20), overflow.size(), layer);
    }
  }

  // GPU fused path for slot-resident experts.
  qlwc::Int4View gates[64], ups[64], downs[64];
  hal::cuda::MoeExpertInt4 exp_views[64];
  int n_gpu = 0;
  for (int i = 0; i < K; ++i) {
    const int slot = topk[static_cast<size_t>(i)];
    if (slot < 0) continue;
    if (!moe_cache_->views_for_slot(slot, gates[n_gpu], ups[n_gpu], downs[n_gpu])) continue;
    const int e = order[static_cast<size_t>(i)];
    exp_views[n_gpu].gate = &gates[n_gpu];
    exp_views[n_gpu].up = &ups[n_gpu];
    exp_views[n_gpu].down = &downs[n_gpu];
    exp_views[n_gpu].weight = static_cast<float>(logits[static_cast<size_t>(e)] / wsum);
    ++n_gpu;
  }

  bool gpu_ok = false;
  if (n_gpu > 0 && moe_gpu_experts_enabled() &&hal::cuda::enabled()) {
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
    std::vector<float> gpu_y(static_cast<size_t>(H), 0.f);
    if (hal::cuda::try_moe_ffn_int4(normed, H, I, exp_views, n_gpu, sg, su, sd, shared_scale,
                                    gpu_y.data())) {
      for (int d = 0; d < H; ++d) down_acc[d] += gpu_y[static_cast<size_t>(d)];
      gpu_ok = true;
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
    }
  }

  if (!gpu_ok && n_gpu > 0) {
    // Fallback: per-expert host GEMM for intended GPU set.
    std::vector<int> fallback;
    for (int i = 0; i < K; ++i) {
      if (topk[static_cast<size_t>(i)] >= 0) fallback.push_back(order[static_cast<size_t>(i)]);
    }
    moe_cpu_experts(layer, normed, down_acc, fallback, logits, wsum);
  }

  if (cpu_fut.valid()) {
    cpu_fut.get();
    for (int d = 0; d < H; ++d) down_acc[d] += cpu_acc[static_cast<size_t>(d)];
  } else if (!overflow.empty() && !gpu_ok) {
    // Already handled if cpu path ran; if no async, run sync.
  }

  // Shared expert when GPU fused path did not run (or CPU-only).
  if (!gpu_ok && (lp.shared_gate.pass || lp.shared_gate.is_int4)) {
    const int Is = lp.shared_gate.M > 0 ? lp.shared_gate.M : I;
    std::vector<float> g(static_cast<size_t>(Is)), u(static_cast<size_t>(Is)),
        mid(static_cast<size_t>(Is)), down(static_cast<size_t>(H));
    if (lp.shared_gate.is_int4 && lp.shared_up.is_int4 && lp.shared_down.is_int4) {
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

  if (moe_cache_ && moe_cache_->ready() && moe_banks_ && moe_banks_->has_layer(layer)) {
    moe_ffn_token_offload(layer, normed, down_acc, order, logits, wsum, K);
    return;
  }

  // Legacy path: sync store_->ensure (no host banks).
  const std::string base = prefix_ + "layers." + std::to_string(layer) + ".mlp.experts.";
  const bool gpu_exp = moe_gpu_experts_enabled() &&hal::cuda::enabled();

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
  }

  std::vector<int> exps(order.begin(), order.begin() + K);
  moe_cpu_experts(layer, normed, down_acc, exps, logits, wsum);

  if (lp.shared_gate.pass || lp.shared_gate.is_int4) {
    const int Is = lp.shared_gate.M > 0 ? lp.shared_gate.M : I;
    std::vector<float> g(static_cast<size_t>(Is)), u(static_cast<size_t>(Is)),
        mid(static_cast<size_t>(Is)), down(static_cast<size_t>(H));
    if (lp.shared_gate.is_int4 && lp.shared_up.is_int4 && lp.shared_down.is_int4) {
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
