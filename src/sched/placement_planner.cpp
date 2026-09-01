// llm-on-cpu :: sched/placement_planner.cpp
#include "sched/placement_planner.h"

#include <algorithm>
#include <sstream>

namespace llmoc::sched {
namespace {

int experts_owned_count(const contracts::DeviceMesh& mesh, int n_experts, int rank) {
  if (n_experts <= 0) return 0;
  if (mesh.ep_size <= 1) return n_experts;
  int n = 0;
  for (int e = 0; e < n_experts; ++e) {
    if (mesh.owns_expert(rank % mesh.ep_size, e, n_experts)) ++n;
  }
  return n;
}

}  // namespace

PlacementPlan PlacementPlanner::solve(ExecMode mode, const Config& cfg,
                                      const std::vector<ExpertHint>& experts) {
  PlacementPlan plan;
  plan.mode = mode;
  plan.vram_budget = cfg.vram_bytes;
  plan.vram_budget_per_rank = cfg.vram_bytes;
  plan.dram_budget = cfg.dram_bytes;
  plan.ok = true;

  if (mode == ExecMode::kPureCpu) {
    plan.vram_attn_reserve = 0;
    plan.vram_expert_budget = 0;
    plan.dram_experts = experts;
    plan.summary = "pure_cpu: all experts DRAM/NVMe";
    return plan;
  }

  if (mode == ExecMode::kLayerStream) {
    plan.vram_attn_reserve = 0;
    plan.vram_expert_budget = 0;
    plan.dram_experts = experts;
    plan.summary =
        "layer_stream: pin window_layers only (run-first; see docs/DESIGN_LAYER_STREAM.md); "
        "forward wiring = S1+";
    return plan;
  }

  double frac = cfg.attn_reserve_frac;
  if (frac < 0.1) frac = 0.1;
  if (frac > 0.9) frac = 0.9;
  if (mode == ExecMode::kPureGpu) {
    // Legacy path: still bag by freq, but prefer not claiming "full model fit".
    plan.vram_attn_reserve = cfg.vram_bytes / 10;
    plan.vram_expert_budget = cfg.vram_bytes > plan.vram_attn_reserve
                                  ? cfg.vram_bytes - plan.vram_attn_reserve
                                  : 0;
  } else {
    plan.vram_attn_reserve = static_cast<uint64_t>(cfg.vram_bytes * frac);
    plan.vram_expert_budget = cfg.vram_bytes > plan.vram_attn_reserve
                                  ? cfg.vram_bytes - plan.vram_attn_reserve
                                  : 0;
  }

  std::vector<ExpertHint> sorted = experts;
  std::sort(sorted.begin(), sorted.end(),
            [](const ExpertHint& a, const ExpertHint& b) { return a.freq > b.freq; });

  uint64_t used_v = 0, used_d = 0;
  for (const auto& e : sorted) {
    if (e.bytes == 0) continue;
    if (used_v + e.bytes <= plan.vram_expert_budget) {
      plan.vram_experts.push_back(e);
      used_v += e.bytes;
    } else if (used_d + e.bytes <= plan.dram_budget) {
      plan.dram_experts.push_back(e);
      used_d += e.bytes;
    } else {
      plan.dram_experts.push_back(e);
    }
  }

  std::ostringstream os;
  os << (mode == ExecMode::kPureGpu ? "pure_gpu" : "hybrid_gpu")
     << ": vram_experts=" << plan.vram_experts.size() << " dram_experts=" << plan.dram_experts.size()
     << " vram_used=" << (used_v / (1.0 * (1ull << 30))) << "GiB"
     << " attn_reserve=" << (plan.vram_attn_reserve / (1.0 * (1ull << 30))) << "GiB";
  plan.summary = os.str();
  return plan;
}

PlacementPlan PlacementPlanner::solve_active(ExecMode mode, const Config& cfg,
                                             const contracts::DeviceMesh& mesh,
                                             const contracts::ActiveSetProfile& profile) {
  PlacementPlan plan;
  plan.mode = mode;
  plan.mesh = mesh;
  plan.vram_budget = cfg.vram_bytes * static_cast<uint64_t>(std::max(1, mesh.world_size));
  plan.vram_budget_per_rank = cfg.vram_bytes;
  plan.dram_budget = cfg.dram_bytes;
  plan.n_experts = profile.n_experts;
  plan.top_k = profile.top_k;
  plan.ok = true;

  if (mode == ExecMode::kPureCpu) {
    plan.summary = "pure_cpu active-set: DRAM/NVMe experts";
    return plan;
  }

  if (mode == ExecMode::kLayerStream) {
    plan.summary =
        "layer_stream active-set: layer window only (DESIGN_LAYER_STREAM); not full residency";
    return plan;
  }

  const int tp = std::max(1, mesh.tp_size);
  const uint64_t attn_share = profile.per_layer_attn_bytes / static_cast<uint64_t>(tp);
  const uint64_t shared_share = profile.shared_bytes / static_cast<uint64_t>(tp);
  const uint64_t kv_share = profile.kv_working_bytes;  // may be replicated; conservative

  if (mode == ExecMode::kHybridGpu) {
    plan.vram_attn_reserve = attn_share + shared_share + kv_share + profile.workspace_bytes;
    const uint64_t need = plan.vram_attn_reserve + cfg.margin_bytes;
    if (cfg.strict_vram && need > cfg.vram_bytes) {
      plan.ok = false;
      plan.error = "hybrid_gpu: attn+KV working set exceeds per-rank VRAM";
      plan.summary = plan.error;
      return plan;
    }
    std::ostringstream os;
    os << "hybrid_gpu " << mesh.summary() << " attn_kv_mb=" << (plan.vram_attn_reserve >> 20)
       << " experts=cpu";
    plan.summary = os.str();
    return plan;
  }

  // pure_gpu: per-rank worst-case expert slots
  const int top_k = std::max(0, profile.top_k);
  const int extra = std::max(0, cfg.expert_slot_extra);
  uint64_t max_need = 0;
  for (int r = 0; r < std::max(1, mesh.ep_size); ++r) {
    const int owned = experts_owned_count(mesh, profile.n_experts, r);
    const int slots = std::min(top_k, owned > 0 ? owned : top_k) + (extra * top_k);
    plan.expert_slots_per_rank = static_cast<uint64_t>(std::max(slots, 1));
    const uint64_t expert_bytes =
        plan.expert_slots_per_rank * profile.expert_bytes;
    const uint64_t need =
        attn_share + shared_share + kv_share + expert_bytes + profile.workspace_bytes +
        cfg.margin_bytes;
    max_need = std::max(max_need, need);
    if (cfg.strict_vram && need > cfg.vram_bytes) {
      plan.ok = false;
      std::ostringstream es;
      es << "pure_gpu: active working set " << (need >> 20) << " MB > vram budget "
         << (cfg.vram_bytes >> 20) << " MB - margin (rank" << r << ")";
      plan.error = es.str();
      plan.summary = plan.error;
      return plan;
    }
  }

  plan.vram_attn_reserve = attn_share + shared_share + kv_share;
  plan.vram_expert_budget =
      cfg.vram_bytes > plan.vram_attn_reserve ? cfg.vram_bytes - plan.vram_attn_reserve : 0;

  std::ostringstream os;
  os << "pure_gpu " << mesh.summary() << " slots_per_rank=" << plan.expert_slots_per_rank
     << " W_active_est_mb=" << (max_need >> 20)
     << " vram_budget_mb_per_rank=" << (cfg.vram_bytes >> 20);
  plan.summary = os.str();
  return plan;
}

}  // namespace llmoc::sched
