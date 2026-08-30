// llm-on-cpu :: sched/placement_planner.cpp
#include "sched/placement_planner.h"

#include <algorithm>
#include <sstream>

namespace llmoc::sched {

PlacementPlan PlacementPlanner::solve(ExecMode mode, const Config& cfg,
                                      const std::vector<ExpertHint>& experts) {
  PlacementPlan plan;
  plan.mode = mode;
  plan.vram_budget = cfg.vram_bytes;
  plan.dram_budget = cfg.dram_bytes;

  if (mode == ExecMode::kPureCpu) {
    plan.vram_attn_reserve = 0;
    plan.vram_expert_budget = 0;
    plan.dram_experts = experts;
    plan.summary = "pure_cpu: all experts DRAM/NVMe";
    return plan;
  }

  double frac = cfg.attn_reserve_frac;
  if (frac < 0.1) frac = 0.1;
  if (frac > 0.9) frac = 0.9;
  if (mode == ExecMode::kPureGpu) {
    plan.vram_attn_reserve = cfg.vram_bytes / 10;  // small reserve for KV/activations
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
      plan.dram_experts.push_back(e);  // overflow → DRAM/NVMe path
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

}  // namespace llmoc::sched
