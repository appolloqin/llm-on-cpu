#pragma once
// llm-on-cpu :: sched/placement_planner.h — active-set × mesh (IMPLEMENTATION 2026-08-31)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "contracts/device_mesh.h"
#include "contracts/family_pack.h"
#include "sched/mode_controller.h"

namespace llmoc::sched {

struct ExpertHint {
  int layer = 0;
  int expert = 0;
  double freq = 0.0;
  uint64_t bytes = 0;
};

struct PlacementPlan {
  ExecMode mode = ExecMode::kPureCpu;
  uint64_t vram_budget = 0;           // aggregate or primary
  uint64_t vram_budget_per_rank = 0;  // G8
  uint64_t dram_budget = 0;
  uint64_t vram_attn_reserve = 0;
  uint64_t vram_expert_budget = 0;
  uint64_t expert_slots_per_rank = 0;
  int n_experts = 0;
  int top_k = 0;
  bool ok = true;
  std::string error;
  std::vector<ExpertHint> vram_experts;  // optional hot cache (hybrid)
  std::vector<ExpertHint> dram_experts;
  contracts::DeviceMesh mesh;
  std::string summary;
};

class PlacementPlanner {
 public:
  struct Config {
    uint64_t vram_bytes = 0;       // per-rank VRAM budget
    uint64_t dram_bytes = 0;
    double attn_reserve_frac = 0.6;
    uint64_t margin_bytes = 512ull << 20;
    int expert_slot_extra = 1;  // prefetch depth in layers' worth of top-k
    bool strict_vram = true;
  };

  // Legacy bagging (kept for M5 tests / hybrid optional hot experts).
  static PlacementPlan solve(ExecMode mode, const Config& cfg,
                             const std::vector<ExpertHint>& experts);

  // Active-set plan (pure_gpu / hybrid primary path).
  static PlacementPlan solve_active(ExecMode mode, const Config& cfg,
                                    const contracts::DeviceMesh& mesh,
                                    const contracts::ActiveSetProfile& profile);
};

}  // namespace llmoc::sched
