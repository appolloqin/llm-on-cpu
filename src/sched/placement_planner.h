#pragma once
// llm-on-cpu :: sched/placement_planner.h — M5 PlacementPlanner (ARCHITECTURE §3.1 / D6)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sched/mode_controller.h"

namespace llmoc::sched {

struct ExpertHint {
  int layer = 0;
  int expert = 0;
  double freq = 0.0;  // higher = hotter
  uint64_t bytes = 0;
};

struct PlacementPlan {
  ExecMode mode = ExecMode::kPureCpu;
  uint64_t vram_budget = 0;
  uint64_t dram_budget = 0;
  uint64_t vram_attn_reserve = 0;   // reserved for attn/KV/MTP (hybrid)
  uint64_t vram_expert_budget = 0;  // remainder for expert L1
  std::vector<ExpertHint> vram_experts;
  std::vector<ExpertHint> dram_experts;
  std::string summary;
};

class PlacementPlanner {
 public:
  struct Config {
    uint64_t vram_bytes = 0;
    uint64_t dram_bytes = 0;
    // hybrid: fraction of VRAM reserved for attn/KV before experts (0.6 default)
    double attn_reserve_frac = 0.6;
  };

  // Greedy bagging by freq into VRAM then DRAM. Pure-gpu: all experts prefer VRAM until full.
  static PlacementPlan solve(ExecMode mode, const Config& cfg,
                             const std::vector<ExpertHint>& experts);
};

}  // namespace llmoc::sched
