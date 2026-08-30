// tests/unit/test_m5_mode.cpp — M5 mode + PlacementPlanner (CPU-safe)
#include "test_main.h"

#include <string>
#include <vector>

#include "hal/cuda_backend.h"
#include "sched/mode_controller.h"
#include "sched/placement_planner.h"

TINY_TEST(M5, ResolvePureCpuUnchanged) {
  bool deg = false;
  auto m = llmoc::sched::resolve_mode(llmoc::sched::ExecMode::kPureCpu, true, &deg);
  EXPECT_TRUE(m == llmoc::sched::ExecMode::kPureCpu);
  EXPECT_TRUE(!deg);
  m = llmoc::sched::resolve_mode(llmoc::sched::ExecMode::kPureCpu, false, &deg);
  EXPECT_TRUE(m == llmoc::sched::ExecMode::kPureCpu);
}

TINY_TEST(M5, AutoPicksHybridIfCuda) {
  bool deg = false;
  auto m = llmoc::sched::resolve_mode(llmoc::sched::ExecMode::kAuto, true, &deg);
  EXPECT_TRUE(m == llmoc::sched::ExecMode::kHybridGpu);
  m = llmoc::sched::resolve_mode(llmoc::sched::ExecMode::kAuto, false, &deg);
  EXPECT_TRUE(m == llmoc::sched::ExecMode::kPureCpu);
}

TINY_TEST(M5, HybridDegradesWithoutCuda) {
  bool deg = false;
  auto m = llmoc::sched::resolve_mode(llmoc::sched::ExecMode::kHybridGpu, false, &deg);
  EXPECT_TRUE(m == llmoc::sched::ExecMode::kPureCpu);
  EXPECT_TRUE(deg);
}

TINY_TEST(M5, PlannerPureCpuAllDram) {
  llmoc::sched::PlacementPlanner::Config cfg;
  cfg.vram_bytes = 8ull << 30;
  cfg.dram_bytes = 16ull << 30;
  std::vector<llmoc::sched::ExpertHint> ex = {{0, 0, 1.0, 1ull << 20}, {0, 1, 0.5, 1ull << 20}};
  auto p = llmoc::sched::PlacementPlanner::solve(llmoc::sched::ExecMode::kPureCpu, cfg, ex);
  EXPECT_TRUE(p.vram_experts.empty());
  EXPECT_TRUE(p.dram_experts.size() == 2);
}

TINY_TEST(M5, PlannerHybridBagsHotExperts) {
  llmoc::sched::PlacementPlanner::Config cfg;
  cfg.vram_bytes = 100ull << 20;  // 100 MiB
  cfg.dram_bytes = 200ull << 20;
  cfg.attn_reserve_frac = 0.5;  // 50 MiB experts
  std::vector<llmoc::sched::ExpertHint> ex = {
      {0, 0, 10.0, 30ull << 20},
      {0, 1, 5.0, 30ull << 20},
      {0, 2, 1.0, 30ull << 20},
  };
  auto p = llmoc::sched::PlacementPlanner::solve(llmoc::sched::ExecMode::kHybridGpu, cfg, ex);
  EXPECT_TRUE(p.vram_experts.size() >= 1);
  EXPECT_TRUE(!p.summary.empty());
}

TINY_TEST(M5, CudaProbeDoesNotEnableGemm) {
  // Probe must not turn on gemm path (pure_cpu safety).
  (void)llmoc::hal::cuda::probe_available();
  EXPECT_TRUE(!llmoc::hal::cuda::enabled());
}
