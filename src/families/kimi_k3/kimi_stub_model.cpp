// llm-on-cpu :: families/kimi_k3/kimi_stub_model.cpp
#include "families/kimi_k3/kimi_stub_model.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "families/family_packs.h"
#include "sched/placement_planner.h"

namespace llmoc::families::kimi {

void write_fake_kimiq(const std::string& path, KimiStubGeometry g) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("write_fake_kimiq failed");
  out.write("KIM1", 4);
  uint32_t ver = 1;
  out.write(reinterpret_cast<const char*>(&ver), 4);
  out.write(reinterpret_cast<const char*>(&g.hidden), 4);
  out.write(reinterpret_cast<const char*>(&g.layers), 4);
  out.write(reinterpret_cast<const char*>(&g.vocab), 4);
  out.write(reinterpret_cast<const char*>(&g.n_experts), 4);
  out.write(reinterpret_cast<const char*>(&g.top_k), 4);
  out.write(reinterpret_cast<const char*>(&g.d_latent), 4);
  out.write(reinterpret_cast<const char*>(&g.intermediate), 4);
}

bool pure_gpu_single_card_does_not_fit(contracts::ExecMode mode, int world_size) {
  if (mode != contracts::ExecMode::kPureGpu) return false;
  if (world_size > 1) return false;
  KimiK3Pack pack;
  sched::PlacementPlanner::Config pcfg;
  pcfg.vram_bytes = 24ull << 30;
  pcfg.dram_bytes = 64ull << 30;
  contracts::DeviceMesh mesh;
  mesh.ids = {0};
  mesh.world_size = mesh.ep_size = mesh.tp_size = 1;
  auto plan = sched::PlacementPlanner::solve_active(mode, pcfg, mesh, pack.active_profile());
  return !plan.ok;
}

contracts::ExecMode resolve_kimi_exec_mode(contracts::ExecMode requested, int world_size,
                                           bool* degraded) {
  if (degraded) *degraded = false;
  if (!pure_gpu_single_card_does_not_fit(requested, world_size)) return requested;
  if (degraded) *degraded = true;
  return contracts::ExecMode::kLayerStream;
}

}  // namespace llmoc::families::kimi
