// tests/unit/test_exec_mesh.cpp — contracts mesh + active-set planner + make_exec
#include "test_main.h"

#include <memory>
#include <string>

#include "contracts/device_mesh.h"
#include "contracts/family_pack.h"
#include "exec/factory.h"
#include "exec/gpu/residency_gpu.h"
#include "families/family_packs.h"
#include "hal/cuda_backend.h"
#include "sched/mode_controller.h"
#include "sched/placement_planner.h"

TINY_TEST(ExecMesh, ResolveSingleDevice) {
  llmoc::contracts::DeviceMeshSpec spec;
  spec.ids = {0};
  spec.strategy = llmoc::contracts::MeshStrategy::kAuto;
  spec.require_nccl = true;
  llmoc::contracts::DeviceMesh mesh;
  std::string err;
  EXPECT_TRUE(llmoc::contracts::resolve_device_mesh(spec, 1, false, true, &mesh, &err));
  EXPECT_TRUE(mesh.world_size == 1);
  EXPECT_TRUE(mesh.ep_size == 1);
}

TINY_TEST(ExecMesh, MultiGpuRequiresNccl) {
  llmoc::contracts::DeviceMeshSpec spec;
  spec.ids = {0, 1};
  spec.strategy = llmoc::contracts::MeshStrategy::kEp;
  spec.require_nccl = true;
  llmoc::contracts::DeviceMesh mesh;
  std::string err;
  EXPECT_TRUE(!llmoc::contracts::resolve_device_mesh(spec, 2, false, true, &mesh, &err));
  EXPECT_TRUE(!err.empty());
}

TINY_TEST(ExecMesh, MultiGpuOkWhenNcclPresentOrWaived) {
  llmoc::contracts::DeviceMeshSpec spec;
  spec.ids = {0, 1};
  spec.strategy = llmoc::contracts::MeshStrategy::kEp;
  spec.require_nccl = false;
  llmoc::contracts::DeviceMesh mesh;
  std::string err;
  EXPECT_TRUE(llmoc::contracts::resolve_device_mesh(spec, 2, false, true, &mesh, &err));
  EXPECT_TRUE(mesh.world_size == 2);
  EXPECT_TRUE(mesh.ep_size == 2);
  EXPECT_TRUE(mesh.tp_size == 1);
  EXPECT_TRUE(mesh.owner_rank(0, 8) == 0);
  EXPECT_TRUE(mesh.owner_rank(1, 8) == 1);
}

TINY_TEST(ExecMesh, ActiveSetPureGpuFitAndFail) {
  llmoc::contracts::DeviceMesh mesh;
  mesh.ids = {0};
  mesh.world_size = 1;
  mesh.ep_size = 1;
  mesh.tp_size = 1;

  llmoc::contracts::ActiveSetProfile ap;
  ap.has_moe = true;
  ap.n_experts = 64;
  ap.top_k = 8;
  ap.expert_bytes = 32ull << 20;  // 32 MiB each
  ap.per_layer_attn_bytes = 64ull << 20;
  ap.kv_working_bytes = 16ull << 20;
  ap.workspace_bytes = 8ull << 20;

  llmoc::sched::PlacementPlanner::Config cfg;
  cfg.vram_bytes = 8ull << 30;
  cfg.dram_bytes = 16ull << 30;
  cfg.strict_vram = true;
  cfg.margin_bytes = 512ull << 20;
  cfg.expert_slot_extra = 1;

  auto ok = llmoc::sched::PlacementPlanner::solve_active(llmoc::sched::ExecMode::kPureGpu, cfg, mesh,
                                                        ap);
  EXPECT_TRUE(ok.ok);
  EXPECT_TRUE(ok.expert_slots_per_rank > 0);

  cfg.vram_bytes = 100ull << 20;  // too small
  auto bad = llmoc::sched::PlacementPlanner::solve_active(llmoc::sched::ExecMode::kPureGpu, cfg,
                                                         mesh, ap);
  EXPECT_TRUE(!bad.ok);
  EXPECT_TRUE(bad.error.find("active working set") != std::string::npos);
}

TINY_TEST(ExecMesh, MakeExecCpuAndGpuCaps) {
  llmoc::exec::MakeExecOptions opt;
  opt.mesh.ids = {0};
  opt.mesh.world_size = 1;
  opt.mesh.ep_size = 1;
  opt.mesh.tp_size = 1;

  auto cpu = llmoc::exec::make_exec(llmoc::contracts::ExecMode::kPureCpu, opt);
  EXPECT_TRUE(cpu != nullptr);
  EXPECT_TRUE(!cpu->caps().experts_on_gpu);
  EXPECT_TRUE(!cpu->caps().attn_on_gpu);

  opt.mesh.nccl_ok = true;
  auto gpu = llmoc::exec::make_exec(llmoc::contracts::ExecMode::kPureGpu, opt);
  EXPECT_TRUE(gpu != nullptr);
  EXPECT_TRUE(gpu->caps().experts_on_gpu);
  EXPECT_TRUE(gpu->caps().attn_on_gpu);

  auto hy = llmoc::exec::make_exec(llmoc::contracts::ExecMode::kHybridGpu, opt);
  EXPECT_TRUE(hy != nullptr);
  EXPECT_TRUE(!hy->caps().experts_on_gpu);
  EXPECT_TRUE(hy->caps().attn_on_gpu);
}

TINY_TEST(ExecMesh, GpuExpertOwnsEp) {
  llmoc::contracts::DeviceMesh mesh;
  mesh.ids = {0, 1};
  mesh.world_size = 2;
  mesh.ep_size = 2;
  mesh.tp_size = 1;
  mesh.nccl_ok = true;
  mesh.strategy = llmoc::contracts::MeshStrategy::kEp;

  llmoc::exec::MakeExecOptions opt;
  opt.mesh = mesh;
  opt.n_experts_hint = 8;
  auto gpu = llmoc::exec::make_exec(llmoc::contracts::ExecMode::kPureGpu, opt);
  EXPECT_TRUE(gpu != nullptr);
  EXPECT_TRUE(gpu->experts()->owns({0, 0}));
  EXPECT_TRUE(!gpu->experts()->owns({0, 1}));  // rank0 default, expert 1 → rank1
}

TINY_TEST(ExecMesh, KimiSingleCardPureGpuRejected) {
  llmoc::families::KimiK3Pack pack;
  llmoc::contracts::DeviceMesh mesh;
  mesh.ids = {0};
  mesh.world_size = mesh.ep_size = mesh.tp_size = 1;
  llmoc::sched::PlacementPlanner::Config cfg;
  cfg.vram_bytes = 24ull << 30;
  cfg.dram_bytes = 64ull << 30;
  cfg.strict_vram = true;
  cfg.margin_bytes = 512ull << 20;
  auto plan = llmoc::sched::PlacementPlanner::solve_active(llmoc::sched::ExecMode::kPureGpu, cfg,
                                                           mesh, pack.active_profile());
  EXPECT_TRUE(!plan.ok);
}

TINY_TEST(ExecMesh, ResidencyPoolUploadWithoutCudaIsNull) {
  llmoc::exec::gpu::ExpertSlotPool pool;
  pool.configure(4, 1ull << 30);
  std::vector<char> host(1024, 1);
  auto* p = pool.pin_upload({0, 0}, host.data(), host.size());
  EXPECT_TRUE(p == nullptr || llmoc::hal::cuda::enabled());
}
