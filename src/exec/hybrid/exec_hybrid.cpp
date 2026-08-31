// llm-on-cpu :: exec/hybrid/exec_hybrid.cpp
#include "exec/hybrid/exec_hybrid.h"

#include "exec/mesh_nccl.h"
#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"

namespace llmoc::exec::hybrid {

bool GemmHybrid::gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) {
  if (llmoc::hal::cuda::enabled() &&
      llmoc::hal::cuda::try_gemm_w16(x, reinterpret_cast<const uint16_t*>(W), y, M, K, is_f16)) {
    return true;
  }
  const auto dt = is_f16 ? llmoc::hal::WDtype::kF16 : llmoc::hal::WDtype::kBF16;
  llmoc::hal::gemm_bias_free(x, reinterpret_cast<const uint16_t*>(W), y, M, K, dt);
  return true;
}

void ExpertRuntimeCpuBridge::pin(int layer, const contracts::ExpertId* ids, int n,
                                 contracts::BlockHandle* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = {};
    out[i].id = ids ? ids[i] : contracts::ExpertId{layer, i};
    out[i].device_ordinal = -1;
  }
}

ExecHybrid::ExecHybrid(contracts::DeviceMesh mesh)
    : mesh_(std::move(mesh)), coll_(make_collective(mesh_)) {}

contracts::ExecCaps ExecHybrid::caps() const {
  contracts::ExecCaps c;
  c.experts_on_gpu = false;
  c.attn_on_gpu = true;
  c.world_size = mesh_.world_size;
  c.ep_size = mesh_.ep_size;
  c.tp_size = mesh_.tp_size;
  c.vram_budget_per_rank = vram_budget_;
  c.dram_budget = dram_budget_;
  return c;
}

void ExecHybrid::configure(const sched::PlacementPlan& plan) {
  plan_ = plan;
  vram_budget_ = plan.vram_budget;
  dram_budget_ = plan.dram_budget;
}

}  // namespace llmoc::exec::hybrid
