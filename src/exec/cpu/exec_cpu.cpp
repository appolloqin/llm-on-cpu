// llm-on-cpu :: exec/cpu/exec_cpu.cpp
#include "exec/cpu/exec_cpu.h"

#include "hal/cpu_ops.h"

namespace llmoc::exec::cpu {

bool GemmCpu::gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) {
  if (!x || !W || !y || M <= 0 || K <= 0) return false;
  // CPU path: use existing free-function GEMM (BF16/F16 weights).
  const auto dt = is_f16 ? llmoc::hal::WDtype::kF16 : llmoc::hal::WDtype::kBF16;
  llmoc::hal::gemm_bias_free(x, reinterpret_cast<const uint16_t*>(W), y, M, K, dt);
  return true;
}

void ExpertRuntimeCpu::pin(int layer, const contracts::ExpertId* ids, int n,
                           contracts::BlockHandle* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = {};
    out[i].id = ids ? ids[i] : contracts::ExpertId{layer, i};
    out[i].device_ordinal = -1;
  }
}

ExecCpu::ExecCpu() {
  mesh_.ids = {0};
  mesh_.world_size = 1;
  mesh_.ep_size = 1;
  mesh_.tp_size = 1;
  mesh_.strategy = contracts::MeshStrategy::kTp;
}

contracts::ExecCaps ExecCpu::caps() const {
  contracts::ExecCaps c;
  c.experts_on_gpu = false;
  c.attn_on_gpu = false;
  c.world_size = 1;
  c.ep_size = 1;
  c.tp_size = 1;
  c.vram_budget_per_rank = 0;
  c.dram_budget = dram_budget_;
  return c;
}

void ExecCpu::configure(const sched::PlacementPlan& plan) {
  plan_ = plan;
  dram_budget_ = plan.dram_budget;
}

}  // namespace llmoc::exec::cpu
