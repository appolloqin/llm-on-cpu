// llm-on-cpu :: exec/factory.cpp
#include "exec/factory.h"

#include "exec/cpu/exec_cpu.h"
#include "exec/gpu/exec_gpu.h"
#include "exec/hybrid/exec_hybrid.h"

namespace llmoc::exec {

std::unique_ptr<contracts::IExecBackend> make_exec(contracts::ExecMode mode,
                                                   const MakeExecOptions& opt,
                                                   std::string* err) {
  using contracts::ExecMode;
  if (mode == ExecMode::kAuto) {
    if (err) *err = "make_exec: pass resolved mode (not auto)";
    return nullptr;
  }
  if (mode == ExecMode::kPureCpu) {
    return std::make_unique<cpu::ExecCpu>();
  }
  if (mode == ExecMode::kLayerStream) {
    // S0: backend 与 pure_cpu 相同占位；真正层窗口在 WeightSource（S1），forward（S2）
    return std::make_unique<cpu::ExecCpu>();
  }
  if (mode == ExecMode::kHybridGpu) {
    if (opt.mesh.world_size > 1 && opt.mesh.nccl_ok == false && opt.mesh.ids.size() > 1) {
      // Mesh resolver should have failed already; belt-and-suspenders.
      if (err) *err = "hybrid mesh requires NCCL when world_size>1";
      return nullptr;
    }
    return std::make_unique<hybrid::ExecHybrid>(opt.mesh);
  }
  // pure_gpu
  if (opt.mesh.world_size > 1 && !opt.mesh.nccl_ok) {
    if (err) *err = "pure_gpu mesh requires NCCL when world_size>1";
    return nullptr;
  }
  return std::make_unique<gpu::ExecGpu>(opt.mesh, opt.n_experts_hint);
}

}  // namespace llmoc::exec
