// llm-on-cpu :: sched/mode_controller.cpp
#include "sched/mode_controller.h"

namespace llmoc::sched {

ExecMode resolve_mode(ExecMode requested, bool cuda_ok, bool* degraded, std::string* err) {
  if (degraded) *degraded = false;
  if (requested == ExecMode::kAuto) {
    return cuda_ok ? ExecMode::kHybridGpu : ExecMode::kPureCpu;
  }
  if (requested == ExecMode::kPureCpu) return ExecMode::kPureCpu;
  if (requested == ExecMode::kPureGpu) {
    if (!cuda_ok) {
      if (err) *err = "mode=pure_gpu requires CUDA (cudart+cublas); refusing silent CPU fallback";
      return ExecMode::kPureGpu;
    }
    return ExecMode::kPureGpu;
  }
  if (!cuda_ok) {
    if (degraded) *degraded = true;
    return ExecMode::kPureCpu;
  }
  return ExecMode::kHybridGpu;
}

bool resolve_mesh_for_mode(ExecMode mode, const contracts::DeviceMeshSpec& spec, int visible_gpus,
                           bool nccl_available, bool moe_family, contracts::DeviceMesh* out,
                           std::string* err) {
  if (mode == ExecMode::kPureCpu) {
    if (!out) return false;
    out->ids = {0};
    out->world_size = 1;
    out->ep_size = 1;
    out->tp_size = 1;
    out->strategy = contracts::MeshStrategy::kTp;
    out->nccl_ok = false;
    return true;
  }
  contracts::DeviceMeshSpec s = spec;
  if (mode == ExecMode::kHybridGpu && s.strategy == contracts::MeshStrategy::kAuto &&
      s.ids.size() > 1) {
    // Hybrid multi-GPU: TP attn only (doc §2.3).
    s.strategy = contracts::MeshStrategy::kTp;
    s.has_moe = false;
  }
  return contracts::resolve_device_mesh(s, visible_gpus, nccl_available, moe_family, out, err);
}

}  // namespace llmoc::sched
