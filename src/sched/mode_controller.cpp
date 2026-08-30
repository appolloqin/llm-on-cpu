// llm-on-cpu :: sched/mode_controller.cpp
#include "sched/mode_controller.h"

#include <string>

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
      // Still return kPureGpu; caller must abort on err
      return ExecMode::kPureGpu;
    }
    return ExecMode::kPureGpu;
  }
  // hybrid_gpu
  if (!cuda_ok) {
    if (degraded) *degraded = true;
    return ExecMode::kPureCpu;
  }
  return ExecMode::kHybridGpu;
}

}  // namespace llmoc::sched
