#pragma once
// llm-on-cpu :: glm/hal/glm_nvfp4_ops.h — NVFP4（共享 hal::Nvfp4View）

#include "hal/quant_views.h"

namespace llmoc::glm::hal {

using Nvfp4View = llmoc::hal::Nvfp4View;

void gemm_nvfp4(const float* x, const Nvfp4View& W, float* y);

}  // namespace llmoc::glm::hal
