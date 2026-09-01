#pragma once
// llm-on-cpu :: glm/hal/glm_awq_int4_ops.h — GLM AWQ（共享 hal::AwqView）

#include "hal/quant_views.h"

namespace llmoc::glm::hal {

using AwqView = llmoc::hal::AwqView;

// y[M] = W[M,K] @ x[K]  (AWQ symmetric q-7)
void gemm_awq_int4(const float* x, const AwqView& W, float* y);

}  // namespace llmoc::glm::hal
