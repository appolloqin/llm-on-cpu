#pragma once
// llm-on-cpu :: glm/hal/glm_awq_int4_ops.h — GLM 自有 AWQ 核（不依赖 hal/int4_ops）

#include <cstdint>

namespace llmoc::glm::hal {

struct AwqView {
  const uint8_t* qweight = nullptr;  // packed nibbles, row-major
  const uint16_t* scales = nullptr;  // f16
  const float* scales_f32 = nullptr;
  int M = 0;
  int K = 0;
  int group_size = 128;
};

// y[M] = W[M,K] @ x[K]  (AWQ symmetric q-7)
void gemm_awq_int4(const float* x, const AwqView& W, float* y);

}  // namespace llmoc::glm::hal
