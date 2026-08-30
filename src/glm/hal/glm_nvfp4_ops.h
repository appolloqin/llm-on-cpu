#pragma once
// llm-on-cpu :: glm/hal/glm_nvfp4_ops.h — NVFP4（E2M1）CPU 参考核

#include <cstdint>

namespace llmoc::glm::hal {

struct Nvfp4View {
  const uint8_t* qweight = nullptr;  // packed FP4
  const uint8_t* scales_fp8 = nullptr;  // E4M3 per group
  float global_scale = 1.f;
  int M = 0;
  int K = 0;
  int group_size = 16;  // NVFP4 g16
};

void gemm_nvfp4(const float* x, const Nvfp4View& W, float* y);

}  // namespace llmoc::glm::hal
