#pragma once
// llm-on-cpu :: hal/quant_views.h — shared AWQ / NVFP4 views (CPU + GPU dequant)

#include <cstdint>

namespace llmoc::hal {

struct AwqView {
  const uint8_t* qweight = nullptr;
  const uint16_t* scales = nullptr;
  const float* scales_f32 = nullptr;
  int M = 0;
  int K = 0;
  int group_size = 128;
};

struct Nvfp4View {
  const uint8_t* qweight = nullptr;
  const uint8_t* scales_fp8 = nullptr;
  float global_scale = 1.f;
  int M = 0;
  int K = 0;
  int group_size = 16;
};

// Host dequant to FP32 row-major M×K (for GPU cache / debug).
void dequant_awq_matrix(const AwqView& W, float* out);
void dequant_nvfp4_matrix(const Nvfp4View& W, float* out);

}  // namespace llmoc::hal
