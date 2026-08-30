// llm-on-cpu :: glm/hal/glm_nvfp4_ops.cpp
#include "glm/hal/glm_nvfp4_ops.h"

#include <cmath>

namespace llmoc::glm::hal {
namespace {

// E2M1 decode (NVIDIA NVFP4 nibble) → float approx
float e2m1_to_f32(unsigned nibble) {
  static const float kLut[16] = {
      0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
  };
  return kLut[nibble & 15u];
}

float fp8_e4m3_to_f32(uint8_t u) {
  const int sign = (u >> 7) & 1;
  const int exp = (u >> 3) & 0xF;
  const int man = u & 0x7;
  if (exp == 0) {
    float v = static_cast<float>(man) / 8.f * std::ldexp(1.f, -6);
    return sign ? -v : v;
  }
  if (exp == 15 && man == 7) return sign ? -0.f : 0.f;  // treat NaN-ish as 0 for safety
  float v = (1.f + static_cast<float>(man) / 8.f) * std::ldexp(1.f, exp - 7);
  return sign ? -v : v;
}

}  // namespace

void gemm_nvfp4(const float* x, const Nvfp4View& W, float* y) {
  const int gs = W.group_size > 0 ? W.group_size : 16;
  const int rb = (W.K + 1) / 2;
  const int ng = (W.K + gs - 1) / gs;
  for (int m = 0; m < W.M; ++m) {
    const uint8_t* row = W.qweight + static_cast<size_t>(m) * rb;
    float acc = 0.f;
    for (int k = 0; k < W.K; ++k) {
      const uint8_t b = row[k / 2];
      const unsigned nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
      float s = W.global_scale;
      if (W.scales_fp8) {
        const int g = k / gs;
        s *= fp8_e4m3_to_f32(W.scales_fp8[static_cast<size_t>(m) * ng + g]);
      }
      acc += x[k] * (e2m1_to_f32(nib) * s);
    }
    y[m] = std::isfinite(acc) ? acc : 0.f;
  }
}

}  // namespace llmoc::glm::hal
