// llm-on-cpu :: glm/hal/glm_awq_int4_ops.cpp
#include "glm/hal/glm_awq_int4_ops.h"

#include <cmath>
#include <cstring>

namespace llmoc::glm::hal {
namespace {

float f16_to_f32(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) out = sign << 31;
    else {
      float f = std::ldexp(static_cast<float>(man), -24);
      return sign ? -f : f;
    }
  } else if (exp == 31) {
    out = (sign << 31) | 0x7F800000u | (man << 13);
  } else {
    out = (sign << 31) | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  static_assert(sizeof(f) == 4, "");
  std::memcpy(&f, &out, 4);
  return f;
}

}  // namespace

void gemm_awq_int4(const float* x, const AwqView& W, float* y) {
  int gs = W.group_size > 0 ? W.group_size : W.K;
  if (gs > W.K || W.K % gs != 0) gs = W.K;
  const int ng = W.K / gs;
  const int rb = (W.K + 1) / 2;
  for (int m = 0; m < W.M; ++m) {
    const uint8_t* row = W.qweight + static_cast<size_t>(m) * rb;
    float acc = 0.f;
    for (int k = 0; k < W.K; ++k) {
      const int g = k / gs;
      float scale = 0.01f;
      if (W.scales_f32) {
        scale = W.scales_f32[static_cast<size_t>(m) * ng + g];
      } else if (W.scales) {
        scale = f16_to_f32(W.scales[static_cast<size_t>(m) * ng + g]);
      }
      const uint8_t b = row[k / 2];
      const int qi = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
      acc += x[k] * (static_cast<float>(qi - 7) * scale);
    }
    y[m] = std::isfinite(acc) ? acc : 0.f;
  }
}

}  // namespace llmoc::glm::hal
