// llm-on-cpu :: hal/quant_dequant.cpp
#include "hal/quant_views.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llmoc::hal {
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
  std::memcpy(&f, &out, 4);
  return f;
}

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
  if (exp == 15 && man == 7) return sign ? -0.f : 0.f;
  float v = (1.f + static_cast<float>(man) / 8.f) * std::ldexp(1.f, exp - 7);
  return sign ? -v : v;
}

}  // namespace

void dequant_awq_matrix(const AwqView& W, float* out) {
  if (!out || !W.qweight || W.M <= 0 || W.K <= 0) throw std::runtime_error("dequant_awq bad args");
  int gs = W.group_size > 0 ? W.group_size : W.K;
  if (gs > W.K || W.K % gs != 0) gs = W.K;
  const int ng = W.K / gs;
  const int rb = (W.K + 1) / 2;
  for (int m = 0; m < W.M; ++m) {
    const uint8_t* row = W.qweight + static_cast<size_t>(m) * rb;
    float* dst = out + static_cast<size_t>(m) * W.K;
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
      dst[k] = static_cast<float>(qi - 7) * scale;
    }
  }
}

void dequant_nvfp4_matrix(const Nvfp4View& W, float* out) {
  if (!out || !W.qweight || W.M <= 0 || W.K <= 0) throw std::runtime_error("dequant_nvfp4 bad args");
  const int gs = W.group_size > 0 ? W.group_size : 16;
  const int rb = (W.K + 1) / 2;
  const int ng = (W.K + gs - 1) / gs;
  for (int m = 0; m < W.M; ++m) {
    const uint8_t* row = W.qweight + static_cast<size_t>(m) * rb;
    float* dst = out + static_cast<size_t>(m) * W.K;
    for (int k = 0; k < W.K; ++k) {
      const uint8_t b = row[k / 2];
      const unsigned nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
      float s = W.global_scale;
      if (W.scales_fp8) {
        const int g = k / gs;
        s *= fp8_e4m3_to_f32(W.scales_fp8[static_cast<size_t>(m) * ng + g]);
      }
      dst[k] = e2m1_to_f32(nib) * s;
    }
  }
}

}  // namespace llmoc::hal
