// llm-on-cpu :: tests/unit/test_int4_gemm.cpp
#include <cmath>
#include <cstdint>
#include <vector>

#include "hal/int4_ops.h"
#include "test_main.h"

using namespace llmoc;

static void pack_row(const std::vector<uint8_t>& q /*0..15*/, std::vector<uint8_t>& out, int M,
                     int K) {
  out.assign(static_cast<size_t>(M) * ((K + 1) / 2), 0);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; k += 2) {
      const uint8_t lo = q[static_cast<size_t>(m) * K + k];
      const uint8_t hi = (k + 1 < K) ? q[static_cast<size_t>(m) * K + k + 1] : 0;
      out[static_cast<size_t>(m) * ((K + 1) / 2) + k / 2] =
          static_cast<uint8_t>(lo | (hi << 4));
    }
  }
}

static uint16_t f32_to_f16bits(float x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.f = x;
  uint32_t sign = (v.u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((v.u >> 23) & 0xFF) - 127 + 15;
  uint32_t man = v.u & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

TINY_TEST(Int4, GemmAwqMatchesScalarRef) {
  const int M = 17, K = 128, gs = 128;
  std::vector<uint8_t> q(static_cast<size_t>(M) * K);
  for (int i = 0; i < M * K; ++i) q[i] = static_cast<uint8_t>((i * 3) % 16);
  std::vector<uint8_t> packed;
  pack_row(q, packed, M, K);
  std::vector<uint16_t> scales(M);
  for (int m = 0; m < M; ++m) scales[m] = f32_to_f16bits(0.01f + 0.001f * static_cast<float>(m));
  std::vector<float> x(K), y(M), y_ref(M);
  for (int k = 0; k < K; ++k) x[k] = 0.02f * static_cast<float>((k % 7) - 3);

  qlwc::Int4View W;
  W.qweight = packed.data();
  W.scales = scales.data();
  W.zeros = nullptr;
  W.M = M;
  W.K = K;
  W.group_size = gs;
  W.scheme = qlwc::Scheme::kAwqSym;

  for (int m = 0; m < M; ++m) {
    float acc = 0.f;
    const float sc = llmoc::hal::f16_to_f32(scales[m]);
    for (int k = 0; k < K; ++k) {
      const float w =
          static_cast<float>(static_cast<int>(q[static_cast<size_t>(m) * K + k]) - 7) * sc;
      acc += x[k] * w;
    }
    y_ref[m] = acc;
  }

  llmoc::hal::gemm_int4(x.data(), W, y.data());
  for (int m = 0; m < M; ++m) {
    const float e = std::fabs(y[m] - y_ref[m]);
    EXPECT_TRUE(e < 1e-3f * (1.f + std::fabs(y_ref[m])));
  }
}

TINY_TEST(Int4, GemmBatchMatchesSingle) {
  const int M = 64, K = 128, gs = 128, n = 7;
  std::vector<uint8_t> q(static_cast<size_t>(M) * K);
  for (int i = 0; i < M * K; ++i) q[i] = static_cast<uint8_t>((i * 5 + 1) % 16);
  std::vector<uint8_t> packed;
  pack_row(q, packed, M, K);
  std::vector<uint16_t> scales(M);
  std::vector<float> scales_f32(M);
  for (int m = 0; m < M; ++m) {
    scales[m] = f32_to_f16bits(0.02f);
    scales_f32[m] = 0.02f;
  }
  std::vector<float> X(static_cast<size_t>(n) * K), Yb(static_cast<size_t>(n) * M),
      Ys(static_cast<size_t>(n) * M);
  for (int i = 0; i < n * K; ++i) X[i] = 0.01f * static_cast<float>((i % 11) - 5);

  qlwc::Int4View W;
  W.qweight = packed.data();
  W.scales = scales.data();
  W.scales_f32 = scales_f32.data();
  W.zeros = nullptr;
  W.M = M;
  W.K = K;
  W.group_size = gs;
  W.scheme = qlwc::Scheme::kAwqSym;

  llmoc::hal::gemm_int4_batch(X.data(), n, W, Yb.data());
  for (int t = 0; t < n; ++t)
    llmoc::hal::gemm_int4(X.data() + t * K, W, Ys.data() + t * M);
  for (int i = 0; i < n * M; ++i) {
    const float e = std::fabs(Yb[i] - Ys[i]);
    EXPECT_TRUE(e < 1e-4f * (1.f + std::fabs(Ys[i])));
  }
}
