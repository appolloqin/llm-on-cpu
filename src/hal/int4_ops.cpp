// llm-on-cpu :: src/hal/int4_ops.cpp
// INT4 GEMM：标量参考 + AVX2/FMA（多行复用 x、scale 外提、预取）

#include "hal/int4_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(LLMOC_ENABLE_AVX2)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace llmoc::hal {
namespace {

inline float f16b_to_f32(uint16_t h) { return f16_to_f32(h); }

inline int row_bytes(int K) { return ((K + 1) / 2); }

#if defined(LLMOC_ENABLE_AVX2)
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(lo);
  lo = _mm_add_ps(lo, sh);
  sh = _mm_movehl_ps(sh, lo);
  lo = _mm_add_ss(lo, sh);
  return _mm_cvtss_f32(lo);
}

// 16B packed nibble → 4×8 int32（byte0.lo, byte0.hi, ...）
inline void unpack32_nibbles(const uint8_t* qrow_at_k_div2, __m256i& i0, __m256i& i1, __m256i& i2,
                             __m256i& i3) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qrow_at_k_div2));
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(raw, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
  const __m128i p0 = _mm_unpacklo_epi8(lo, hi);
  const __m128i p1 = _mm_unpackhi_epi8(lo, hi);
  i0 = _mm256_cvtepu8_epi32(p0);
  i1 = _mm256_cvtepu8_epi32(_mm_srli_si128(p0, 8));
  i2 = _mm256_cvtepu8_epi32(p1);
  i3 = _mm256_cvtepu8_epi32(_mm_srli_si128(p1, 8));
}

// AWQ: 累加 x*(q-7)，scale 在组外乘（减内层 mul）
inline float dot_awq_noscale(const float* x, const uint8_t* qrow, int k0, int k1) {
  __m256 vacc0 = _mm256_setzero_ps();
  __m256 vacc1 = _mm256_setzero_ps();
  __m256 vacc2 = _mm256_setzero_ps();
  __m256 vacc3 = _mm256_setzero_ps();
  const __m256 v7 = _mm256_set1_ps(7.f);
  int k = k0;
  for (; k + 32 <= k1; k += 32) {
    __m256i i0, i1, i2, i3;
    unpack32_nibbles(qrow + (k / 2), i0, i1, i2, i3);
    vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k),
                            _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), vacc0);
    vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8),
                            _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), vacc1);
    vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16),
                            _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), vacc2);
    vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24),
                            _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), vacc3);
  }
  float acc = hsum256(_mm256_add_ps(_mm256_add_ps(vacc0, vacc1), _mm256_add_ps(vacc2, vacc3)));
  for (; k < k1; ++k) {
    const uint8_t b = qrow[k / 2];
    const int qi = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    acc += x[k] * static_cast<float>(qi - 7);
  }
  return acc;
}

inline float dot_gptq_avx2(const float* x, const uint8_t* qrow, int k0, int k1, float scale,
                           float zero) {
  __m256 vacc0 = _mm256_setzero_ps();
  __m256 vacc1 = _mm256_setzero_ps();
  __m256 vacc2 = _mm256_setzero_ps();
  __m256 vacc3 = _mm256_setzero_ps();
  const __m256 vscale = _mm256_set1_ps(scale);
  const __m256 vzero = _mm256_set1_ps(zero);
  int k = k0;
  for (; k + 32 <= k1; k += 32) {
    __m256i i0, i1, i2, i3;
    unpack32_nibbles(qrow + (k / 2), i0, i1, i2, i3);
    __m256 w0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), vscale, vzero);
    __m256 w1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), vscale, vzero);
    __m256 w2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i2), vscale, vzero);
    __m256 w3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i3), vscale, vzero);
    vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), w0, vacc0);
    vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), w1, vacc1);
    vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16), w2, vacc2);
    vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24), w3, vacc3);
  }
  float acc = hsum256(_mm256_add_ps(_mm256_add_ps(vacc0, vacc1), _mm256_add_ps(vacc2, vacc3)));
  for (; k < k1; ++k) {
    const uint8_t b = qrow[k / 2];
    const int qi = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    acc += x[k] * (static_cast<float>(qi) * scale + zero);
  }
  return acc;
}

// 2 行同时：同一段 x 只 load 一次
inline void dot_awq_noscale_2row(const float* x, const uint8_t* q0, const uint8_t* q1, int k0,
                                 int k1, float& out0, float& out1) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
  __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
  const __m256 v7 = _mm256_set1_ps(7.f);
  int k = k0;
  for (; k + 32 <= k1; k += 32) {
    const __m256 x0 = _mm256_loadu_ps(x + k);
    const __m256 x1 = _mm256_loadu_ps(x + k + 8);
    const __m256 x2 = _mm256_loadu_ps(x + k + 16);
    const __m256 x3 = _mm256_loadu_ps(x + k + 24);
    __m256i i0, i1, i2, i3;
    unpack32_nibbles(q0 + (k / 2), i0, i1, i2, i3);
    a0 = _mm256_fmadd_ps(x0, _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), a0);
    b0 = _mm256_fmadd_ps(x1, _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), b0);
    c0 = _mm256_fmadd_ps(x2, _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), c0);
    d0 = _mm256_fmadd_ps(x3, _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), d0);
    unpack32_nibbles(q1 + (k / 2), i0, i1, i2, i3);
    a1 = _mm256_fmadd_ps(x0, _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), a1);
    b1 = _mm256_fmadd_ps(x1, _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), b1);
    c1 = _mm256_fmadd_ps(x2, _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), c1);
    d1 = _mm256_fmadd_ps(x3, _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), d1);
  }
  out0 = hsum256(_mm256_add_ps(_mm256_add_ps(a0, b0), _mm256_add_ps(c0, d0)));
  out1 = hsum256(_mm256_add_ps(_mm256_add_ps(a1, b1), _mm256_add_ps(c1, d1)));
  for (; k < k1; ++k) {
    const uint8_t qb0 = q0[k / 2], qb1 = q1[k / 2];
    const int q0i = (k & 1) ? ((qb0 >> 4) & 0xF) : (qb0 & 0xF);
    const int q1i = (k & 1) ? ((qb1 >> 4) & 0xF) : (qb1 & 0xF);
    out0 += x[k] * static_cast<float>(q0i - 7);
    out1 += x[k] * static_cast<float>(q1i - 7);
  }
}

// 4 行：大 M（MLP / lm_head）复用 x load，提高 L1 命中
inline void dot_awq_noscale_4row(const float* x, const uint8_t* q0, const uint8_t* q1,
                                 const uint8_t* q2, const uint8_t* q3, int k0, int k1, float& out0,
                                 float& out1, float& out2, float& out3) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
         a3 = _mm256_setzero_ps();
  __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps(),
         b3 = _mm256_setzero_ps();
  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(), c2 = _mm256_setzero_ps(),
         c3 = _mm256_setzero_ps();
  __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps(), d2 = _mm256_setzero_ps(),
         d3 = _mm256_setzero_ps();
  const __m256 v7 = _mm256_set1_ps(7.f);
  int k = k0;
  for (; k + 32 <= k1; k += 32) {
    const __m256 x0 = _mm256_loadu_ps(x + k);
    const __m256 x1 = _mm256_loadu_ps(x + k + 8);
    const __m256 x2 = _mm256_loadu_ps(x + k + 16);
    const __m256 x3 = _mm256_loadu_ps(x + k + 24);
    __m256i i0, i1, i2, i3;
    unpack32_nibbles(q0 + (k / 2), i0, i1, i2, i3);
    a0 = _mm256_fmadd_ps(x0, _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), a0);
    b0 = _mm256_fmadd_ps(x1, _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), b0);
    c0 = _mm256_fmadd_ps(x2, _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), c0);
    d0 = _mm256_fmadd_ps(x3, _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), d0);
    unpack32_nibbles(q1 + (k / 2), i0, i1, i2, i3);
    a1 = _mm256_fmadd_ps(x0, _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), a1);
    b1 = _mm256_fmadd_ps(x1, _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), b1);
    c1 = _mm256_fmadd_ps(x2, _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), c1);
    d1 = _mm256_fmadd_ps(x3, _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), d1);
    unpack32_nibbles(q2 + (k / 2), i0, i1, i2, i3);
    a2 = _mm256_fmadd_ps(x0, _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), a2);
    b2 = _mm256_fmadd_ps(x1, _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), b2);
    c2 = _mm256_fmadd_ps(x2, _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), c2);
    d2 = _mm256_fmadd_ps(x3, _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), d2);
    unpack32_nibbles(q3 + (k / 2), i0, i1, i2, i3);
    a3 = _mm256_fmadd_ps(x0, _mm256_sub_ps(_mm256_cvtepi32_ps(i0), v7), a3);
    b3 = _mm256_fmadd_ps(x1, _mm256_sub_ps(_mm256_cvtepi32_ps(i1), v7), b3);
    c3 = _mm256_fmadd_ps(x2, _mm256_sub_ps(_mm256_cvtepi32_ps(i2), v7), c3);
    d3 = _mm256_fmadd_ps(x3, _mm256_sub_ps(_mm256_cvtepi32_ps(i3), v7), d3);
  }
  out0 = hsum256(_mm256_add_ps(_mm256_add_ps(a0, b0), _mm256_add_ps(c0, d0)));
  out1 = hsum256(_mm256_add_ps(_mm256_add_ps(a1, b1), _mm256_add_ps(c1, d1)));
  out2 = hsum256(_mm256_add_ps(_mm256_add_ps(a2, b2), _mm256_add_ps(c2, d2)));
  out3 = hsum256(_mm256_add_ps(_mm256_add_ps(a3, b3), _mm256_add_ps(c3, d3)));
  for (; k < k1; ++k) {
    const uint8_t qb0 = q0[k / 2], qb1 = q1[k / 2], qb2 = q2[k / 2], qb3 = q3[k / 2];
    const int s = (k & 1) ? 4 : 0;
    out0 += x[k] * static_cast<float>(((qb0 >> s) & 0xF) - 7);
    out1 += x[k] * static_cast<float>(((qb1 >> s) & 0xF) - 7);
    out2 += x[k] * static_cast<float>(((qb2 >> s) & 0xF) - 7);
    out3 += x[k] * static_cast<float>(((qb3 >> s) & 0xF) - 7);
  }
}

void gemm_int4_awq_avx2(const float* x, const qlwc::Int4View& W, float* y) {
  const int M = W.M, K = W.K, gs = W.group_size;
  const int ng = (K + gs - 1) / gs;
  const int rb = row_bytes(K);

  // 优先用 QlwcStore 预转的 f32 scales；否则本帧转一次（大 M 极慢）
  const float* scf = W.scales_f32;
  std::vector<float> scales_scratch;
  if (!scf) {
    scales_scratch.resize(static_cast<size_t>(M) * ng);
    for (int i = 0; i < M * ng; ++i)
      scales_scratch[static_cast<size_t>(i)] = f16b_to_f32(W.scales[i]);
    scf = scales_scratch.data();
  }

  auto scale_at = [&](int m, int g) -> float { return scf[static_cast<size_t>(m) * ng + g]; };

  auto finish_1 = [&](int m0) {
    const uint8_t* qrow = W.qweight + static_cast<size_t>(m0) * rb;
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
      const int k0 = g * gs;
      const int k1 = std::min(K, k0 + gs);
      acc += dot_awq_noscale(x, qrow, k0, k1) * scale_at(m0, g);
    }
    y[m0] = std::isfinite(acc) ? acc : 0.f;
  };
  auto finish_2 = [&](int m0) {
    const uint8_t* q0 = W.qweight + static_cast<size_t>(m0) * rb;
    const uint8_t* q1 = W.qweight + static_cast<size_t>(m0 + 1) * rb;
    if (m0 + 2 < M) {
      _mm_prefetch(reinterpret_cast<const char*>(W.qweight + static_cast<size_t>(m0 + 2) * rb),
                   _MM_HINT_T0);
    }
    float acc0 = 0.f, acc1 = 0.f;
    for (int g = 0; g < ng; ++g) {
      const int k0 = g * gs;
      const int k1 = std::min(K, k0 + gs);
      float p0 = 0.f, p1 = 0.f;
      dot_awq_noscale_2row(x, q0, q1, k0, k1, p0, p1);
      acc0 += p0 * scale_at(m0, g);
      acc1 += p1 * scale_at(m0 + 1, g);
    }
    y[m0] = std::isfinite(acc0) ? acc0 : 0.f;
    y[m0 + 1] = std::isfinite(acc1) ? acc1 : 0.f;
  };

  // 大 M：4-row（OpenMP 步长须为常量，分两路）
  if (M >= 4096) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 128)
#endif
    for (int m0 = 0; m0 < M; m0 += 4) {
      if (m0 + 3 < M) {
        const uint8_t* q0 = W.qweight + static_cast<size_t>(m0) * rb;
        const uint8_t* q1 = W.qweight + static_cast<size_t>(m0 + 1) * rb;
        const uint8_t* q2 = W.qweight + static_cast<size_t>(m0 + 2) * rb;
        const uint8_t* q3 = W.qweight + static_cast<size_t>(m0 + 3) * rb;
        if (m0 + 4 < M) {
          _mm_prefetch(reinterpret_cast<const char*>(W.qweight + static_cast<size_t>(m0 + 4) * rb),
                       _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(W.qweight + static_cast<size_t>(m0 + 5) * rb),
                       _MM_HINT_T0);
        }
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        for (int g = 0; g < ng; ++g) {
          const int k0 = g * gs;
          const int k1 = std::min(K, k0 + gs);
          float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
          dot_awq_noscale_4row(x, q0, q1, q2, q3, k0, k1, p0, p1, p2, p3);
          acc0 += p0 * scale_at(m0, g);
          acc1 += p1 * scale_at(m0 + 1, g);
          acc2 += p2 * scale_at(m0 + 2, g);
          acc3 += p3 * scale_at(m0 + 3, g);
        }
        y[m0] = std::isfinite(acc0) ? acc0 : 0.f;
        y[m0 + 1] = std::isfinite(acc1) ? acc1 : 0.f;
        y[m0 + 2] = std::isfinite(acc2) ? acc2 : 0.f;
        y[m0 + 3] = std::isfinite(acc3) ? acc3 : 0.f;
      } else if (m0 + 1 < M) {
        finish_2(m0);
      } else if (m0 < M) {
        finish_1(m0);
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 128)
#endif
    for (int m0 = 0; m0 < M; m0 += 2) {
      if (m0 + 1 < M) finish_2(m0);
      else finish_1(m0);
    }
  }
}

void gemm_int4_gptq_avx2(const float* x, const qlwc::Int4View& W, float* y) {
  const int M = W.M, K = W.K, gs = W.group_size;
  const int ng = (K + gs - 1) / gs;
  const int rb = row_bytes(K);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 128)
#endif
  for (int m = 0; m < M; ++m) {
    const uint8_t* qrow = W.qweight + static_cast<size_t>(m) * rb;
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
      const int k0 = g * gs;
      const int k1 = std::min(K, k0 + gs);
      const float scale = f16b_to_f32(W.scales[m * ng + g]);
      const float zero = W.zeros ? f16b_to_f32(W.zeros[m * ng + g]) : 0.f;
      acc += dot_gptq_avx2(x, qrow, k0, k1, scale, zero);
    }
    y[m] = std::isfinite(acc) ? acc : 0.f;
  }
}
#endif  // AVX2

float gemm_row_scalar(const float* x, const uint8_t* qbase, int M, int K, int m, int gs, int ng,
                      const uint16_t* scales, const uint16_t* zeros, bool awq) {
  (void)M;
  const int rb = row_bytes(K);
  const uint8_t* qrow = qbase + static_cast<size_t>(m) * rb;
  double acc = 0.0;
  for (int g = 0; g < ng; ++g) {
    const int k0 = g * gs;
    const int k1 = std::min(K, k0 + gs);
    const float scale = f16b_to_f32(scales[m * ng + g]);
    const float zero = (!awq && zeros) ? f16b_to_f32(zeros[m * ng + g]) : 0.f;
    for (int k = k0; k < k1; ++k) {
      const uint8_t b = qrow[k / 2];
      const int qi = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
      float w = awq ? static_cast<float>(qi - 7) * scale : static_cast<float>(qi) * scale + zero;
      acc += static_cast<double>(x[k]) * w;
    }
  }
  return static_cast<float>(acc);
}

Int4ArgmaxResult gemm_int4_argmax_impl(const float* x, const qlwc::Int4View& W) {
  Int4ArgmaxResult best{};
  const bool awq = W.scheme == qlwc::Scheme::kAwqSym;
  const int M = W.M, K = W.K, gs = W.group_size;
  const int ng = (K + gs - 1) / gs;
  const int rb = row_bytes(K);

  const float* scf = W.scales_f32;
  std::vector<float> scales_scratch;
  if (awq && !scf) {
    scales_scratch.resize(static_cast<size_t>(M) * ng);
    for (int i = 0; i < M * ng; ++i)
      scales_scratch[static_cast<size_t>(i)] = f16b_to_f32(W.scales[i]);
    scf = scales_scratch.data();
  }

  auto row_acc = [&](int m) -> float {
    float acc = 0.f;
    if (awq) {
      const uint8_t* qrow = W.qweight + static_cast<size_t>(m) * rb;
#if defined(LLMOC_ENABLE_AVX2)
      for (int g = 0; g < ng; ++g) {
        const int k0 = g * gs;
        const int k1 = std::min(K, k0 + gs);
        acc += dot_awq_noscale(x, qrow, k0, k1) * scf[static_cast<size_t>(m) * ng + g];
      }
#else
      acc = gemm_row_scalar(x, W.qweight, M, K, m, gs, ng, W.scales, W.zeros, true);
#endif
    } else {
      acc = gemm_row_scalar(x, W.qweight, M, K, m, gs, ng, W.scales, W.zeros, false);
    }
    return std::isfinite(acc) ? acc : -1e30f;
  };

#if defined(_OPENMP)
  if (M >= 4096) {
    int32_t lb = 0;
    float lv = -1e30f;
#pragma omp parallel
    {
      int32_t tb = 0;
      float tv = -1e30f;
#pragma omp for schedule(static) nowait
      for (int m = 0; m < M; ++m) {
        const float v = row_acc(m);
        if (v > tv) {
          tv = v;
          tb = m;
        }
      }
#pragma omp critical
      {
        if (tv > lv) {
          lv = tv;
          lb = tb;
        }
      }
    }
    best.index = lb;
    best.value = lv;
    return best;
  }
#endif

  for (int m = 0; m < M; ++m) {
    const float v = row_acc(m);
    if (v > best.value) {
      best.value = v;
      best.index = m;
    }
  }
  return best;
}

}  // namespace

void gemm_int4(const float* x, const qlwc::Int4View& W, float* y) {
  const bool awq = W.scheme == qlwc::Scheme::kAwqSym;

#if defined(LLMOC_ENABLE_AVX2)
  if (awq) {
    gemm_int4_awq_avx2(x, W, y);
    return;
  }
  gemm_int4_gptq_avx2(x, W, y);
#else
  const int M = W.M, K = W.K, gs = W.group_size;
  const int ng = (K + gs - 1) / gs;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 128)
#endif
  for (int m = 0; m < M; ++m) {
    y[m] = gemm_row_scalar(x, W.qweight, M, K, m, gs, ng, W.scales, W.zeros, awq);
    if (!std::isfinite(y[m])) y[m] = 0.f;
  }
#endif
}

Int4ArgmaxResult gemm_int4_argmax(const float* x, const qlwc::Int4View& W) {
  return gemm_int4_argmax_impl(x, W);
}

#if defined(LLMOC_ENABLE_AVX2)
void gemm_int4_awq_batch_avx2(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  const int M = W.M, K = W.K, gs = W.group_size;
  const int ng = (K + gs - 1) / gs;
  const int rb = row_bytes(K);

  const float* scf = W.scales_f32;
  std::vector<float> scales_scratch;
  if (!scf) {
    scales_scratch.resize(static_cast<size_t>(M) * ng);
    for (int i = 0; i < M * ng; ++i)
      scales_scratch[static_cast<size_t>(i)] = f16b_to_f32(W.scales[i]);
    scf = scales_scratch.data();
  }
  auto scale_at = [&](int m, int g) -> float { return scf[static_cast<size_t>(m) * ng + g]; };

  // 按 M 并行；同一组 weight 行对全部 token 复用
  if (M >= 4096) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int m0 = 0; m0 < M; m0 += 4) {
      if (m0 + 3 < M) {
        const uint8_t* q0 = W.qweight + static_cast<size_t>(m0) * rb;
        const uint8_t* q1 = W.qweight + static_cast<size_t>(m0 + 1) * rb;
        const uint8_t* q2 = W.qweight + static_cast<size_t>(m0 + 2) * rb;
        const uint8_t* q3 = W.qweight + static_cast<size_t>(m0 + 3) * rb;
        if (m0 + 4 < M) {
          _mm_prefetch(reinterpret_cast<const char*>(W.qweight + static_cast<size_t>(m0 + 4) * rb),
                       _MM_HINT_T0);
        }
        for (int t = 0; t < n; ++t) {
          const float* x = X + static_cast<size_t>(t) * K;
          float* y = Y + static_cast<size_t>(t) * M;
          float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
          for (int g = 0; g < ng; ++g) {
            const int k0 = g * gs;
            const int k1 = std::min(K, k0 + gs);
            float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
            dot_awq_noscale_4row(x, q0, q1, q2, q3, k0, k1, p0, p1, p2, p3);
            acc0 += p0 * scale_at(m0, g);
            acc1 += p1 * scale_at(m0 + 1, g);
            acc2 += p2 * scale_at(m0 + 2, g);
            acc3 += p3 * scale_at(m0 + 3, g);
          }
          y[m0] = std::isfinite(acc0) ? acc0 : 0.f;
          y[m0 + 1] = std::isfinite(acc1) ? acc1 : 0.f;
          y[m0 + 2] = std::isfinite(acc2) ? acc2 : 0.f;
          y[m0 + 3] = std::isfinite(acc3) ? acc3 : 0.f;
        }
      } else {
        for (int m = m0; m < M; ++m) {
          const uint8_t* qrow = W.qweight + static_cast<size_t>(m) * rb;
          for (int t = 0; t < n; ++t) {
            const float* x = X + static_cast<size_t>(t) * K;
            float acc = 0.f;
            for (int g = 0; g < ng; ++g) {
              const int k0 = g * gs;
              const int k1 = std::min(K, k0 + gs);
              acc += dot_awq_noscale(x, qrow, k0, k1) * scale_at(m, g);
            }
            Y[static_cast<size_t>(t) * M + m] = std::isfinite(acc) ? acc : 0.f;
          }
        }
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
    for (int m0 = 0; m0 < M; m0 += 2) {
      if (m0 + 1 < M) {
        const uint8_t* q0 = W.qweight + static_cast<size_t>(m0) * rb;
        const uint8_t* q1 = W.qweight + static_cast<size_t>(m0 + 1) * rb;
        for (int t = 0; t < n; ++t) {
          const float* x = X + static_cast<size_t>(t) * K;
          float* y = Y + static_cast<size_t>(t) * M;
          float acc0 = 0.f, acc1 = 0.f;
          for (int g = 0; g < ng; ++g) {
            const int k0 = g * gs;
            const int k1 = std::min(K, k0 + gs);
            float p0 = 0.f, p1 = 0.f;
            dot_awq_noscale_2row(x, q0, q1, k0, k1, p0, p1);
            acc0 += p0 * scale_at(m0, g);
            acc1 += p1 * scale_at(m0 + 1, g);
          }
          y[m0] = std::isfinite(acc0) ? acc0 : 0.f;
          y[m0 + 1] = std::isfinite(acc1) ? acc1 : 0.f;
        }
      } else {
        const uint8_t* qrow = W.qweight + static_cast<size_t>(m0) * rb;
        for (int t = 0; t < n; ++t) {
          const float* x = X + static_cast<size_t>(t) * K;
          float acc = 0.f;
          for (int g = 0; g < ng; ++g) {
            const int k0 = g * gs;
            const int k1 = std::min(K, k0 + gs);
            acc += dot_awq_noscale(x, qrow, k0, k1) * scale_at(m0, g);
          }
          Y[static_cast<size_t>(t) * M + m0] = std::isfinite(acc) ? acc : 0.f;
        }
      }
    }
  }
}
#endif

void gemm_int4_batch(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  if (n <= 0) return;
  if (n == 1) {
    gemm_int4(X, W, Y);
    return;
  }
  const bool awq = W.scheme == qlwc::Scheme::kAwqSym;
#if defined(LLMOC_ENABLE_AVX2)
  if (awq) {
    gemm_int4_awq_batch_avx2(X, n, W, Y);
    return;
  }
#endif
  // GPTQ / 无 AVX2：逐 token（正确优先）
  for (int t = 0; t < n; ++t)
    gemm_int4(X + static_cast<size_t>(t) * W.K, W, Y + static_cast<size_t>(t) * W.M);
}

void dequant_int4_row(const qlwc::Int4View& W, int row, float* out) {
  if (row < 0 || row >= W.M) throw std::runtime_error("int4 row OOB");
  const int K = W.K, gs = W.group_size;
  const int ng = (K + gs - 1) / gs;
  const bool awq = W.scheme == qlwc::Scheme::kAwqSym;
  const int rb = row_bytes(K);
  const uint8_t* qrow = W.qweight + static_cast<size_t>(row) * rb;
  for (int g = 0; g < ng; ++g) {
    const int k0 = g * gs;
    const int k1 = std::min(K, k0 + gs);
    const float scale = f16b_to_f32(W.scales[row * ng + g]);
    const float zero = (!awq && W.zeros) ? f16b_to_f32(W.zeros[row * ng + g]) : 0.f;
    for (int k = k0; k < k1; ++k) {
      const uint8_t b = qrow[k / 2];
      const int qi = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
      if (awq)
        out[k] = static_cast<float>(qi - 7) * scale;
      else
        out[k] = static_cast<float>(qi) * scale + zero;
    }
  }
}

void dequant_int4_matrix(const qlwc::Int4View& W, float* out) {
  if (!out || !W.qweight || W.M <= 0 || W.K <= 0) throw std::runtime_error("dequant_int4_matrix bad args");
  for (int row = 0; row < W.M; ++row)
    dequant_int4_row(W, row, out + static_cast<size_t>(row) * W.K);
}

}  // namespace llmoc::hal
