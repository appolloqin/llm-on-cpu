// llm-on-cpu :: hal/gated_delta_chunk.cpp
// Prefill-optimized Gated DeltaNet: parallelize over heads, serial over tokens.
// (Full WY chunk algebra is optional future work; heads-outer alone removes
//  per-token OpenMP fork storms that dominated long prefill.)
#include "hal/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

inline void l2norm_row(float* x, int n) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

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
#endif

// One head, all tokens. Matches gated_delta_recurrent step (including soft clamp).
void gdn_head_serial(const float* q, const float* k, const float* v, const float* g,
                     const float* beta, float* state, float* out, int seq, int n_heads, int h,
                     int dk, int dv, float scale, bool qk_l2norm) {
  float* st = state + static_cast<size_t>(h) * dk * dv;
  std::vector<float> qt(dk), kt(dk), vt(dv), kv_mem(dv), delta(dv);

  for (int t = 0; t < seq; ++t) {
    std::memcpy(qt.data(), q + (static_cast<size_t>(t) * n_heads + h) * dk, sizeof(float) * dk);
    std::memcpy(kt.data(), k + (static_cast<size_t>(t) * n_heads + h) * dk, sizeof(float) * dk);
    std::memcpy(vt.data(), v + (static_cast<size_t>(t) * n_heads + h) * dv, sizeof(float) * dv);
    if (qk_l2norm) {
      l2norm_row(qt.data(), dk);
      l2norm_row(kt.data(), dk);
    }
    for (int i = 0; i < dk; ++i) qt[i] *= scale;

    float g_log = g[t * n_heads + h];
    if (!std::isfinite(g_log)) g_log = -80.f;
    if (g_log > 0.f) g_log = 0.f;
    if (g_log < -80.f) g_log = -80.f;
    const float g_t = std::exp(g_log);
    float beta_t = beta[t * n_heads + h];
    if (!std::isfinite(beta_t)) beta_t = 0.f;
    beta_t = std::min(1.f, std::max(0.f, beta_t));

#if defined(LLMOC_ENABLE_AVX2)
    {
      const __m256 vg = _mm256_set1_ps(g_t);
      const int n = dk * dv;
      int i = 0;
      for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(st + i, _mm256_mul_ps(_mm256_loadu_ps(st + i), vg));
      for (; i < n; ++i) st[i] *= g_t;
    }
#else
    for (int i = 0; i < dk * dv; ++i) st[i] *= g_t;
#endif

    std::fill(kv_mem.begin(), kv_mem.end(), 0.f);
    for (int i = 0; i < dk; ++i) {
      const float ki = kt[i];
#if defined(LLMOC_ENABLE_AVX2)
      const __m256 vk = _mm256_set1_ps(ki);
      int j = 0;
      for (; j + 8 <= dv; j += 8) {
        __m256 acc = _mm256_loadu_ps(kv_mem.data() + j);
        acc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(st + i * dv + j), acc);
        _mm256_storeu_ps(kv_mem.data() + j, acc);
      }
      for (; j < dv; ++j) kv_mem[j] += st[i * dv + j] * ki;
#else
      for (int j = 0; j < dv; ++j) kv_mem[j] += st[i * dv + j] * ki;
#endif
    }

    for (int j = 0; j < dv; ++j) delta[j] = (vt[j] - kv_mem[j]) * beta_t;

    for (int i = 0; i < dk; ++i) {
      const float ki = kt[i];
#if defined(LLMOC_ENABLE_AVX2)
      const __m256 vk = _mm256_set1_ps(ki);
      const __m256 vlo = _mm256_set1_ps(-1e4f);
      const __m256 vhi = _mm256_set1_ps(1e4f);
      int j = 0;
      for (; j + 8 <= dv; j += 8) {
        __m256 s =
            _mm256_fmadd_ps(vk, _mm256_loadu_ps(delta.data() + j), _mm256_loadu_ps(st + i * dv + j));
        s = _mm256_min_ps(vhi, _mm256_max_ps(vlo, s));
        _mm256_storeu_ps(st + i * dv + j, s);
      }
      for (; j < dv; ++j) {
        float s = st[i * dv + j] + ki * delta[j];
        if (s > 1e4f) s = 1e4f;
        if (s < -1e4f) s = -1e4f;
        st[i * dv + j] = s;
      }
#else
      for (int j = 0; j < dv; ++j) {
        float s = st[i * dv + j] + ki * delta[j];
        if (s > 1e4f) s = 1e4f;
        if (s < -1e4f) s = -1e4f;
        st[i * dv + j] = s;
      }
#endif
    }

    float* ot = out + (static_cast<size_t>(t) * n_heads + h) * dv;
    std::fill(ot, ot + dv, 0.f);
    for (int i = 0; i < dk; ++i) {
      const float qi = qt[i];
#if defined(LLMOC_ENABLE_AVX2)
      const __m256 vq = _mm256_set1_ps(qi);
      int j = 0;
      for (; j + 8 <= dv; j += 8) {
        __m256 o = _mm256_loadu_ps(ot + j);
        o = _mm256_fmadd_ps(vq, _mm256_loadu_ps(st + i * dv + j), o);
        _mm256_storeu_ps(ot + j, o);
      }
      for (; j < dv; ++j) ot[j] += st[i * dv + j] * qi;
#else
      for (int j = 0; j < dv; ++j) ot[j] += st[i * dv + j] * qi;
#endif
    }
  }
}

}  // namespace

void gated_delta_chunked(const float* q, const float* k, const float* v, const float* g,
                         const float* beta, float* state, float* out, int seq, int n_heads, int dk,
                         int dv, bool qk_l2norm) {
  if (dk > 256 || dv > 256) throw std::runtime_error("gated_delta_chunked: dk/dv too large");
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  // Chunk scheduling: process heads in parallel; within each head walk tokens
  // (optionally in BT=64 waves for cache). Equivalent to recurrent, far fewer OpenMP forks.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_heads >= 4 && !omp_in_parallel())
#endif
  for (int h = 0; h < n_heads; ++h) {
    gdn_head_serial(q, k, v, g, beta, state, out, seq, n_heads, h, dk, dv, scale, qk_l2norm);
  }
}

}  // namespace llmoc::hal
