// llm-on-cpu :: hal/cpu_ops_attn.cpp
// Softmax + causal attn prefill — compiled WITHOUT -mavx2/-mfma (see CMake).
// Linux GCC CI: same TU as AVX kernels made attn disagree with a non-AVX naive ref
// (~1e10). Mac CI is arm64 (no AVX); MSVC is less sensitive. Keep this file scalar.

#include "hal/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-tree-vectorize")
#endif

namespace llmoc::hal {

void softmax_inplace(float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - m);
    s += x[i];
  }
  const float inv = static_cast<float>(1.0 / s);
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

namespace {

void attn_prefill_one(const float* q, const float* k, const float* v, float* out, int n_heads,
                      int n_kv_heads, int head_dim, float scale, int tq, int h, float* scores) {
  const int g = n_heads / n_kv_heads;
  const int hkv = h / g;
  const float* qh = q + (static_cast<size_t>(tq) * n_heads + h) * head_dim;
  for (int tk = 0; tk <= tq; ++tk) {
    const float* kt = k + (static_cast<size_t>(tk) * n_kv_heads + hkv) * head_dim;
    double dot = 0.0;
    for (int d = 0; d < head_dim; ++d) {
      dot += static_cast<double>(qh[d]) * static_cast<double>(kt[d]);
    }
    scores[tk] = static_cast<float>(dot) * scale;
  }
  softmax_inplace(scores, tq + 1);
  float* oh = out + (static_cast<size_t>(tq) * n_heads + h) * head_dim;
  std::fill(oh, oh + head_dim, 0.f);
  for (int tk = 0; tk <= tq; ++tk) {
    const float* vt = v + (static_cast<size_t>(tk) * n_kv_heads + hkv) * head_dim;
    const float s = scores[tk];
    for (int d = 0; d < head_dim; ++d) oh[d] += s * vt[d];
  }
}

}  // namespace

void attn_prefill(const float* q, const float* k, const float* v, float* out, int seq, int n_heads,
                  int n_kv_heads, int head_dim, float scale) {
  // Parallel over (query × head). Scratch indexed by omp thread id.
  const int work = seq * n_heads;
  int nthreads = 1;
#if defined(_OPENMP)
  if (work >= 64) nthreads = std::max(1, omp_get_max_threads());
#endif
  std::vector<float> scratch(static_cast<size_t>(nthreads) * static_cast<size_t>(seq));

#if defined(_OPENMP)
  if (work >= 64) {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      float* scores = scratch.data() + static_cast<size_t>(tid) * static_cast<size_t>(seq);
#pragma omp for schedule(static)
      for (int wi = 0; wi < work; ++wi) {
        attn_prefill_one(q, k, v, out, n_heads, n_kv_heads, head_dim, scale, wi / n_heads,
                         wi % n_heads, scores);
      }
    }
    return;
  }
#endif
  float* scores = scratch.data();
  for (int wi = 0; wi < work; ++wi) {
    attn_prefill_one(q, k, v, out, n_heads, n_kv_heads, head_dim, scale, wi / n_heads,
                     wi % n_heads, scores);
  }
}

}  // namespace llmoc::hal
