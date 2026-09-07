// llm-on-cpu :: hal/cpu_ops.cpp
#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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

inline __m256 load8_w_f32(const uint16_t* p, WDtype dt) {
  if (dt == WDtype::kF16) {
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; ++i) tmp[i] = f16_to_f32(p[i]);
    return _mm256_load_ps(tmp);
  }
  // BF16: (u16<<16) as f32
  __m128i v16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  __m256i v32 = _mm256_cvtepu16_epi32(v16);
  v32 = _mm256_slli_epi32(v32, 16);
  return _mm256_castsi256_ps(v32);
}
#endif

}  // namespace

void bf16_to_f32_buf(const uint16_t* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}
void f32_to_bf16_buf(const float* src, uint16_t* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = f32_to_bf16(src[i]);
}

void gemm_bias_free(const float* x, const uint16_t* W, float* y, int M, int K, WDtype dt,
                    bool allow_gpu) {
  // M5: optional CUDA path — inactive unless hal::cuda::enable(); pure_cpu identical.
  if (allow_gpu && cuda::try_gemm_w16(x, W, y, M, K, dt == WDtype::kF16)) return;
#if defined(LLMOC_ENABLE_AVX2)
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 128)
#endif
  for (int m = 0; m < M; ++m) {
    const uint16_t* row = W + static_cast<size_t>(m) * K;
    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
      vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), load8_w_f32(row + k, dt), vacc0);
      vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), load8_w_f32(row + k + 8, dt), vacc1);
      vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16), load8_w_f32(row + k + 16, dt), vacc2);
      vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24), load8_w_f32(row + k + 24, dt), vacc3);
    }
    __m256 vacc = _mm256_add_ps(_mm256_add_ps(vacc0, vacc1), _mm256_add_ps(vacc2, vacc3));
    float acc = hsum256(vacc);
    for (; k < K; ++k) acc += x[k] * load_w(row + k, dt);
    y[m] = std::isfinite(acc) ? acc : 0.f;
  }
#else
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 128)
#endif
  for (int m = 0; m < M; ++m) {
    const uint16_t* row = W + static_cast<size_t>(m) * K;
    double acc = 0.0;
    for (int k = 0; k < K; ++k) acc += static_cast<double>(x[k]) * load_w(row + k, dt);
    const float o = static_cast<float>(acc);
    y[m] = std::isfinite(o) ? o : 0.f;
  }
#endif
}

void gemm_bias_free_batch(const float* X, int n, const uint16_t* W, float* Y, int M, int K,
                          WDtype dt) {
  if (n <= 0 || !X || !W || !Y || M <= 0 || K <= 0) return;
  if (n == 1) {
    gemm_bias_free(X, W, Y, M, K, dt);
    return;
  }
  if (cuda::try_gemm_w16_batch(X, n, W, Y, M, K, dt == WDtype::kF16)) return;
  // Weight-stationary: each W row streamed once across all tokens (critical for long prefill).
#if defined(LLMOC_ENABLE_AVX2)
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
  for (int m = 0; m < M; ++m) {
    const uint16_t* row = W + static_cast<size_t>(m) * K;
    for (int t = 0; t < n; ++t) {
      const float* x = X + static_cast<size_t>(t) * K;
      __m256 vacc0 = _mm256_setzero_ps();
      __m256 vacc1 = _mm256_setzero_ps();
      __m256 vacc2 = _mm256_setzero_ps();
      __m256 vacc3 = _mm256_setzero_ps();
      int k = 0;
      for (; k + 32 <= K; k += 32) {
        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), load8_w_f32(row + k, dt), vacc0);
        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), load8_w_f32(row + k + 8, dt), vacc1);
        vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16), load8_w_f32(row + k + 16, dt), vacc2);
        vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24), load8_w_f32(row + k + 24, dt), vacc3);
      }
      __m256 vacc = _mm256_add_ps(_mm256_add_ps(vacc0, vacc1), _mm256_add_ps(vacc2, vacc3));
      float acc = hsum256(vacc);
      for (; k < K; ++k) acc += x[k] * load_w(row + k, dt);
      Y[static_cast<size_t>(t) * M + m] = std::isfinite(acc) ? acc : 0.f;
    }
  }
#else
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (M >= 64)
#endif
  for (int m = 0; m < M; ++m) {
    const uint16_t* row = W + static_cast<size_t>(m) * K;
    for (int t = 0; t < n; ++t) {
      const float* x = X + static_cast<size_t>(t) * K;
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += static_cast<double>(x[k]) * load_w(row + k, dt);
      const float o = static_cast<float>(acc);
      Y[static_cast<size_t>(t) * M + m] = std::isfinite(o) ? o : 0.f;
    }
  }
#endif
}

void rmsnorm(const float* x, const uint16_t* w, float* y, int n, float eps, WDtype dt,
             bool one_plus_weight) {
  double ss = 0.0;
#if defined(LLMOC_ENABLE_AVX2)
  {
    __m256 vacc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      __m256 v = _mm256_loadu_ps(x + i);
      // zero non-finite → treat as 0 via mask is expensive; keep scalar check rare
      vacc = _mm256_fmadd_ps(v, v, vacc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vacc);
    for (int j = 0; j < 8; ++j) ss += tmp[j];
    for (; i < n; ++i) {
      float v = x[i];
      if (!std::isfinite(v)) v = 0.f;
      ss += static_cast<double>(v) * v;
    }
  }
#else
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    if (!std::isfinite(v)) v = 0.f;
    ss += static_cast<double>(v) * v;
  }
#endif
  const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
#if defined(LLMOC_ENABLE_AVX2)
  int i = 0;
  const __m256 vinv = _mm256_set1_ps(inv);
  const __m256 vone = _mm256_set1_ps(1.f);
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    __m256 scale = load8_w_f32(w + i, dt);
    if (one_plus_weight) scale = _mm256_add_ps(vone, scale);
    __m256 o = _mm256_mul_ps(_mm256_mul_ps(v, vinv), scale);
    _mm256_storeu_ps(y + i, o);
  }
  for (; i < n; ++i) {
    float v = x[i];
    if (!std::isfinite(v)) v = 0.f;
    float scale = load_w(w + i, dt);
    if (one_plus_weight) scale = 1.f + scale;
    float o = v * inv * scale;
    y[i] = std::isfinite(o) ? o : 0.f;
  }
#else
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    if (!std::isfinite(v)) v = 0.f;
    float scale = load_w(w + i, dt);
    if (one_plus_weight) scale = 1.f + scale;
    float o = v * inv * scale;
    y[i] = std::isfinite(o) ? o : 0.f;
  }
#endif
}

void rmsnorm_gated(const float* x, const float* gate, const uint16_t* w, float* y, int n, float eps,
                   WDtype dt) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    if (!std::isfinite(v)) v = 0.f;
    ss += static_cast<double>(v) * v;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    if (!std::isfinite(v)) v = 0.f;
    float g = gate[i];
    if (!std::isfinite(g)) g = 0.f;
    const float silu = g / (1.f + std::exp(-g));
    float o = v * inv * load_w(w + i, dt) * silu;
    y[i] = std::isfinite(o) ? o : 0.f;
  }
}

void silu_and_mul(const float* gate, const float* up, float* out, int n) {
#if defined(LLMOC_ENABLE_AVX2)
  int i = 0;
  const __m256 one = _mm256_set1_ps(1.f);
  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(gate + i);
    __m256 u = _mm256_loadu_ps(up + i);
    // silu(g) = g / (1 + exp(-g))
    alignas(32) float gt[8], ot[8];
    _mm256_store_ps(gt, g);
    for (int j = 0; j < 8; ++j) ot[j] = gt[j] / (1.f + std::exp(-gt[j]));
    __m256 s = _mm256_load_ps(ot);
    _mm256_storeu_ps(out + i, _mm256_mul_ps(s, u));
  }
  for (; i < n; ++i) {
    const float g = gate[i];
    out[i] = (g / (1.f + std::exp(-g))) * up[i];
  }
  (void)one;
#else
  for (int i = 0; i < n; ++i) {
    const float g = gate[i];
    out[i] = (g / (1.f + std::exp(-g))) * up[i];
  }
#endif
}

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

void apply_rope_freqs(float* q_or_k, int head_dim, int rotary_dim, int pos, float theta) {
  apply_mrope_freqs(q_or_k, head_dim, rotary_dim, pos, pos, pos, theta, nullptr, false);
}

void apply_mrope_freqs(float* q_or_k, int head_dim, int rotary_dim, int pos_t, int pos_h, int pos_w,
                       float theta, const int section[3], bool interleaved) {
  (void)head_dim;
  const int half = rotary_dim / 2;
  if (half <= 0 || half > 128 || rotary_dim > 256) return;
  float ang[128];
  const bool thw_equal = (pos_t == pos_h && pos_h == pos_w);
  const bool valid_sec = section && (section[0] + section[1] + section[2] == half);
  // 纯文本或无 section：标准 1D（T=H=W 时 interleaved 也等价于 1D）
  if (!valid_sec || thw_equal) {
    for (int i = 0; i < half; ++i) {
      const float freq = 1.f / std::pow(theta, static_cast<float>(i) / half);
      ang[i] = static_cast<float>(pos_t) * freq;
    }
  } else {
    float freqs[3][128];
    const int pos3[3] = {pos_t, pos_h, pos_w};
    for (int d = 0; d < 3; ++d) {
      for (int i = 0; i < half; ++i) {
        const float freq = 1.f / std::pow(theta, static_cast<float>(i) / half);
        freqs[d][i] = static_cast<float>(pos3[d]) * freq;
      }
    }
    if (interleaved) {
      for (int i = 0; i < half; ++i) ang[i] = freqs[0][i];
      for (int dim = 1; dim <= 2; ++dim) {
        const int length = section[dim] * 3;
        for (int i = dim; i < length && i < half; i += 3) ang[i] = freqs[dim][i];
      }
    } else {
      int off = 0;
      for (int dim = 0; dim < 3; ++dim) {
        for (int i = 0; i < section[dim] && off + i < half; ++i) ang[off + i] = freqs[dim][off + i];
        off += section[dim];
      }
    }
  }
  float rot[256];
  for (int i = 0; i < half; ++i) {
    const float c = std::cos(ang[i]), s = std::sin(ang[i]);
    const float x1 = q_or_k[i], x2 = q_or_k[i + half];
    rot[i] = x1 * c - x2 * s;
    rot[i + half] = x1 * s + x2 * c;
  }
  std::memcpy(q_or_k, rot, sizeof(float) * rotary_dim);
}

void attn_decode_one(const float* q, const float* k_cache, const float* v_cache, float* out,
                     int n_heads, int n_kv_heads, int head_dim, int seq_len, int cache_cap,
                     float scale) {
  const int g = n_heads / n_kv_heads;
  const int stride = cache_cap > 0 ? cache_cap : seq_len;
  thread_local std::vector<float> scores;
  if (scores.size() < static_cast<size_t>(seq_len)) scores.resize(static_cast<size_t>(seq_len));
  for (int h = 0; h < n_heads; ++h) {
    const int hkv = h / g;
    const float* qh = q + h * head_dim;
    for (int t = 0; t < seq_len; ++t) {
      const float* kt = k_cache + (static_cast<size_t>(hkv) * stride + t) * head_dim;
#if defined(LLMOC_ENABLE_AVX2)
      __m256 vacc0 = _mm256_setzero_ps();
      __m256 vacc1 = _mm256_setzero_ps();
      int d = 0;
      for (; d + 16 <= head_dim; d += 16) {
        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d), _mm256_loadu_ps(kt + d), vacc0);
        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d + 8), _mm256_loadu_ps(kt + d + 8), vacc1);
      }
      float dot = hsum256(_mm256_add_ps(vacc0, vacc1));
      for (; d < head_dim; ++d) dot += qh[d] * kt[d];
      scores[t] = dot * scale;
#else
      double dot = 0.0;
      for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(qh[d]) * kt[d];
      scores[t] = static_cast<float>(dot) * scale;
#endif
    }
    softmax_inplace(scores.data(), seq_len);
    float* oh = out + h * head_dim;
    std::fill(oh, oh + head_dim, 0.f);
    for (int t = 0; t < seq_len; ++t) {
      const float* vt = v_cache + (static_cast<size_t>(hkv) * stride + t) * head_dim;
      const float s = scores[t];
#if defined(LLMOC_ENABLE_AVX2)
      const __m256 vs = _mm256_set1_ps(s);
      int d = 0;
      for (; d + 8 <= head_dim; d += 8) {
        __m256 o = _mm256_loadu_ps(oh + d);
        o = _mm256_fmadd_ps(vs, _mm256_loadu_ps(vt + d), o);
        _mm256_storeu_ps(oh + d, o);
      }
      for (; d < head_dim; ++d) oh[d] += s * vt[d];
#else
      for (int d = 0; d < head_dim; ++d) oh[d] += s * vt[d];
#endif
    }
  }
}

void attn_prefill(const float* q, const float* k, const float* v, float* out, int seq, int n_heads,
                  int n_kv_heads, int head_dim, float scale) {
  // Parallel over (query×head). Per-iteration score buffer (MSVC OpenMP-safe).
  // Note: keep scalar dots here — prior dual-AVX+hsum path produced huge errors on MSVC;
  // attn_decode_one retains AVX for the decode hot path.
  const int g = n_heads / n_kv_heads;
  const int work = seq * n_heads;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (work >= 64)
#endif
  for (int wi = 0; wi < work; ++wi) {
    const int tq = wi / n_heads;
    const int h = wi % n_heads;
    const int hkv = h / g;
    std::vector<float> scores(static_cast<size_t>(tq) + 1u);
    const float* qh = q + (static_cast<size_t>(tq) * n_heads + h) * head_dim;
    for (int tk = 0; tk <= tq; ++tk) {
      const float* kt = k + (static_cast<size_t>(tk) * n_kv_heads + hkv) * head_dim;
      float dot = 0.f;
      for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
      scores[tk] = dot * scale;
    }
    float m = scores[0];
    for (int i = 1; i <= tq; ++i) m = std::max(m, scores[i]);
    float sum = 0.f;
    for (int i = 0; i <= tq; ++i) {
      scores[i] = std::exp(scores[i] - m);
      sum += scores[i];
    }
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int i = 0; i <= tq; ++i) scores[i] *= inv;
    float* oh = out + (static_cast<size_t>(tq) * n_heads + h) * head_dim;
    std::fill(oh, oh + head_dim, 0.f);
    for (int tk = 0; tk <= tq; ++tk) {
      const float* vt = v + (static_cast<size_t>(tk) * n_kv_heads + hkv) * head_dim;
      const float s = scores[tk];
      for (int d = 0; d < head_dim; ++d) oh[d] += s * vt[d];
    }
  }
}

void gated_delta_recurrent(const float* q, const float* k, const float* v, const float* g,
                           const float* beta, float* state, float* out, int seq, int n_heads,
                           int dk, int dv, bool qk_l2norm) {
  // Prefill/decode: heads-outer path (gated_delta_chunked). Avoids forking OpenMP once per token.
  gated_delta_chunked(q, k, v, g, beta, state, out, seq, n_heads, dk, dv, qk_l2norm);
}

}  // namespace llmoc::hal
