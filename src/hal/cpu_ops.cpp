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

void gemm_bias_free(const float* x, const uint16_t* W, float* y, int M, int K, WDtype dt) {
  // M5: optional CUDA path — inactive unless hal::cuda::enable(); pure_cpu identical.
  if (cuda::try_gemm_w16(x, W, y, M, K, dt == WDtype::kF16)) return;
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
  std::vector<float> scores(seq_len);
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
  const int g = n_heads / n_kv_heads;
  std::vector<float> scores(seq);
  for (int tq = 0; tq < seq; ++tq) {
    for (int h = 0; h < n_heads; ++h) {
      const int hkv = h / g;
      const float* qh = q + (static_cast<size_t>(tq) * n_heads + h) * head_dim;
      for (int tk = 0; tk <= tq; ++tk) {
        const float* kt = k + (static_cast<size_t>(tk) * n_kv_heads + hkv) * head_dim;
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(qh[d]) * kt[d];
        scores[tk] = static_cast<float>(dot) * scale;
      }
      softmax_inplace(scores.data(), tq + 1);
      float* oh = out + (static_cast<size_t>(tq) * n_heads + h) * head_dim;
      std::fill(oh, oh + head_dim, 0.f);
      for (int tk = 0; tk <= tq; ++tk) {
        const float* vt = v + (static_cast<size_t>(tk) * n_kv_heads + hkv) * head_dim;
        for (int d = 0; d < head_dim; ++d) oh[d] += scores[tk] * vt[d];
      }
    }
  }
}

static void l2norm_row(float* x, int n) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

#if defined(LLMOC_ENABLE_AVX2)
// 清除 NaN（保留 Inf 交给后续 clamp）；长 prefill/视觉路径否则会污染 GDN 递推状态
inline __m256 sanitize_ord_ps(__m256 v) {
  const __m256 ord = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
  return _mm256_and_ps(v, ord);
}

// Qwen3.5 固定 dk=dv=128：AVX clamp + NaN 清洗（视觉多 token prefill 必需）
static void gated_delta_head_128(const float* q, const float* k, const float* v, float g_log,
                                 float beta_t, float* st, float* ot, float scale, bool qk_l2norm) {
  alignas(32) float qt[128], kt[128], vt[128], kv_mem[128], delta[128];
  std::memcpy(qt, q, sizeof(qt));
  std::memcpy(kt, k, sizeof(kt));
  std::memcpy(vt, v, sizeof(vt));
  if (qk_l2norm) {
    l2norm_row(qt, 128);
    l2norm_row(kt, 128);
  }
  const __m256 vscale = _mm256_set1_ps(scale);
  for (int i = 0; i < 128; i += 8)
    _mm256_store_ps(qt + i, sanitize_ord_ps(_mm256_mul_ps(_mm256_load_ps(qt + i), vscale)));

  if (!std::isfinite(g_log)) g_log = -80.f;
  if (g_log > 0.f) g_log = 0.f;
  if (g_log < -80.f) g_log = -80.f;
  const float g_t = std::exp(g_log);
  if (!std::isfinite(beta_t)) beta_t = 0.f;
  beta_t = std::min(1.f, std::max(0.f, beta_t));

  const __m256 vg = _mm256_set1_ps(g_t);
  for (int i = 0; i < 128 * 128; i += 8) {
    __m256 s = sanitize_ord_ps(_mm256_loadu_ps(st + i));
    _mm256_storeu_ps(st + i, _mm256_mul_ps(s, vg));
  }

  std::memset(kv_mem, 0, sizeof(kv_mem));
  for (int i = 0; i < 128; ++i) {
    const float ki = kt[i];
    if (!std::isfinite(ki)) continue;
    const __m256 vk = _mm256_set1_ps(ki);
    float* row = st + i * 128;
    for (int j = 0; j < 128; j += 8) {
      __m256 acc = _mm256_load_ps(kv_mem + j);
      acc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(row + j), acc);
      _mm256_store_ps(kv_mem + j, sanitize_ord_ps(acc));
    }
  }

  const __m256 vb = _mm256_set1_ps(beta_t);
  for (int j = 0; j < 128; j += 8) {
    __m256 d = _mm256_mul_ps(
        _mm256_sub_ps(sanitize_ord_ps(_mm256_load_ps(vt + j)), _mm256_load_ps(kv_mem + j)), vb);
    _mm256_store_ps(delta + j, sanitize_ord_ps(d));
  }

  const __m256 vlo = _mm256_set1_ps(-1e4f);
  const __m256 vhi = _mm256_set1_ps(1e4f);
  for (int i = 0; i < 128; ++i) {
    const float ki = kt[i];
    if (!std::isfinite(ki)) continue;
    const __m256 vk = _mm256_set1_ps(ki);
    float* row = st + i * 128;
    for (int j = 0; j < 128; j += 8) {
      __m256 s = _mm256_fmadd_ps(vk, _mm256_load_ps(delta + j), _mm256_loadu_ps(row + j));
      s = sanitize_ord_ps(s);
      s = _mm256_min_ps(vhi, _mm256_max_ps(vlo, s));
      _mm256_storeu_ps(row + j, s);
    }
  }

  std::memset(ot, 0, 128 * sizeof(float));
  for (int i = 0; i < 128; ++i) {
    const float qi = qt[i];
    if (!std::isfinite(qi)) continue;
    const __m256 vq = _mm256_set1_ps(qi);
    const float* row = st + i * 128;
    for (int j = 0; j < 128; j += 8) {
      __m256 o = _mm256_loadu_ps(ot + j);
      o = _mm256_fmadd_ps(vq, _mm256_loadu_ps(row + j), o);
      _mm256_storeu_ps(ot + j, sanitize_ord_ps(o));
    }
  }
}
#endif

void gated_delta_recurrent(const float* q, const float* k, const float* v, const float* g,
                           const float* beta, float* state, float* out, int seq, int n_heads,
                           int dk, int dv, bool qk_l2norm) {
  // Layouts (token-major):
  // q/k: [seq, n_heads, dk]; v: [seq, n_heads, dv]; g/beta: [seq, n_heads]
  // state: [n_heads, dk, dv]
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  constexpr int kMaxD = 256;
  if (dk > kMaxD || dv > kMaxD) throw std::runtime_error("gated_delta_recurrent: dk/dv too large");

#if defined(LLMOC_ENABLE_AVX2)
  if (dk == 128 && dv == 128) {
    for (int t = 0; t < seq; ++t) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_heads >= 8 && !omp_in_parallel())
#endif
      for (int h = 0; h < n_heads; ++h) {
        const size_t qk = (static_cast<size_t>(t) * n_heads + h) * 128;
        gated_delta_head_128(q + qk, k + qk, v + (static_cast<size_t>(t) * n_heads + h) * 128,
                             g[t * n_heads + h], beta[t * n_heads + h],
                             state + static_cast<size_t>(h) * 128 * 128,
                             out + (static_cast<size_t>(t) * n_heads + h) * 128, scale, qk_l2norm);
      }
    }
    return;
  }
#endif

  for (int t = 0; t < seq; ++t) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_heads >= 8 && !omp_in_parallel())
#endif
    for (int h = 0; h < n_heads; ++h) {
      float qt[kMaxD], kt[kMaxD], vt[kMaxD], kv_mem[kMaxD], delta[kMaxD];
      std::memcpy(qt, q + (static_cast<size_t>(t) * n_heads + h) * dk, sizeof(float) * dk);
      std::memcpy(kt, k + (static_cast<size_t>(t) * n_heads + h) * dk, sizeof(float) * dk);
      std::memcpy(vt, v + (static_cast<size_t>(t) * n_heads + h) * dv, sizeof(float) * dv);
      if (qk_l2norm) {
        l2norm_row(qt, dk);
        l2norm_row(kt, dk);
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
      float* st = state + (static_cast<size_t>(h) * dk * dv);

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

      std::memset(kv_mem, 0, sizeof(float) * dv);
      for (int i = 0; i < dk; ++i) {
        const float ki = kt[i];
#if defined(LLMOC_ENABLE_AVX2)
        const __m256 vk = _mm256_set1_ps(ki);
        int j = 0;
        for (; j + 8 <= dv; j += 8) {
          __m256 acc = _mm256_loadu_ps(kv_mem + j);
          acc = _mm256_fmadd_ps(vk, _mm256_loadu_ps(st + i * dv + j), acc);
          _mm256_storeu_ps(kv_mem + j, acc);
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
          __m256 s = _mm256_fmadd_ps(vk, _mm256_loadu_ps(delta + j), _mm256_loadu_ps(st + i * dv + j));
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
}

}  // namespace llmoc::hal
