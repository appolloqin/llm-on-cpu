// llm-on-cpu :: glm/ops/glm_ops.cpp
#include "glm/ops/glm_ops.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace llmoc::glm::ops {
namespace {

float sigmoid(float x) {
  if (x > 20.f) return 1.f;
  if (x < -20.f) return 0.f;
  return 1.f / (1.f + std::exp(-x));
}

}  // namespace

void rmsnorm(const float* x, const uint16_t* w, float* y, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
  for (int i = 0; i < n; ++i) y[i] = x[i] * inv * bf16_to_f32(w[i]);
}

void gemm_bf16(const float* x, const uint16_t* W, float* y, int M, int K) {
  for (int m = 0; m < M; ++m) {
    const uint16_t* row = W + static_cast<size_t>(m) * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += x[k] * bf16_to_f32(row[k]);
    y[m] = std::isfinite(acc) ? acc : 0.f;
  }
}

void silu_mul(const float* gate, const float* up, float* out, int n) {
  for (int i = 0; i < n; ++i) {
    const float g = gate[i];
    out[i] = (g / (1.f + std::exp(-g))) * up[i];
  }
}

void softmax(float* x, int n) {
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

void rope_inplace(float* qk, int head_dim, int pos, float theta) {
  const int half = head_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float freq = 1.f / std::pow(theta, static_cast<float>(i) / half);
    const float ang = static_cast<float>(pos) * freq;
    const float c = std::cos(ang), s = std::sin(ang);
    const float a = qk[i], b = qk[i + half];
    qk[i] = a * c - b * s;
    qk[i + half] = a * s + b * c;
  }
}

void gqa_attn(const float* q, const float* k_cache, const float* v_cache, float* out, int n_heads,
              int n_kv, int head_dim, int max_seq, int pos) {
  std::vector<int> idx(static_cast<size_t>(pos + 1));
  std::iota(idx.begin(), idx.end(), 0);
  gqa_attn_indexed(q, k_cache, v_cache, out, n_heads, n_kv, head_dim, max_seq, idx.data(),
                   pos + 1);
}

void gqa_attn_indexed(const float* q, const float* k_cache, const float* v_cache, float* out,
                      int n_heads, int n_kv, int head_dim, int max_seq, const int* idx, int n_idx) {
  if (n_idx <= 0) {
    std::fill(out, out + n_heads * head_dim, 0.f);
    return;
  }
  const int rep = n_heads / std::max(n_kv, 1);
  std::vector<float> scores(static_cast<size_t>(n_idx));
  for (int h = 0; h < n_heads; ++h) {
    const int kv_h = h / rep;
    const float* qh = q + h * head_dim;
    float* oh = out + h * head_dim;
    for (int i = 0; i < n_idx; ++i) {
      const int t = idx[i];
      const float* kh = k_cache + (static_cast<size_t>(kv_h) * max_seq + t) * head_dim;
      float dot = 0.f;
      for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
      scores[i] = dot / std::sqrt(static_cast<float>(head_dim));
    }
    softmax(scores.data(), n_idx);
    for (int d = 0; d < head_dim; ++d) oh[d] = 0.f;
    for (int i = 0; i < n_idx; ++i) {
      const int t = idx[i];
      const float* vh = v_cache + (static_cast<size_t>(kv_h) * max_seq + t) * head_dim;
      for (int d = 0; d < head_dim; ++d) oh[d] += scores[i] * vh[d];
    }
  }
}

void short_conv1d_step(const float* x_ch, const uint16_t* W, float* state, float* out, int C,
                       int K) {
  for (int c = 0; c < C; ++c) {
    float* st = state + static_cast<size_t>(c) * K;
    for (int k = K - 1; k > 0; --k) st[k] = st[k - 1];
    st[0] = x_ch[c];
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      // weight [C,K]: k=0 newest matches FLA ShortConvolution
      acc += static_cast<double>(st[k]) * bf16_to_f32(W[c * K + (K - 1 - k)]);
    }
    const float y = static_cast<float>(acc);
    out[c] = y / (1.f + std::exp(-y));  // silu
  }
}

void rms_norm_gated(const float* x, const float* gate, const uint16_t* w, float* y, int n_heads,
                    int head_dim, float eps) {
  for (int h = 0; h < n_heads; ++h) {
    const float* xh = x + h * head_dim;
    const float* gh = gate + h * head_dim;
    float* yh = y + h * head_dim;
    double ss = 0.0;
    for (int d = 0; d < head_dim; ++d) ss += static_cast<double>(xh[d]) * xh[d];
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / head_dim + eps));
    for (int d = 0; d < head_dim; ++d) {
      const float ww = w ? bf16_to_f32(w[h * head_dim + d]) : 1.f;
      yh[d] = xh[d] * inv * ww * sigmoid(gh[d]);
    }
  }
}

void kda_gated_delta_step(const float* q, const float* k, const float* v, const float* g,
                          const float* beta, float* state, float* out, int n_heads, int dk, int dv) {
  for (int h = 0; h < n_heads; ++h) {
    float* S = state + static_cast<size_t>(h) * dk * dv;
    const float* qh = q + h * dk;
    const float* kh = k + h * dk;
    const float* vh = v + h * dv;
    float* oh = out + h * dv;

    float alpha = std::exp(g[h]);
    if (!std::isfinite(alpha) || alpha > 1e4f) alpha = 1e4f;
    if (alpha < 0.f) alpha = 0.f;
    const float b = beta[h];

    std::vector<float> u(dv, 0.f);
    for (int i = 0; i < dk; ++i) {
      const float ki = kh[i];
      const float* Si = S + static_cast<size_t>(i) * dv;
      for (int j = 0; j < dv; ++j) u[j] += Si[j] * ki;
    }
    for (int i = 0; i < dk; ++i) {
      float* Si = S + static_cast<size_t>(i) * dv;
      const float ki = kh[i];
      for (int j = 0; j < dv; ++j) {
        Si[j] = alpha * Si[j] + b * ki * (vh[j] - u[j]);
        if (!std::isfinite(Si[j])) Si[j] = 0.f;
      }
    }
    for (int j = 0; j < dv; ++j) oh[j] = 0.f;
    for (int i = 0; i < dk; ++i) {
      const float qi = qh[i];
      const float* Si = S + static_cast<size_t>(i) * dv;
      for (int j = 0; j < dv; ++j) oh[j] += Si[j] * qi;
    }
  }
}

void mla_absorb_q(const float* x, const uint16_t* W_qa, const uint16_t* W_qb, float* q, int H,
                  int q_lora, int n_heads, int qk_dim) {
  std::vector<float> mid(q_lora);
  gemm_bf16(x, W_qa, mid.data(), q_lora, H);
  gemm_bf16(mid.data(), W_qb, q, n_heads * qk_dim, q_lora);
}

void mla_absorb_kv(const float* x, const uint16_t* W_kva, const uint16_t* W_kvb, float* k, float* v,
                   int H, int kv_lora, int n_kv, int qk_dim, int v_dim) {
  std::vector<float> mid(kv_lora);
  gemm_bf16(x, W_kva, mid.data(), kv_lora, H);
  std::vector<float> kv(n_kv * (qk_dim + v_dim));
  gemm_bf16(mid.data(), W_kvb, kv.data(), n_kv * (qk_dim + v_dim), kv_lora);
  for (int h = 0; h < n_kv; ++h) {
    const float* row = kv.data() + h * (qk_dim + v_dim);
    std::memcpy(k + h * qk_dim, row, sizeof(float) * qk_dim);
    std::memcpy(v + h * v_dim, row + qk_dim, sizeof(float) * v_dim);
  }
}

void mhc_mix(const float* streams_in, float* streams_out, int hc_mult, int hidden,
             const uint16_t* H_mix) {
  if (hc_mult <= 1) {
    std::memcpy(streams_out, streams_in, sizeof(float) * hidden);
    return;
  }
  if (!H_mix) {
    for (int d = 0; d < hidden; ++d) {
      double s = 0.0;
      for (int c = 0; c < hc_mult; ++c) s += streams_in[c * hidden + d];
      const float avg = static_cast<float>(s / hc_mult);
      for (int c = 0; c < hc_mult; ++c) streams_out[c * hidden + d] = avg;
    }
    return;
  }
  for (int o = 0; o < hc_mult; ++o) {
    for (int d = 0; d < hidden; ++d) {
      double acc = 0.0;
      for (int i = 0; i < hc_mult; ++i) {
        acc += bf16_to_f32(H_mix[o * hc_mult + i]) * streams_in[i * hidden + d];
      }
      streams_out[o * hidden + d] = static_cast<float>(acc);
    }
  }
}

int indexer_topk(const float* q_idx, const float* k_idx_cache, int* out_idx, int topk, int pos,
                 int max_seq, int d_idx) {
  const int seq = pos + 1;
  if (seq <= 0) return 0;
  topk = std::min(topk, seq);
  if (topk <= 0) topk = seq;
  std::vector<std::pair<float, int>> scores;
  scores.reserve(static_cast<size_t>(seq));
  for (int t = 0; t < seq && t < max_seq; ++t) {
    const float* kt = k_idx_cache + static_cast<size_t>(t) * d_idx;
    float dot = 0.f;
    for (int d = 0; d < d_idx; ++d) dot += q_idx[d] * kt[d];
    scores.emplace_back(dot, t);
  }
  if (static_cast<int>(scores.size()) <= topk) {
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) out_idx[i] = scores[i].second;
    std::sort(out_idx, out_idx + scores.size());
    return static_cast<int>(scores.size());
  }
  std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  bool has_pos = false;
  for (int i = 0; i < topk; ++i) {
    out_idx[i] = scores[i].second;
    if (out_idx[i] == pos) has_pos = true;
  }
  if (!has_pos && topk > 0) out_idx[topk - 1] = pos;
  std::sort(out_idx, out_idx + topk);
  return topk;
}

int kpool_select(const float* q_idx, const float* k_idx_cache, int* out_idx, int topk_pools,
                 int pos, int max_seq, int d_idx, int pool_size) {
  if (pool_size <= 1) return indexer_topk(q_idx, k_idx_cache, out_idx, topk_pools, pos, max_seq, d_idx);
  const int seq = pos + 1;
  if (seq <= 0) return 0;
  const int n_pools = (seq + pool_size - 1) / pool_size;
  std::vector<float> pool_keys(static_cast<size_t>(n_pools) * d_idx, 0.f);
  for (int p = 0; p < n_pools; ++p) {
    const int t0 = p * pool_size;
    const int t1 = std::min(seq, t0 + pool_size);
    const int n = t1 - t0;
    for (int t = t0; t < t1; ++t) {
      const float* kt = k_idx_cache + static_cast<size_t>(t) * d_idx;
      for (int d = 0; d < d_idx; ++d) pool_keys[static_cast<size_t>(p) * d_idx + d] += kt[d];
    }
    for (int d = 0; d < d_idx; ++d)
      pool_keys[static_cast<size_t>(p) * d_idx + d] /= static_cast<float>(n);
  }
  std::vector<std::pair<float, int>> scores;
  scores.reserve(static_cast<size_t>(n_pools));
  for (int p = 0; p < n_pools; ++p) {
    const float* pk = pool_keys.data() + static_cast<size_t>(p) * d_idx;
    float dot = 0.f;
    for (int d = 0; d < d_idx; ++d) dot += q_idx[d] * pk[d];
    scores.emplace_back(dot, p);
  }
  int keep = std::min(topk_pools, n_pools);
  if (keep <= 0) keep = n_pools;
  std::partial_sort(scores.begin(), scores.begin() + keep, scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<int> tokens;
  tokens.reserve(static_cast<size_t>(keep) * pool_size + 1);
  for (int i = 0; i < keep; ++i) {
    const int p = scores[i].second;
    const int t0 = p * pool_size;
    const int t1 = std::min(seq, t0 + pool_size);
    for (int t = t0; t < t1; ++t) tokens.push_back(t);
  }
  tokens.push_back(pos);
  std::sort(tokens.begin(), tokens.end());
  tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
  const int n = static_cast<int>(tokens.size());
  for (int i = 0; i < n; ++i) out_idx[i] = tokens[i];
  return n;
}

}  // namespace llmoc::glm::ops
