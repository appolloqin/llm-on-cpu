#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace llmoc::glm::ops {

inline float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

inline float softplus(float x) {
  if (x > 20.f) return x;
  if (x < -20.f) return std::exp(x);
  return std::log1p(std::exp(x));
}

void rmsnorm(const float* x, const uint16_t* w, float* y, int n, float eps);
void gemm_bf16(const float* x, const uint16_t* W, float* y, int M, int K);
void silu_mul(const float* gate, const float* up, float* out, int n);
void softmax(float* x, int n);
void rope_inplace(float* qk, int head_dim, int pos, float theta);
void gqa_attn(const float* q /*[nh*hd]*/, const float* k_cache /*[nkv*max_seq*hd]*/,
              const float* v_cache, float* out /*[nh*hd]*/, int n_heads, int n_kv, int head_dim,
              int max_seq, int pos);

// Attend only over idx[0..n_idx) (must be sorted ascending, in [0,pos]).
void gqa_attn_indexed(const float* q, const float* k_cache, const float* v_cache, float* out,
                      int n_heads, int n_kv, int head_dim, int max_seq, const int* idx, int n_idx);

// Depthwise causal conv1d, one step. x_ch[C], W[C*K] layout [c,k] k=0 newest weight,
// state[C*K] ring (index 0 = current). out[C] after silu.
void short_conv1d_step(const float* x_ch, const uint16_t* W, float* state, float* out, int C,
                       int K);

// Per-head gated RMSNorm: y = rms(x) * w * sigmoid(gate); x/gate/out [nh*hd]
void rms_norm_gated(const float* x, const float* gate, const uint16_t* w, float* y, int n_heads,
                    int head_dim, float eps);

// GLM KDA gated-delta step.
void kda_gated_delta_step(const float* q, const float* k, const float* v, const float* g,
                          const float* beta, float* state, float* out, int n_heads, int dk, int dv);

void mla_absorb_q(const float* x, const uint16_t* W_qa, const uint16_t* W_qb, float* q, int H,
                  int q_lora, int n_heads, int qk_dim);
void mla_absorb_kv(const float* x, const uint16_t* W_kva, const uint16_t* W_kvb, float* k, float* v,
                   int H, int kv_lora, int n_kv, int qk_dim, int v_dim);

void mhc_mix(const float* streams_in /*[hc*H]*/, float* streams_out /*[hc*H]*/, int hc_mult,
             int hidden, const uint16_t* H_mix /*[hc*hc] or null*/);

// Indexer: score past positions with q_idx · k_idx[t]; write up to topk indices (always include pos).
// k_idx_cache layout [max_seq * d_idx]. Returns count written to out_idx (sorted).
int indexer_topk(const float* q_idx, const float* k_idx_cache, int* out_idx, int topk, int pos,
                 int max_seq, int d_idx);

// KPool: compress every pool_size tokens (mean), select topk pools, expand to token indices.
// always includes `pos`. Returns count written to out_idx (sorted, unique).
int kpool_select(const float* q_idx, const float* k_idx_cache, int* out_idx, int topk_pools,
                 int pos, int max_seq, int d_idx, int pool_size);

}  // namespace llmoc::glm::ops
