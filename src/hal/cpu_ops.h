#pragma once
// llm-on-cpu :: hal/cpu_ops.h
// CPU 参考算子(正确优先)。BF16 权重 / 激活, 内部升 F32 计算。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hal/device.h"

namespace llmoc::hal {

inline float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1;
  const uint32_t exp = (h >> 10) & 0x1F;
  const uint32_t man = h & 0x3FF;
  uint32_t out;
  if (exp == 0) {
    if (man == 0)
      out = sign << 31;
    else {
      // subnormal
      float f = std::ldexp(static_cast<float>(man), -24);
      if (sign) f = -f;
      return f;
    }
  } else if (exp == 31) {
    out = (sign << 31) | 0x7F800000u | (man << 13);
  } else {
    out = (sign << 31) | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}
inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

enum class WDtype : uint8_t { kBF16 = 1, kF16 = 2, kF32 = 3 };

inline float load_w(const uint16_t* p, WDtype dt) {
  return dt == WDtype::kF16 ? f16_to_f32(*p) : bf16_to_f32(*p);
}

void bf16_to_f32_buf(const uint16_t* src, float* dst, size_t n);
void f32_to_bf16_buf(const float* src, uint16_t* dst, size_t n);

// y[M] = x[K] @ W[M,K]^T  (PyTorch Linear); W 为 BF16 或 F16
void gemm_bias_free(const float* x, const uint16_t* W, float* y, int M, int K,
                    WDtype dt = WDtype::kBF16);

// Qwen3.5: scale = (1 + w)（权重零初始化、以 1 为中心）；Llama/Qwen2/MoE: scale = w
void rmsnorm(const float* x, const uint16_t* w, float* y, int n, float eps,
             WDtype dt = WDtype::kBF16, bool one_plus_weight = false);
void rmsnorm_gated(const float* x, const float* gate, const uint16_t* w, float* y, int n, float eps,
                   WDtype dt = WDtype::kBF16);

void silu_and_mul(const float* gate, const float* up, float* out, int n);
void softmax_inplace(float* x, int n);

// partial RoPE: rotate first rotary_dim dims (must be even)
void apply_rope_inplace(float* q_or_k, int head_dim, int rotary_dim, float cos, float sin,
                        int pos /* unused if cos/sin precomputed per-dim */);
void apply_rope_freqs(float* q_or_k, int head_dim, int rotary_dim, int pos, float theta);
void apply_mrope_freqs(float* q_or_k, int head_dim, int rotary_dim, int pos_t, int pos_h, int pos_w,
                       float theta, const int section[3], bool interleaved);

// GQA causal attention for one query position against KV cache
// q: [n_heads, head_dim]
// k/v cache: [n_kv_heads, cache_cap, head_dim] （cache_cap=分配长度，通常 max_seq；勿用当前 seq_len 当地步）
void attn_decode_one(const float* q, const float* k_cache, const float* v_cache, float* out,
                     int n_heads, int n_kv_heads, int head_dim, int seq_len, int cache_cap,
                     float scale);

// Prefill causal attention: q/k/v [seq, n_heads/n_kv, head_dim] -> out [seq, n_heads, head_dim]
void attn_prefill(const float* q, const float* k, const float* v, float* out, int seq, int n_heads,
                  int n_kv_heads, int head_dim, float scale);

// Gated DeltaNet recurrent step (seq tokens). State: [n_v_heads, dk, dv]
void gated_delta_recurrent(const float* q, const float* k, const float* v, const float* g,
                           const float* beta, float* state, float* out, int seq, int n_heads,
                           int dk, int dv, bool qk_l2norm);

class CpuOps final : public Ops {
 public:
  Backend backend() const override { return Backend::kCpuAvx2; }
};

}  // namespace llmoc::hal
