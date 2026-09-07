#pragma once
// llm-on-cpu :: moe/qlwc_expert_bytes.h
// INT4 (AWQ/GPTQ) per-expert byte geometry for QLWC MoE host banks / VRAM slots.

#include <cstddef>
#include <cstdint>

namespace llmoc::moe {

inline size_t int4_row_bytes(int K) {
  return (static_cast<size_t>(K) + 1) / 2;
}

inline size_t int4_q_bytes(int M, int K) {
  return static_cast<size_t>(M) * int4_row_bytes(K);
}

inline int int4_ngroups(int K, int group_size) {
  const int gs = group_size > 0 ? group_size : 128;
  return (K + gs - 1) / gs;
}

inline size_t int4_scales_f16_bytes(int M, int K, int group_size) {
  return static_cast<size_t>(M) * static_cast<size_t>(int4_ngroups(K, group_size)) * sizeof(uint16_t);
}

inline size_t int4_zeros_f16_bytes(int M, int K, int group_size, bool has_zeros) {
  return has_zeros ? int4_scales_f16_bytes(M, K, group_size) : 0;
}

inline size_t int4_scales_f32_bytes(int M, int K, int group_size) {
  return static_cast<size_t>(M) * static_cast<size_t>(int4_ngroups(K, group_size)) * sizeof(float);
}

// One projection [M,K]: packed q + fp16 scales + optional zeros.
// Host f32 scales are NOT stored — gemm_int4 converts on demand for CPU overflow.
inline size_t int4_proj_storage_bytes(int M, int K, int group_size, bool has_zeros) {
  return int4_q_bytes(M, K) + int4_scales_f16_bytes(M, K, group_size) +
         int4_zeros_f16_bytes(M, K, group_size, has_zeros);
}

// gate[IxH] + up[IxH] + down[HxI]
inline size_t expert_storage_bytes(int H, int I, int group_size, bool has_zeros) {
  return int4_proj_storage_bytes(I, H, group_size, has_zeros) +
         int4_proj_storage_bytes(I, H, group_size, has_zeros) +
         int4_proj_storage_bytes(H, I, group_size, has_zeros);
}

// Device-resident footprint (no f32 scales scratch) — matches ensure_int4_resident.
inline size_t int4_proj_device_bytes(int M, int K, int group_size, bool has_zeros) {
  return int4_q_bytes(M, K) + int4_scales_f16_bytes(M, K, group_size) +
         int4_zeros_f16_bytes(M, K, group_size, has_zeros);
}

inline size_t expert_device_bytes(int H, int I, int group_size, bool has_zeros) {
  return int4_proj_device_bytes(I, H, group_size, has_zeros) +
         int4_proj_device_bytes(I, H, group_size, has_zeros) +
         int4_proj_device_bytes(H, I, group_size, has_zeros);
}

}  // namespace llmoc::moe
