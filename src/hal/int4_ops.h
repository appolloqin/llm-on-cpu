#pragma once
// llm-on-cpu :: hal/int4_ops.h
// INT4 解量化 GEMM（GPTQ 非对称 / AWQ 对称），独立新算子。

#include "hal/cpu_ops.h"
#include "weights/qlwc_store.h"

namespace llmoc::hal {

// y[M] = x[K] @ W[M,K]^T ，W 为 INT4 打包
void gemm_int4(const float* x, const qlwc::Int4View& W, float* y);

struct Int4ArgmaxResult {
  int32_t index = 0;
  float value = -1e30f;
};

// greedy decode：求 argmax(x @ W^T)，不写全词表 logits（省 DRAM 写带宽）
Int4ArgmaxResult gemm_int4_argmax(const float* x, const qlwc::Int4View& W);

// Y[n, M] = X[n, K] @ W[M,K]^T ；权重行只扫一遍，供 prefill 批处理
void gemm_int4_batch(const float* X, int n, const qlwc::Int4View& W, float* Y);

// 解量化单行：out[K] ← W[row, :]（embedding 查表）
void dequant_int4_row(const qlwc::Int4View& W, int row, float* out);

// 解量化整矩阵：out[M*K] row-major，供 GPU FP32 缓存（与 cublas W[M,K] 约定一致）
void dequant_int4_matrix(const qlwc::Int4View& W, float* out);

}  // namespace llmoc::hal
