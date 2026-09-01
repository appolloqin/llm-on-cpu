#pragma once
// llm-on-cpu :: hal/cuda_backend.h
// M5: dynamic cudart + cuBLAS. Inactive unless enable() — pure_cpu untouched.

#include <cstddef>
#include <cstdint>
#include <string>

#include "weights/qlwc_store.h"
#include "hal/quant_views.h"

namespace llmoc::hal::cuda {

bool probe_available();  // load DLLs + device count; does not enable gemm path
bool enabled();
const char* status();
int device_count();  // 0 if not probed/unavailable

// Enable GPU GEMM acceleration. vram_budget_bytes caps uploaded weight cache.
// Returns false if CUDA unavailable (caller may degrade or abort).
bool enable(size_t vram_budget_bytes);
void disable();

size_t vram_used();
size_t vram_budget();

// y[M] = W[M,K] @ x[K]. W pointer is cache key (stable WeightManager buffers).
// Returns false → caller must use CPU gemm. No-op false when !enabled().
bool try_gemm_w16(const float* x, const uint16_t* W, float* y, int M, int K, bool is_f16);
bool prefetch_w16(const uint16_t* W, int M, int K, bool is_f16);

// INT4 → host FP32 dequant once → VRAM cache → cublasSgemm. Key = W.qweight.
// Returns false → caller must use CPU gemm_int4. Skips vocab-scale M (lm_head).
bool try_gemm_int4(const float* x, const qlwc::Int4View& W, float* y);
bool try_gemm_int4_batch(const float* X, int n, const qlwc::Int4View& W, float* Y);
// Upload only (warm); false if !enabled / OOM / budget.
bool prefetch_int4_weight(const qlwc::Int4View& W);

// AWQ / NVFP4 (shared views) — same dequant→FP32→SGEMM path.
bool try_gemm_awq(const float* x, const AwqView& W, float* y);
bool prefetch_awq_weight(const AwqView& W);
bool try_gemm_nvfp4(const float* x, const Nvfp4View& W, float* y);
bool prefetch_nvfp4_weight(const Nvfp4View& W);

// Raw device buffer helpers for expert-slot residency (H2D).
void* device_alloc(size_t bytes);
void device_free(void* p);
bool h2d(void* dst, const void* src, size_t bytes);
bool d2h(void* dst, const void* src, size_t bytes);

void log_status();

}  // namespace llmoc::hal::cuda
