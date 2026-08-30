#pragma once
// llm-on-cpu :: hal/cuda_backend.h
// M5: dynamic cudart + cuBLAS. Inactive unless enable() — pure_cpu untouched.

#include <cstddef>
#include <cstdint>
#include <string>

namespace llmoc::hal::cuda {

bool probe_available();  // load DLLs + device count; does not enable gemm path
bool enabled();
const char* status();

// Enable GPU GEMM acceleration. vram_budget_bytes caps uploaded weight cache.
// Returns false if CUDA unavailable (caller may degrade or abort).
bool enable(size_t vram_budget_bytes);
void disable();

size_t vram_used();
size_t vram_budget();

// y[M] = W[M,K] @ x[K]. W pointer is cache key (stable WeightManager buffers).
// Returns false → caller must use CPU gemm. No-op false when !enabled().
bool try_gemm_w16(const float* x, const uint16_t* W, float* y, int M, int K, bool is_f16);

void log_status();

}  // namespace llmoc::hal::cuda
