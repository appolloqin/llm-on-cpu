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
// Fused multi-GEMV: 2-4 weight views sharing same x/K/ng/gs. One launch, one H2D.
// Returns false → caller falls back to sequential try_gemm_int4 calls.
bool try_gemm_int4_multi(const float* x, const qlwc::Int4View* const* Ws, float* const* ys, int n);
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

// ---- NVRTC JIT (no nvcc): 运行时编译 CUDA kernel, driver API 启动 ----
// 动态加载 nvrtc64_XXX.dll + nvcuda.dll; 任一缺失 → jit_available()=false, 优雅降级。
bool jit_available();
// 编译 cuda_src 中名为 kernel_name 的 __global__ kernel; 成功返回 true 并填充 out_fn(供 jit_launch)。
// 失败返回 false, g_status 记录原因。函数句柄缓存到 disable() 自动卸载。
bool jit_compile(const char* cuda_src, const char* kernel_name, void** out_fn);
// 启动已编译 kernel; params 为内核参数地址数组(void*[])。grid/block 为线程组织。
bool jit_launch(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                unsigned bz, unsigned shmem_bytes, void** params);

// GPU 原生 INT4 dequant-GEMV(kernel 内按 group 反量化, 权重保持量化形态驻留 VRAM)。
// 所有指针均为设备指针。is_awq=true: w=(q-7)*scale; false: w=q*scale+zero(zeros 不可为空)。
// 返回 false 表示 JIT 不可用, 调用方回退 CPU。
bool jit_gemv_int4(const uint8_t* d_qweight, const uint16_t* d_scales, const uint16_t* d_zeros,
                   const float* d_x, float* d_y, int M, int K, int ng, int gs, bool is_awq);

void log_status();

}  // namespace llmoc::hal::cuda
