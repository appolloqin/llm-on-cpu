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

// Auto resident-GPU path: if free VRAM >= workspace_bytes (or LLMOC_RESIDENT_GPU=1),
// reserve headroom and enable GPU GDN / tighter decode residency. LLMOC_RESIDENT_GPU=0 forces off.
bool try_enable_resident_gpu(size_t workspace_bytes);
bool resident_gpu_enabled();
void log_resident_stats();

// ---- Decode residual stream (device); additive — CPU path unchanged when unused ----
bool decode_act_begin(const float* h_host, int H);
bool decode_act_sync_to_host(float* h_host, int H);
bool decode_act_load_from_host(const float* h_host, int H);
float* decode_act_ptr();
bool decode_act_valid();
void decode_act_invalidate();

// From device residual: rmsnorm(ln1) + 2..4 INT4 GEMVs; only projection outputs D2H.
bool try_rmsnorm_gemm_multi_from_act(const uint16_t* ln1, const qlwc::Int4View* const* Ws,
                                     float* const* ys, int n, float eps, bool ln_is_f16);

// After host conv/GDN/rmsnorm_gated: H2D(core) + wout + residual add + MLP on device act.
// wout: INT4 if wout_i4 non-null, else BF16/F16 pass-through.
bool try_ffn_on_act(const float* host_core, int core_dim, const qlwc::Int4View* wout_i4,
                    const uint16_t* wout_pass, bool wout_is_f16, const uint16_t* ln2,
                    const qlwc::Int4View& wgate, const qlwc::Int4View& wup,
                    const qlwc::Int4View& wdown, int I, float eps, bool ln_is_f16);

// Final RMSNorm + lm_head; D2H logits only (hidden stays on device until invalidate).
bool try_lm_head_w16_from_act(const uint16_t* final_norm, const uint16_t* lm_pass, int V,
                              float* logits_host, float eps, bool ln_is_f16, bool lm_is_f16);
bool try_lm_head_int4_from_act(const uint16_t* final_norm, const qlwc::Int4View& lm,
                               float* logits_host, float eps, bool ln_is_f16);

// Legacy host↔device helpers (still used by non-stream paths / fallbacks).
bool try_mlp_decode_resident(const float* x, const uint16_t* ln2, const qlwc::Int4View& wgate,
                             const qlwc::Int4View& wup, const qlwc::Int4View& wdown, float* y, int H,
                             int I, float eps, bool ln_is_f16);
bool try_rmsnorm_gemm_multi_resident(const float* x, const uint16_t* ln1,
                                     const qlwc::Int4View* const* Ws, float* const* ys, int n, int H,
                                     float eps, bool ln_is_f16);
bool try_out_mlp_resident(const float* residual, const float* core, const qlwc::Int4View& wout,
                          const uint16_t* ln2, const qlwc::Int4View& wgate,
                          const qlwc::Int4View& wup, const qlwc::Int4View& wdown, float* y, int H,
                          int I, float eps, bool ln_is_f16);

// y[M] = W[M,K] @ x[K]. W pointer is cache key (stable WeightManager buffers).
bool try_gemm_w16(const float* x, const uint16_t* W, float* y, int M, int K, bool is_f16);
bool try_gemm_w16_batch(const float* X, int n, const uint16_t* W, float* Y, int M, int K,
                        bool is_f16);
bool prefetch_w16(const uint16_t* W, int M, int K, bool is_f16);

bool try_gemm_int4(const float* x, const qlwc::Int4View& W, float* y);
bool try_gemm_int4_batch(const float* X, int n, const qlwc::Int4View& W, float* Y);
bool try_gemm_int4_multi(const float* x, const qlwc::Int4View* const* Ws, float* const* ys, int n);
bool prefetch_int4_weight(const qlwc::Int4View& W);

bool try_gemm_awq(const float* x, const AwqView& W, float* y);
bool prefetch_awq_weight(const AwqView& W);
bool try_gemm_nvfp4(const float* x, const Nvfp4View& W, float* y);
bool prefetch_nvfp4_weight(const Nvfp4View& W);

void* device_alloc(size_t bytes);
void device_free(void* p);
bool h2d(void* dst, const void* src, size_t bytes);
bool d2h(void* dst, const void* src, size_t bytes);

bool try_attn_prefill(const float* q, const float* k, const float* v, float* out, int seq,
                      int n_heads, int n_kv_heads, int head_dim, float scale);

bool try_gated_delta_gpu(const float* q, const float* k, const float* v, const float* g,
                         const float* beta, float* state, float* out, int n_heads, int dk, int dv);
void flush_gdn_state_to_host(float* host_state, int n_heads, int dk, int dv);

bool jit_available();
bool jit_compile(const char* cuda_src, const char* kernel_name, void** out_fn);
bool jit_launch(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                unsigned bz, unsigned shmem_bytes, void** params);
bool jit_gemv_int4(const uint8_t* d_qweight, const uint16_t* d_scales, const uint16_t* d_zeros,
                   const float* d_x, float* d_y, int M, int K, int ng, int gs, bool is_awq);

void log_status();

}  // namespace llmoc::hal::cuda
