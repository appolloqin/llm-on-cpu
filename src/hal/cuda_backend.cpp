// llm-on-cpu :: hal/cuda_backend.cpp — M5 dynamic CUDA (no nvcc)
#include "hal/cuda_backend.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "hal/int4_ops.h"
#include "hal/quant_views.h"
#include "weights/qlwc_store.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace llmoc::hal::cuda {
namespace {

constexpr int kCudaSuccess = 0;
constexpr int kCublasSuccess = 0;
constexpr int kCudaMemcpyH2D = 1;
constexpr int kCudaMemcpyD2H = 2;
constexpr int kCublasOpN = 0;
constexpr int kCublasOpT = 1;

using cudaMalloc_t = int (*)(void**, size_t);
using cudaFree_t = int (*)(void*);
using cudaMemcpy_t = int (*)(void*, const void*, size_t, int);
using cudaGetDeviceCount_t = int (*)(int*);
using cudaSetDevice_t = int (*)(int);
using cudaGetDeviceProperties_t = int (*)(void*, int);
using cublasCreate_t = int (*)(void**);
using cublasDestroy_t = int (*)(void*);
using cublasSgemm_t = int (*)(void*, int, int, int, int, int, const float*, const float*, int,
                              const float*, int, const float*, float*, int);

// ---- driver API (nvcuda.dll) for JIT kernel launch ----
using cuCtxGetCurrent_t = int (*)(void**);
using cuDevicePrimaryCtxRetain_t = int (*)(void**, int);
using cuCtxSetCurrent_t = int (*)(void*);
using cuDeviceGetAttribute_t = int (*)(int*, int, int);
using cuModuleLoadData_t = int (*)(void**, const void*);
using cuModuleGetFunction_t = int (*)(void**, void*, const char*);
using cuModuleUnload_t = int (*)(void*);
using cuLaunchKernel_t = int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned,
                                 unsigned, unsigned, void*, void**, void**);

// ---- NVRTC (nvrtc64_XXX.dll) for runtime compilation ----
using nvrtcCreateProgram_t = int (*)(void**, const char*, const char*, int, const char* const*,
                                     const char* const*);
using nvrtcCompileProgram_t = int (*)(void*, int, const char* const*);
using nvrtcGetPTXSize_t = int (*)(void*, size_t*);
using nvrtcGetPTX_t = int (*)(void*, char*);
using nvrtcGetProgramLogSize_t = int (*)(void*, size_t*);
using nvrtcGetProgramLog_t = int (*)(void*, char*);
using nvrtcDestroyProgram_t = int (*)(void**);
using nvrtcGetErrorString_t = const char* (*)(int);

struct Api {
  void* cudart = nullptr;
  void* cublas = nullptr;
  cudaMalloc_t cudaMalloc = nullptr;
  cudaFree_t cudaFree = nullptr;
  cudaMemcpy_t cudaMemcpy = nullptr;
  cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
  cudaSetDevice_t cudaSetDevice = nullptr;
  cudaGetDeviceProperties_t cudaGetDeviceProperties = nullptr;
  cublasCreate_t cublasCreate = nullptr;
  cublasDestroy_t cublasDestroy = nullptr;
  cublasSgemm_t cublasSgemm = nullptr;

  void* nvcuda = nullptr;
  cuCtxGetCurrent_t cuCtxGetCurrent = nullptr;
  cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
  cuDeviceGetAttribute_t cuDeviceGetAttribute = nullptr;
  cuModuleLoadData_t cuModuleLoadData = nullptr;
  cuModuleGetFunction_t cuModuleGetFunction = nullptr;
  cuModuleUnload_t cuModuleUnload = nullptr;
  cuLaunchKernel_t cuLaunchKernel = nullptr;

  void* nvrtc = nullptr;
  nvrtcCreateProgram_t nvrtcCreateProgram = nullptr;
  nvrtcCompileProgram_t nvrtcCompileProgram = nullptr;
  nvrtcGetPTXSize_t nvrtcGetPTXSize = nullptr;
  nvrtcGetPTX_t nvrtcGetPTX = nullptr;
  nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize = nullptr;
  nvrtcGetProgramLog_t nvrtcGetProgramLog = nullptr;
  nvrtcDestroyProgram_t nvrtcDestroyProgram = nullptr;
  nvrtcGetErrorString_t nvrtcGetErrorString = nullptr;
};

struct CacheEntry {
  void* d_W = nullptr;
  int M = 0;
  int K = 0;
  size_t bytes = 0;
};

// INT4 量化形态驻留条目: 权重按 packed uint8 + fp16 scales/zeros 上传, GEMV 时 kernel 内反量化。
// bytes 计入 VRAM 用量。预算按量化后字节数, 比 FP32(M*K*4) 省 8x。
struct Int4Resident {
  void* d_qweight = nullptr;
  void* d_scales = nullptr;
  void* d_zeros = nullptr;
  int M = 0, K = 0, ng = 0, gs = 0;
  bool is_awq = true;
  size_t bytes = 0;
  uint64_t last_use = 0;
};

Api g_api;
std::mutex g_mu;
bool g_probed = false;
bool g_probe_ok = false;
bool g_enabled = false;
std::string g_status = "off";
size_t g_budget = 0;
size_t g_used = 0;
void* g_cublas = nullptr;
void* g_dx = nullptr;
void* g_dy = nullptr;
int g_cap_k = 0;
int g_cap_m = 0;
int g_cap_n = 0;  // batch columns for X/Y
std::unordered_map<const void*, CacheEntry> g_cache;
std::unordered_map<const void*, Int4Resident> g_int4_cache;
uint64_t g_lru_tick = 0;
std::unordered_map<void*, void*> g_jit_modules;  // CUfunction -> CUmodule (JIT 句柄, disable 时卸载)
std::unordered_map<std::string, void*> g_jit_kernels;  // kernel 名→CUfunction 缓存(disable 时清空)

// GDN device state: host state ptr → device state ptr (persists between decode steps)
std::unordered_map<const float*, float*> g_gdn_state;
float* g_gdn_buf = nullptr;   // device scratch for q/k/v/g/beta/out uploads
size_t g_gdn_buf_cap = 0;

// Prefill attention scratch (host↔device per call; only used when cuda enabled)
void* g_attn_q = nullptr;
void* g_attn_k = nullptr;
void* g_attn_v = nullptr;
void* g_attn_o = nullptr;
size_t g_attn_q_bytes = 0;
size_t g_attn_k_bytes = 0;
size_t g_attn_v_bytes = 0;
size_t g_attn_o_bytes = 0;

// 累计性能采样: 用于诊断 GEMV 路径瓶颈。disable 时清零。
double g_prof_h2d_us = 0.0;
double g_prof_kernel_us = 0.0;
double g_prof_d2h_us = 0.0;
uint64_t g_prof_calls = 0;

// lm_head / vocab 大张量也允许上 GPU(VRAM 受 budget 约束); 大于 256K 行仍跳过以免单次分配过大。
constexpr int kMaxGpuInt4Rows = 262144;

#if defined(_WIN32)
void* load_lib(const char* n) { return reinterpret_cast<void*>(LoadLibraryA(n)); }
void* load_lib_path(const std::string& p) { return reinterpret_cast<void*>(LoadLibraryA(p.c_str())); }
void* sym(void* h, const char* n) {
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(h), n));
}
#else
void* load_lib(const char* n) { return dlopen(n, RTLD_NOW); }
void* load_lib_path(const std::string& p) { return dlopen(p.c_str(), RTLD_NOW); }
void* sym(void* h, const char* n) { return dlsym(h, n); }
#endif

float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float f16_to_f32(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  if (exp == 0) {
    if (man == 0) return sign ? -0.f : 0.f;
    float f = std::ldexp(static_cast<float>(man), -24);
    return sign ? -f : f;
  }
  if (exp == 31) {
    uint32_t out = (sign << 31) | 0x7F800000u | (man << 13);
    float f;
    std::memcpy(&f, &out, 4);
    return f;
  }
  uint32_t out = (sign << 31) | ((exp + (127 - 15)) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

bool load_apis(std::string& err) {
  if (g_api.cudaMalloc) return true;
  std::vector<std::string> dirs;  // Windows: toolkit bin 搜索路径；供 cudart/nvrtc 共用
#if defined(_WIN32)
  // CUDA 13: cudart64_13 / cublas64_13；官方布局常在 bin\x64（仅 bin 在 PATH 时找不到）
  const char* cudart_names[] = {"cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", nullptr};
  const char* cublas_names[] = {"cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll", nullptr};
  auto add_dir = [&](std::string d) {
    if (d.empty()) return;
    if (d.back() != '\\' && d.back() != '/') d.push_back('\\');
    dirs.push_back(std::move(d));
  };
  auto add_toolkit_root = [&](const std::string& root) {
    if (root.empty()) return;
    add_dir(root + "\\bin\\x64");
    add_dir(root + "\\bin");
  };
  for (const char* ev : {"CUDA_PATH", "CUDA_PATH_V13_3", "CUDA_PATH_V13_2", "CUDA_PATH_V13_1",
                         "CUDA_PATH_V13_0", "CUDA_PATH_V12_6", "CUDA_PATH_V12_5", "CUDA_HOME"}) {
    const char* v = std::getenv(ev);
    if (v && v[0]) add_toolkit_root(v);
  }
  static const char* kRoots[] = {
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
      nullptr};
  for (int i = 0; kRoots[i]; ++i) add_toolkit_root(kRoots[i]);
  dirs.emplace_back("");  // PATH / 默认搜索

  for (const auto& dir : dirs) {
    for (int i = 0; cudart_names[i] && !g_api.cudart; ++i) {
      g_api.cudart = dir.empty() ? load_lib(cudart_names[i]) : load_lib_path(dir + cudart_names[i]);
    }
    for (int i = 0; cublas_names[i] && !g_api.cublas; ++i) {
      g_api.cublas = dir.empty() ? load_lib(cublas_names[i]) : load_lib_path(dir + cublas_names[i]);
    }
    if (g_api.cudart && g_api.cublas) break;
  }
#else
  g_api.cudart = load_lib("libcudart.so.13");
  if (!g_api.cudart) g_api.cudart = load_lib("libcudart.so.12");
  if (!g_api.cudart) g_api.cudart = load_lib("libcudart.so");
  g_api.cublas = load_lib("libcublas.so.13");
  if (!g_api.cublas) g_api.cublas = load_lib("libcublas.so.12");
  if (!g_api.cublas) g_api.cublas = load_lib("libcublas.so");
#endif
  if (!g_api.cudart || !g_api.cublas) {
    err = "cudart/cublas not found (CUDA 12/13; on Win CUDA13 often under bin\\x64)";
    return false;
  }
  g_api.cudaMalloc = reinterpret_cast<cudaMalloc_t>(sym(g_api.cudart, "cudaMalloc"));
  g_api.cudaFree = reinterpret_cast<cudaFree_t>(sym(g_api.cudart, "cudaFree"));
  g_api.cudaMemcpy = reinterpret_cast<cudaMemcpy_t>(sym(g_api.cudart, "cudaMemcpy"));
  g_api.cudaGetDeviceCount =
      reinterpret_cast<cudaGetDeviceCount_t>(sym(g_api.cudart, "cudaGetDeviceCount"));
  g_api.cudaSetDevice = reinterpret_cast<cudaSetDevice_t>(sym(g_api.cudart, "cudaSetDevice"));
  g_api.cublasCreate = reinterpret_cast<cublasCreate_t>(sym(g_api.cublas, "cublasCreate_v2"));
  if (!g_api.cublasCreate)
    g_api.cublasCreate = reinterpret_cast<cublasCreate_t>(sym(g_api.cublas, "cublasCreate"));
  g_api.cublasDestroy = reinterpret_cast<cublasDestroy_t>(sym(g_api.cublas, "cublasDestroy_v2"));
  if (!g_api.cublasDestroy)
    g_api.cublasDestroy = reinterpret_cast<cublasDestroy_t>(sym(g_api.cublas, "cublasDestroy"));
  g_api.cublasSgemm = reinterpret_cast<cublasSgemm_t>(sym(g_api.cublas, "cublasSgemm_v2"));
  if (!g_api.cublasSgemm)
    g_api.cublasSgemm = reinterpret_cast<cublasSgemm_t>(sym(g_api.cublas, "cublasSgemm"));
  if (!g_api.cudaMalloc || !g_api.cudaMemcpy || !g_api.cudaGetDeviceCount || !g_api.cublasCreate ||
      !g_api.cublasSgemm) {
    err = "missing CUDA symbols";
    return false;
  }
  g_api.cudaGetDeviceProperties =
      reinterpret_cast<cudaGetDeviceProperties_t>(sym(g_api.cudart, "cudaGetDeviceProperties"));

  // ---- 可选: driver API (nvcuda.dll) + NVRTC, 用于运行时 JIT kernel (no nvcc) ----
  // 加载失败不致命: jit_available()=false, 走 cublas/CPU 回退。
#if defined(_WIN32)
  g_api.nvcuda = load_lib("nvcuda.dll");
  if (g_api.nvcuda) {
    g_api.cuCtxGetCurrent = reinterpret_cast<cuCtxGetCurrent_t>(sym(g_api.nvcuda, "cuCtxGetCurrent"));
    g_api.cuDevicePrimaryCtxRetain =
        reinterpret_cast<cuDevicePrimaryCtxRetain_t>(sym(g_api.nvcuda, "cuDevicePrimaryCtxRetain"));
    g_api.cuCtxSetCurrent = reinterpret_cast<cuCtxSetCurrent_t>(sym(g_api.nvcuda, "cuCtxSetCurrent"));
    g_api.cuDeviceGetAttribute =
        reinterpret_cast<cuDeviceGetAttribute_t>(sym(g_api.nvcuda, "cuDeviceGetAttribute"));
    g_api.cuModuleLoadData =
        reinterpret_cast<cuModuleLoadData_t>(sym(g_api.nvcuda, "cuModuleLoadData"));
    g_api.cuModuleGetFunction =
        reinterpret_cast<cuModuleGetFunction_t>(sym(g_api.nvcuda, "cuModuleGetFunction"));
    g_api.cuModuleUnload = reinterpret_cast<cuModuleUnload_t>(sym(g_api.nvcuda, "cuModuleUnload"));
    g_api.cuLaunchKernel = reinterpret_cast<cuLaunchKernel_t>(sym(g_api.nvcuda, "cuLaunchKernel"));
  }
  {
    const char* nvrtc_names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_110_0.dll",
                                 nullptr};
    for (const auto& dir : dirs) {
      for (int i = 0; nvrtc_names[i] && !g_api.nvrtc; ++i) {
        g_api.nvrtc = dir.empty() ? load_lib(nvrtc_names[i]) : load_lib_path(dir + nvrtc_names[i]);
      }
      if (g_api.nvrtc) break;
    }
  }
#else
  g_api.nvcuda = load_lib("libcuda.so.1");
  if (g_api.nvcuda) {
    g_api.cuCtxGetCurrent = reinterpret_cast<cuCtxGetCurrent_t>(sym(g_api.nvcuda, "cuCtxGetCurrent"));
    g_api.cuDevicePrimaryCtxRetain =
        reinterpret_cast<cuDevicePrimaryCtxRetain_t>(sym(g_api.nvcuda, "cuDevicePrimaryCtxRetain"));
    g_api.cuCtxSetCurrent = reinterpret_cast<cuCtxSetCurrent_t>(sym(g_api.nvcuda, "cuCtxSetCurrent"));
    g_api.cuDeviceGetAttribute =
        reinterpret_cast<cuDeviceGetAttribute_t>(sym(g_api.nvcuda, "cuDeviceGetAttribute"));
    g_api.cuModuleLoadData =
        reinterpret_cast<cuModuleLoadData_t>(sym(g_api.nvcuda, "cuModuleLoadData"));
    g_api.cuModuleGetFunction =
        reinterpret_cast<cuModuleGetFunction_t>(sym(g_api.nvcuda, "cuModuleGetFunction"));
    g_api.cuModuleUnload = reinterpret_cast<cuModuleUnload_t>(sym(g_api.nvcuda, "cuModuleUnload"));
    g_api.cuLaunchKernel = reinterpret_cast<cuLaunchKernel_t>(sym(g_api.nvcuda, "cuLaunchKernel"));
  }
  g_api.nvrtc = load_lib("libnvrtc.so.13");
  if (!g_api.nvrtc) g_api.nvrtc = load_lib("libnvrtc.so.12");
  if (!g_api.nvrtc) g_api.nvrtc = load_lib("libnvrtc.so");
#endif
  if (g_api.nvrtc) {
    g_api.nvrtcCreateProgram =
        reinterpret_cast<nvrtcCreateProgram_t>(sym(g_api.nvrtc, "nvrtcCreateProgram"));
    g_api.nvrtcCompileProgram =
        reinterpret_cast<nvrtcCompileProgram_t>(sym(g_api.nvrtc, "nvrtcCompileProgram"));
    g_api.nvrtcGetPTXSize = reinterpret_cast<nvrtcGetPTXSize_t>(sym(g_api.nvrtc, "nvrtcGetPTXSize"));
    g_api.nvrtcGetPTX = reinterpret_cast<nvrtcGetPTX_t>(sym(g_api.nvrtc, "nvrtcGetPTX"));
    g_api.nvrtcGetProgramLogSize =
        reinterpret_cast<nvrtcGetProgramLogSize_t>(sym(g_api.nvrtc, "nvrtcGetProgramLogSize"));
    g_api.nvrtcGetProgramLog =
        reinterpret_cast<nvrtcGetProgramLog_t>(sym(g_api.nvrtc, "nvrtcGetProgramLog"));
    g_api.nvrtcDestroyProgram =
        reinterpret_cast<nvrtcDestroyProgram_t>(sym(g_api.nvrtc, "nvrtcDestroyProgram"));
    g_api.nvrtcGetErrorString =
        reinterpret_cast<nvrtcGetErrorString_t>(sym(g_api.nvrtc, "nvrtcGetErrorString"));
  }
  return true;
}

bool ensure_xy(int M, int K, int n = 1) {
  if (n < 1) n = 1;
  const int need_k = K * n;
  const int need_m = M * n;
  if (need_k > g_cap_k || !g_dx) {
    if (g_dx) g_api.cudaFree(g_dx);
    if (g_api.cudaMalloc(&g_dx, sizeof(float) * static_cast<size_t>(need_k)) != kCudaSuccess)
      return false;
    g_cap_k = need_k;
  }
  if (need_m > g_cap_m || !g_dy) {
    if (g_dy) g_api.cudaFree(g_dy);
    if (g_api.cudaMalloc(&g_dy, sizeof(float) * static_cast<size_t>(need_m)) != kCudaSuccess)
      return false;
    g_cap_m = need_m;
  }
  g_cap_n = n;
  return true;
}

bool gemm_dev(const float* d_W, const float* x, float* y, int M, int K) {
  if (!ensure_xy(M, K, 1)) return false;
  if (g_api.cudaMemcpy(g_dx, x, sizeof(float) * static_cast<size_t>(K), kCudaMemcpyH2D) !=
      kCudaSuccess)
    return false;
  const float alpha = 1.f, beta = 0.f;
  if (g_api.cublasSgemm(g_cublas, kCublasOpT, kCublasOpN, M, 1, K, &alpha, d_W, K,
                        reinterpret_cast<float*>(g_dx), K, &beta, reinterpret_cast<float*>(g_dy),
                        M) != kCublasSuccess)
    return false;
  return g_api.cudaMemcpy(y, g_dy, sizeof(float) * static_cast<size_t>(M), kCudaMemcpyD2H) ==
         kCudaSuccess;
}

// Y[n,M] row-major = W[M,K] @ X[n,K]^T per-row; X row-major ≡ col-major K×n
bool gemm_dev_batch(const float* d_W, const float* X, int n, float* Y, int M, int K) {
  if (n <= 1) return gemm_dev(d_W, X, Y, M, K);
  if (!ensure_xy(M, K, n)) return false;
  if (g_api.cudaMemcpy(g_dx, X, sizeof(float) * static_cast<size_t>(n) * K, kCudaMemcpyH2D) !=
      kCudaSuccess)
    return false;
  const float alpha = 1.f, beta = 0.f;
  if (g_api.cublasSgemm(g_cublas, kCublasOpT, kCublasOpN, M, n, K, &alpha, d_W, K,
                        reinterpret_cast<float*>(g_dx), K, &beta, reinterpret_cast<float*>(g_dy),
                        M) != kCublasSuccess)
    return false;
  return g_api.cudaMemcpy(Y, g_dy, sizeof(float) * static_cast<size_t>(n) * M, kCudaMemcpyD2H) ==
         kCudaSuccess;
}

// Returns cached d_W or nullptr if cannot upload.
const float* ensure_int4_device(const qlwc::Int4View& W) {
  if (!g_enabled || !W.qweight || W.M <= 0 || W.K <= 0) return nullptr;
  if (W.M >= kMaxGpuInt4Rows) return nullptr;
  auto it = g_cache.find(W.qweight);
  if (it != g_cache.end()) {
    if (it->second.M != W.M || it->second.K != W.K) return nullptr;
    return reinterpret_cast<const float*>(it->second.d_W);
  }
  const size_t nbytes = sizeof(float) * static_cast<size_t>(W.M) * W.K;
  if (g_budget > 0 && g_used + nbytes > g_budget) return nullptr;
  std::vector<float> host(static_cast<size_t>(W.M) * W.K);
  try {
    llmoc::hal::dequant_int4_matrix(W, host.data());
  } catch (...) {
    return nullptr;
  }
  void* dW = nullptr;
  if (g_api.cudaMalloc(&dW, nbytes) != kCudaSuccess) return nullptr;
  if (g_api.cudaMemcpy(dW, host.data(), nbytes, kCudaMemcpyH2D) != kCudaSuccess) {
    g_api.cudaFree(dW);
    return nullptr;
  }
  CacheEntry e;
  e.d_W = dW;
  e.M = W.M;
  e.K = W.K;
  e.bytes = nbytes;
  g_cache[W.qweight] = e;
  g_used += nbytes;
  return reinterpret_cast<const float*>(dW);
}

// 上传 INT4 量化形态(qweight packed + fp16 scales + 可选 zeros)到 VRAM。
// 预算按量化字节数: M*K/2(权重)+ M*ng*2(scales)+ M*ng*2(zeros, 若非空), 相比 FP32 省 ~8x。
// key = W.qweight, 与 g_cache 共享生命周期(disable 一起清)。
const Int4Resident* ensure_int4_resident(const qlwc::Int4View& W) {
  if (!g_enabled || !W.qweight || !W.scales || W.M <= 0 || W.K <= 0) return nullptr;
  if (W.M >= kMaxGpuInt4Rows) return nullptr;
  auto it = g_int4_cache.find(W.qweight);
  if (it != g_int4_cache.end()) {
    if (it->second.M != W.M || it->second.K != W.K) return nullptr;
    it->second.last_use = ++g_lru_tick;
    return &it->second;
  }
  const int gs = W.group_size > 0 ? W.group_size : 128;
  const int ng = (W.K + gs - 1) / gs;
  const bool is_awq = (W.zeros == nullptr) || (W.scheme == qlwc::Scheme::kAwqSym);
  const size_t rb = (static_cast<size_t>(W.K) + 1) / 2;
  const size_t nw = rb * static_cast<size_t>(W.M);
  const size_t ns = static_cast<size_t>(W.M) * ng;
  const size_t nz = is_awq ? 0 : ns;
  const size_t total = nw + ns * sizeof(uint16_t) + nz * sizeof(uint16_t);
  if (g_budget > 0 && g_used + total > g_budget) {
    // LRU 腾出空间
    std::vector<std::pair<uint64_t, const void*>> items;
    items.reserve(g_int4_cache.size());
    for (auto& kv : g_int4_cache) items.push_back({kv.second.last_use, kv.first});
    std::sort(items.begin(), items.end());
    for (auto& p : items) {
      if (g_used + total <= g_budget) break;
      auto it2 = g_int4_cache.find(p.second);
      if (it2 == g_int4_cache.end()) continue;
      if (it2->second.d_qweight) g_api.cudaFree(it2->second.d_qweight);
      if (it2->second.d_scales) g_api.cudaFree(it2->second.d_scales);
      if (it2->second.d_zeros) g_api.cudaFree(it2->second.d_zeros);
      g_used -= it2->second.bytes;
      g_int4_cache.erase(it2);
    }
    if (g_used + total > g_budget) return nullptr;
  }
  void* dq = nullptr;
  void* ds = nullptr;
  void* dz = nullptr;
  if (g_api.cudaMalloc(&dq, nw) != kCudaSuccess) return nullptr;
  if (g_api.cudaMemcpy(dq, W.qweight, nw, kCudaMemcpyH2D) != kCudaSuccess) {
    g_api.cudaFree(dq);
    return nullptr;
  }
  const size_t sz_bytes = ns * sizeof(uint16_t);
  if (g_api.cudaMalloc(&ds, sz_bytes) != kCudaSuccess) {
    g_api.cudaFree(dq);
    return nullptr;
  }
  if (g_api.cudaMemcpy(ds, W.scales, sz_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
    g_api.cudaFree(ds);
    g_api.cudaFree(dq);
    return nullptr;
  }
  if (!is_awq && W.zeros) {
    if (g_api.cudaMalloc(&dz, sz_bytes) != kCudaSuccess) {
      g_api.cudaFree(ds);
      g_api.cudaFree(dq);
      return nullptr;
    }
    if (g_api.cudaMemcpy(dz, W.zeros, sz_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(dz);
      g_api.cudaFree(ds);
      g_api.cudaFree(dq);
      return nullptr;
    }
  }
  Int4Resident e;
  e.d_qweight = dq;
  e.d_scales = ds;
  e.d_zeros = dz;
  e.M = W.M;
  e.K = W.K;
  e.ng = ng;
  e.gs = gs;
  e.is_awq = is_awq;
  e.bytes = total;
  e.last_use = ++g_lru_tick;
  g_int4_cache[W.qweight] = e;
  g_used += total;
  return &g_int4_cache[W.qweight];
}

const float* ensure_fp32_matrix(const void* key, int M, int K,
                                const std::function<void(float*)>& fill) {
  if (!g_enabled || !key || M <= 0 || K <= 0) return nullptr;
  if (M >= kMaxGpuInt4Rows) return nullptr;
  auto it = g_cache.find(key);
  if (it != g_cache.end()) {
    if (it->second.M != M || it->second.K != K) return nullptr;
    return reinterpret_cast<const float*>(it->second.d_W);
  }
  const size_t nbytes = sizeof(float) * static_cast<size_t>(M) * K;
  if (g_budget > 0 && g_used + nbytes > g_budget) return nullptr;
  std::vector<float> host(static_cast<size_t>(M) * K);
  try {
    fill(host.data());
  } catch (...) {
    return nullptr;
  }
  void* dW = nullptr;
  if (g_api.cudaMalloc(&dW, nbytes) != kCudaSuccess) return nullptr;
  if (g_api.cudaMemcpy(dW, host.data(), nbytes, kCudaMemcpyH2D) != kCudaSuccess) {
    g_api.cudaFree(dW);
    return nullptr;
  }
  CacheEntry e;
  e.d_W = dW;
  e.M = M;
  e.K = K;
  e.bytes = nbytes;
  g_cache[key] = e;
  g_used += nbytes;
  return reinterpret_cast<const float*>(dW);
}

const float* ensure_awq_device(const AwqView& W) {
  return ensure_fp32_matrix(W.qweight, W.M, W.K, [&](float* out) { dequant_awq_matrix(W, out); });
}

const float* ensure_nvfp4_device(const Nvfp4View& W) {
  return ensure_fp32_matrix(W.qweight, W.M, W.K, [&](float* out) { dequant_nvfp4_matrix(W, out); });
}

}  // namespace

bool probe_available() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_probed) return g_probe_ok;
  g_probed = true;
  std::string err;
  if (!load_apis(err)) {
    g_status = err;
    g_probe_ok = false;
    return false;
  }
  int n = 0;
  if (g_api.cudaGetDeviceCount(&n) != kCudaSuccess || n < 1) {
    g_status = "no CUDA device";
    g_probe_ok = false;
    return false;
  }
  g_status = "cuda probed";
  g_probe_ok = true;
  return true;
}

bool enabled() { return g_enabled; }
const char* status() { return g_status.c_str(); }
size_t vram_used() { return g_used; }
size_t vram_budget() { return g_budget; }

int device_count() {
  if (!probe_available()) return 0;
  std::lock_guard<std::mutex> lock(g_mu);
  int n = 0;
  if (!g_api.cudaGetDeviceCount || g_api.cudaGetDeviceCount(&n) != kCudaSuccess) return 0;
  return n;
}

void* device_alloc(size_t bytes) {
  if (!g_enabled || bytes == 0) return nullptr;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_budget > 0 && g_used + bytes > g_budget) return nullptr;
  void* p = nullptr;
  if (g_api.cudaMalloc(&p, bytes) != kCudaSuccess) return nullptr;
  g_used += bytes;
  return p;
}

void device_free(void* p) {
  if (!p || !g_enabled) return;
  std::lock_guard<std::mutex> lock(g_mu);
  // g_used is not tracked per-allocation, so we can't subtract here.
  // This is a known limitation; activation buffers are small relative to weights.
  g_api.cudaFree(p);
}

bool h2d(void* dst, const void* src, size_t bytes) {
  if (!g_enabled || !dst || !src || bytes == 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(dst, src, bytes, kCudaMemcpyH2D) == kCudaSuccess;
}

bool d2h(void* dst, const void* src, size_t bytes) {
  if (!g_enabled || !dst || !src || bytes == 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(dst, src, bytes, kCudaMemcpyD2H) == kCudaSuccess;
}

bool enable(size_t vram_budget_bytes) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_budget = vram_budget_bytes;
  if (g_enabled) return true;
  std::string err;
  if (!load_apis(err)) {
    g_status = err;
    return false;
  }
  int n = 0;
  if (g_api.cudaGetDeviceCount(&n) != kCudaSuccess || n < 1) {
    g_status = "no CUDA device";
    return false;
  }
  if (g_api.cudaSetDevice(0) != kCudaSuccess) {
    g_status = "cudaSetDevice failed";
    return false;
  }
  if (g_api.cublasCreate(&g_cublas) != kCublasSuccess || !g_cublas) {
    g_status = "cublasCreate failed";
    return false;
  }
  g_probed = true;
  g_probe_ok = true;
  g_enabled = true;
  g_status = "cuda+cublas enabled";
  return true;
}

void disable() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_jit_kernels.clear();  // 名字缓存先清, 再卸载模块, 避免悬空句柄
  for (auto& kv : g_jit_modules) {
    if (kv.second && g_api.cuModuleUnload) g_api.cuModuleUnload(kv.second);
  }
  g_jit_modules.clear();
  for (auto& kv : g_cache) {
    if (kv.second.d_W) g_api.cudaFree(kv.second.d_W);
  }
  g_cache.clear();
  for (auto& kv : g_int4_cache) {
    if (kv.second.d_qweight) g_api.cudaFree(kv.second.d_qweight);
    if (kv.second.d_scales) g_api.cudaFree(kv.second.d_scales);
    if (kv.second.d_zeros) g_api.cudaFree(kv.second.d_zeros);
  }
  g_int4_cache.clear();
  for (auto& kv : g_gdn_state) {
    if (kv.second) g_api.cudaFree(kv.second);
  }
  g_gdn_state.clear();
  if (g_gdn_buf) g_api.cudaFree(g_gdn_buf);
  g_gdn_buf = nullptr;
  g_gdn_buf_cap = 0;
  if (g_attn_q) g_api.cudaFree(g_attn_q);
  if (g_attn_k) g_api.cudaFree(g_attn_k);
  if (g_attn_v) g_api.cudaFree(g_attn_v);
  if (g_attn_o) g_api.cudaFree(g_attn_o);
  g_attn_q = g_attn_k = g_attn_v = g_attn_o = nullptr;
  g_attn_q_bytes = g_attn_k_bytes = g_attn_v_bytes = g_attn_o_bytes = 0;
  g_prof_h2d_us = g_prof_kernel_us = g_prof_d2h_us = 0.0;
  g_prof_calls = 0;
  if (g_dx) g_api.cudaFree(g_dx);
  if (g_dy) g_api.cudaFree(g_dy);
  g_dx = g_dy = nullptr;
  g_cap_k = g_cap_m = g_cap_n = 0;
  if (g_cublas && g_api.cublasDestroy) g_api.cublasDestroy(g_cublas);
  g_cublas = nullptr;
  g_used = 0;
  g_enabled = false;
  g_status = "disabled";
}

bool jit_available() {
  return g_enabled && g_api.nvrtc && g_api.nvcuda && g_api.nvrtcCreateProgram &&
         g_api.nvrtcCompileProgram && g_api.nvrtcGetPTXSize && g_api.nvrtcGetPTX &&
         g_api.cuModuleLoadData && g_api.cuModuleGetFunction && g_api.cuLaunchKernel;
}

namespace {
bool ensure_driver_ctx() {
  if (!g_api.cuCtxGetCurrent) return false;
  void* cur = nullptr;
  if (g_api.cuCtxGetCurrent(&cur) != 0) return false;
  if (cur) return true;
  if (!g_api.cuDevicePrimaryCtxRetain || !g_api.cuCtxSetCurrent) return false;
  void* pctx = nullptr;
  if (g_api.cuDevicePrimaryCtxRetain(&pctx, 0) != 0 || !pctx) return false;
  return g_api.cuCtxSetCurrent(pctx) == 0;
}

std::string jit_arch_opt() {
  int major = 8, minor = 0;
  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=75, MINOR=76
  if (g_api.cuDeviceGetAttribute) {
    int mj = 0, mn = 0;
    if (g_api.cuDeviceGetAttribute(&mj, 75, 0) == 0 &&
        g_api.cuDeviceGetAttribute(&mn, 76, 0) == 0 && mj >= 1 && mj <= 99) {
      major = mj;
      minor = mn;
    }
  }
  return "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
}
}  // namespace

bool jit_compile(const char* cuda_src, const char* kernel_name, void** out_fn) {
  if (out_fn) *out_fn = nullptr;
  if (!cuda_src || !kernel_name || !out_fn) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!jit_available()) {
    g_status = "jit unavailable (nvrtc/nvcuda)";
    return false;
  }
  if (!ensure_driver_ctx()) {
    g_status = "jit: no driver ctx";
    return false;
  }
  void* prog = nullptr;
  if (g_api.nvrtcCreateProgram(&prog, cuda_src, kernel_name, 0, nullptr, nullptr) != 0 || !prog) {
    g_status = "jit: nvrtcCreateProgram failed";
    return false;
  }
  const std::string arch = jit_arch_opt();
  const char* opts[] = {arch.c_str(), "--use_fast_math"};
  const int rc = g_api.nvrtcCompileProgram(prog, 2, opts);
  if (rc != 0) {
    size_t logsz = 0;
    std::string log = "jit: compile failed";
    if (g_api.nvrtcGetProgramLogSize && g_api.nvrtcGetProgramLog) {
      g_api.nvrtcGetProgramLogSize(prog, &logsz);
      std::string buf(logsz, '\0');
      g_api.nvrtcGetProgramLog(prog, buf.data());
      log += ": " + buf;
    }
    if (g_api.nvrtcGetErrorString) log += std::string(" [") + g_api.nvrtcGetErrorString(rc) + "]";
    g_status = log.substr(0, 512);
    if (g_api.nvrtcDestroyProgram) g_api.nvrtcDestroyProgram(&prog);
    return false;
  }
  size_t ptxsz = 0;
  if (g_api.nvrtcGetPTXSize(prog, &ptxsz) != 0 || ptxsz == 0) {
    g_status = "jit: no ptx";
    if (g_api.nvrtcDestroyProgram) g_api.nvrtcDestroyProgram(&prog);
    return false;
  }
  std::string ptx(ptxsz, '\0');
  if (g_api.nvrtcGetPTX(prog, ptx.data()) != 0) {
    g_status = "jit: get ptx failed";
    if (g_api.nvrtcDestroyProgram) g_api.nvrtcDestroyProgram(&prog);
    return false;
  }
  if (g_api.nvrtcDestroyProgram) g_api.nvrtcDestroyProgram(&prog);
  void* mod = nullptr;
  if (g_api.cuModuleLoadData(&mod, ptx.c_str()) != 0 || !mod) {
    g_status = "jit: cuModuleLoadData failed";
    return false;
  }
  void* fn = nullptr;
  if (g_api.cuModuleGetFunction(&fn, mod, kernel_name) != 0 || !fn) {
    g_status = "jit: cuModuleGetFunction failed";
    if (g_api.cuModuleUnload) g_api.cuModuleUnload(mod);
    return false;
  }
  g_jit_modules[fn] = mod;
  *out_fn = fn;
  return true;
}

bool jit_launch(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                unsigned bz, unsigned shmem_bytes, void** params) {
  if (!fn || !g_api.cuLaunchKernel) return false;
  if (!ensure_driver_ctx()) return false;
  return g_api.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem_bytes, nullptr, params,
                              nullptr) == 0;
}

// ---- GPU 原生 INT4 dequant-GEMV kernel(权重量化形态常驻, kernel 内反量化) ----
namespace {

// kernel 源: 与 CPU hal::gemm_int4 语义一致。
// 打包格式: qweight 行主序 M×rb(rb=(K+1)/2), 偶数 k 取低 4 位, 奇数 k 取高 4 位。
// awq: w=(q-7)*scale; gptq: w=q*scale+zero。每 block 算一行, 256 线程, warp shuffle 归约。
const char* kGemvInt4Src = R"CUDA(
__device__ __forceinline__ float f16_to_f32_dev(unsigned short h) {
  unsigned int sign = (h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int man = h & 0x3FFu;
  unsigned int bits;
  if (exp == 0u) {
    if (man == 0u) { bits = sign; }
    else {
      exp = 1u;
      while ((man & 0x400u) == 0u) { man <<= 1; exp -= 1u; }
      man &= 0x3FFu;
      bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
  } else if (exp == 31u) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  return __uint_as_float(bits);
}

// 多行/块的 GEMV kernel: 每个 block 处理 ROWS_PER_BLOCK 行, blockDim.x 并行 K 维。
// 关键优化: scales/zeros 加载到 shared mem(避免重复 FP16→FP32), qrow 按 int4 向量化读
// (16 字节 = 32 INT4 = 2 warp-iter), x 按 float4 向量化读(16 字节 = 4 floats)。
#define ROWS_PER_BLOCK 8
extern "C" __global__ void gemv_int4(
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int ng, int gs, int is_awq) {
  const int row0 = blockIdx.x * ROWS_PER_BLOCK;
  const int tid = threadIdx.x;
  const int rb = (K + 1) >> 1;
  // 共享 scales/zeros(ng <= K/128 ≈ 24, 适合 shared)
  extern __shared__ unsigned short smem_buf[];
  unsigned short* s_scales = smem_buf;
  unsigned short* s_zeros = is_awq ? nullptr : smem_buf + ROWS_PER_BLOCK * ng;
  for (int row_off = 0; row_off < ROWS_PER_BLOCK; ++row_off) {
    const int m = row0 + row_off;
    if (m >= M) break;
    for (int g = tid; g < ng; g += blockDim.x) {
      s_scales[row_off * ng + g] = scales[m * ng + g];
    }
    if (!is_awq) {
      for (int g = tid; g < ng; g += blockDim.x) {
        s_zeros[row_off * ng + g] = zeros[m * ng + g];
      }
    }
  }
  __syncthreads();
  // 主体: 每行一个 warp 处理; blockDim.x / 32 warps_per_block 行并行
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  if (warp_id >= ROWS_PER_BLOCK) return;
  const int m = row0 + warp_id;
  if (m >= M) return;
  const unsigned char* qrow = qweight + (size_t)m * (size_t)rb;
  float acc = 0.f;
  // 每 lane 处理 K/32 个连续 k; 用 uchar4 向量化读 packed weight(8 INT4/load)
  for (int k = lane * 8; k < K; k += 32 * 8) {
    const int g = k / gs;
    const float s = f16_to_f32_dev(s_scales[warp_id * ng + g]);
    const float z = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + g]);
    // 向量化读 4 bytes = 8 INT4; 安全前提: rb%4==0(K=8*n)
    const int kp = k >> 1;
    if (kp + 3 < rb) {
      const uchar4 v = *reinterpret_cast<const uchar4*>(qrow + kp);
      // v.x[lo,hi], v.z[lo,hi] 等; 每字节 2 个 INT4
      const float x0 = x[k + 0], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
      const float x4 = x[k + 4], x5 = x[k + 5], x6 = x[k + 6], x7 = x[k + 7];
      const float w0 = ((float)((v.x & 0xF) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w1 = ((float)((v.x >> 4) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w2 = ((float)((v.y & 0xF) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w3 = ((float)((v.y >> 4) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w4 = ((float)((v.z & 0xF) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w5 = ((float)((v.z >> 4) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w6 = ((float)((v.w & 0xF) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      const float w7 = ((float)((v.w >> 4) - (is_awq ? 7 : 0))) * s + (is_awq ? 0.f : z);
      acc += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + x5 * w5 + x6 * w6 + x7 * w7;
    } else {
      // 尾部逐字节处理
      for (int kk = k; kk < k + 8 && kk < K; ++kk) {
        const int gg = kk / gs;
        const float ss = f16_to_f32_dev(s_scales[warp_id * ng + gg]);
        const float zz = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + gg]);
        const unsigned char b = qrow[kk >> 1];
        const int qi = (kk & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
        const float w = ((float)qi - (is_awq ? 7 : 0)) * ss + (is_awq ? 0.f : zz);
        acc += x[kk] * w;
      }
    }
  }
  // warp reduce
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[m] = acc;
}

// Prefill batch GEMM: Y[n,M] = X[n,K] @ W[M,K]^T with on-the-fly INT4 dequant.
// grid.x = ceil(M / ROWS_PER_BLOCK). Each warp owns one output row and tiles over
// batch (BT=8): dequantized weights are reused across the batch tile (far fewer
// qweight reads than launching one gemv per token).
extern "C" __global__ void gemm_int4(
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int n, int ng, int gs, int is_awq) {
  const int row0 = blockIdx.x * ROWS_PER_BLOCK;
  const int tid = threadIdx.x;
  const int rb = (K + 1) >> 1;
  extern __shared__ unsigned short smem_buf[];
  unsigned short* s_scales = smem_buf;
  unsigned short* s_zeros = is_awq ? nullptr : smem_buf + ROWS_PER_BLOCK * ng;
  for (int row_off = 0; row_off < ROWS_PER_BLOCK; ++row_off) {
    const int m = row0 + row_off;
    if (m >= M) break;
    for (int g = tid; g < ng; g += blockDim.x) {
      s_scales[row_off * ng + g] = scales[m * ng + g];
    }
    if (!is_awq) {
      for (int g = tid; g < ng; g += blockDim.x) {
        s_zeros[row_off * ng + g] = zeros[m * ng + g];
      }
    }
  }
  __syncthreads();
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  if (warp_id >= ROWS_PER_BLOCK) return;
  const int m = row0 + warp_id;
  if (m >= M) return;
  const unsigned char* qrow = qweight + (size_t)m * (size_t)rb;
  const int off0 = is_awq ? 7 : 0;
  constexpr int BT = 8;
  for (int b0 = 0; b0 < n; b0 += BT) {
    const int bn = (b0 + BT <= n) ? BT : (n - b0);
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;
    for (int k = lane * 8; k < K; k += 32 * 8) {
      const int g = k / gs;
      const float s = f16_to_f32_dev(s_scales[warp_id * ng + g]);
      const float z = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + g]);
      const int kp = k >> 1;
      float w0, w1, w2, w3, w4, w5, w6, w7;
      if (kp + 3 < rb) {
        const uchar4 v = *reinterpret_cast<const uchar4*>(qrow + kp);
        w0 = ((float)((v.x & 0xF) - off0)) * s + z;
        w1 = ((float)((v.x >> 4) - off0)) * s + z;
        w2 = ((float)((v.y & 0xF) - off0)) * s + z;
        w3 = ((float)((v.y >> 4) - off0)) * s + z;
        w4 = ((float)((v.z & 0xF) - off0)) * s + z;
        w5 = ((float)((v.z >> 4) - off0)) * s + z;
        w6 = ((float)((v.w & 0xF) - off0)) * s + z;
        w7 = ((float)((v.w >> 4) - off0)) * s + z;
      } else {
        w0 = w1 = w2 = w3 = w4 = w5 = w6 = w7 = 0.f;
        for (int t = 0; t < 8; ++t) {
          const int kk = k + t;
          if (kk >= K) break;
          const int gg = kk / gs;
          const float ss = f16_to_f32_dev(s_scales[warp_id * ng + gg]);
          const float zz = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + gg]);
          const unsigned char packed = qrow[kk >> 1];
          const int qi = (kk & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
          const float w = ((float)(qi - off0)) * ss + zz;
          if (t == 0) w0 = w; else if (t == 1) w1 = w; else if (t == 2) w2 = w;
          else if (t == 3) w3 = w; else if (t == 4) w4 = w; else if (t == 5) w5 = w;
          else if (t == 6) w6 = w; else w7 = w;
        }
      }
      // Reuse the 8 weights across up to BT batch rows
      for (int bi = 0; bi < bn; ++bi) {
        const float* x = X + (size_t)(b0 + bi) * (size_t)K;
        float partial = 0.f;
        if (k + 7 < K) {
          partial = x[k]*w0 + x[k+1]*w1 + x[k+2]*w2 + x[k+3]*w3
                  + x[k+4]*w4 + x[k+5]*w5 + x[k+6]*w6 + x[k+7]*w7;
        } else {
          if (k + 0 < K) partial += x[k + 0] * w0;
          if (k + 1 < K) partial += x[k + 1] * w1;
          if (k + 2 < K) partial += x[k + 2] * w2;
          if (k + 3 < K) partial += x[k + 3] * w3;
          if (k + 4 < K) partial += x[k + 4] * w4;
          if (k + 5 < K) partial += x[k + 5] * w5;
          if (k + 6 < K) partial += x[k + 6] * w6;
          if (k + 7 < K) partial += x[k + 7] * w7;
        }
        if (bi == 0) acc0 += partial;
        else if (bi == 1) acc1 += partial;
        else if (bi == 2) acc2 += partial;
        else if (bi == 3) acc3 += partial;
        else if (bi == 4) acc4 += partial;
        else if (bi == 5) acc5 += partial;
        else if (bi == 6) acc6 += partial;
        else acc7 += partial;
      }
    }
    for (int off = 16; off > 0; off >>= 1) acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
    for (int off = 16; off > 0; off >>= 1) acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffffu, acc2, off);
    for (int off = 16; off > 0; off >>= 1) acc3 += __shfl_down_sync(0xffffffffu, acc3, off);
    for (int off = 16; off > 0; off >>= 1) acc4 += __shfl_down_sync(0xffffffffu, acc4, off);
    for (int off = 16; off > 0; off >>= 1) acc5 += __shfl_down_sync(0xffffffffu, acc5, off);
    for (int off = 16; off > 0; off >>= 1) acc6 += __shfl_down_sync(0xffffffffu, acc6, off);
    for (int off = 16; off > 0; off >>= 1) acc7 += __shfl_down_sync(0xffffffffu, acc7, off);
    if (lane == 0) {
      if (bn > 0) Y[(size_t)(b0 + 0) * (size_t)M + m] = acc0;
      if (bn > 1) Y[(size_t)(b0 + 1) * (size_t)M + m] = acc1;
      if (bn > 2) Y[(size_t)(b0 + 2) * (size_t)M + m] = acc2;
      if (bn > 3) Y[(size_t)(b0 + 3) * (size_t)M + m] = acc3;
      if (bn > 4) Y[(size_t)(b0 + 4) * (size_t)M + m] = acc4;
      if (bn > 5) Y[(size_t)(b0 + 5) * (size_t)M + m] = acc5;
      if (bn > 6) Y[(size_t)(b0 + 6) * (size_t)M + m] = acc6;
      if (bn > 7) Y[(size_t)(b0 + 7) * (size_t)M + m] = acc7;
    }
  }
}

// Fused multi-GEMV: up to 4 weight sets sharing same x/K/ng/gs/is_awq.
// Grid covers sum(M_i) rows. Each warp handles 1 row, routes to correct task by row index.
// Saves (nt-1) kernel launches + (nt-1) H2D(x) per call site.
extern "C" __global__ void gemv_multi4_int4(
    const unsigned char* __restrict__ q0, const unsigned short* __restrict__ s0,
    const unsigned short* __restrict__ z0, float* __restrict__ y0, int m0,
    const unsigned char* __restrict__ q1, const unsigned short* __restrict__ s1,
    const unsigned short* __restrict__ z1, float* __restrict__ y1, int m1,
    const unsigned char* __restrict__ q2, const unsigned short* __restrict__ s2,
    const unsigned short* __restrict__ z2, float* __restrict__ y2, int m2,
    const unsigned char* __restrict__ q3, const unsigned short* __restrict__ s3,
    const unsigned short* __restrict__ z3, float* __restrict__ y3, int m3,
    int nt, int K, int ng, int gs, int is_awq,
    const float* __restrict__ x) {
  const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  // Route row to task
  int loc, task;
  if (row < m0) { task = 0; loc = row; }
  else if (row < m0 + m1) { task = 1; loc = row - m0; }
  else if (row < m0 + m1 + m2) { task = 2; loc = row - m0 - m1; }
  else if (row < m0 + m1 + m2 + m3) { task = 3; loc = row - m0 - m1 - m2; }
  else return;
  const unsigned char* qw; const unsigned short* sc; const unsigned short* zz; float* yout;
  if (task == 0) { qw = q0; sc = s0; zz = z0; yout = y0; }
  else if (task == 1) { qw = q1; sc = s1; zz = z1; yout = y1; }
  else if (task == 2) { qw = q2; sc = s2; zz = z2; yout = y2; }
  else { qw = q3; sc = s3; zz = z3; yout = y3; }
  const int rb = (K + 1) >> 1;
  const unsigned char* qrow = qw + (size_t)loc * rb;
  // Per-warp shared memory for scales/zeros of this row
  extern __shared__ unsigned short smem_multi[];
  unsigned short* my_s = smem_multi + warp_id * ng * 2;
  unsigned short* my_z = my_s + ng;
  for (int i = lane; i < ng; i += 32) {
    my_s[i] = sc[(size_t)loc * ng + i];
    if (!is_awq) my_z[i] = zz[(size_t)loc * ng + i];
  }
  __syncwarp();
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 32 * 8) {
    const int g = k / gs;
    const float s = f16_to_f32_dev(my_s[g]);
    const float z = is_awq ? 0.f : f16_to_f32_dev(my_z[g]);
    const int kp = k >> 1;
    if (kp + 3 < rb) {
      const uchar4 v = *reinterpret_cast<const uchar4*>(qrow + kp);
      const float x0=x[k],x1=x[k+1],x2=x[k+2],x3=x[k+3],x4=x[k+4],x5=x[k+5],x6=x[k+6],x7=x[k+7];
      const int off = is_awq ? 7 : 0;
      acc += x0*(((float)((v.x&0xF)-off))*s+z) + x1*(((float)((v.x>>4)-off))*s+z)
           + x2*(((float)((v.y&0xF)-off))*s+z) + x3*(((float)((v.y>>4)-off))*s+z)
           + x4*(((float)((v.z&0xF)-off))*s+z) + x5*(((float)((v.z>>4)-off))*s+z)
           + x6*(((float)((v.w&0xF)-off))*s+z) + x7*(((float)((v.w>>4)-off))*s+z);
    } else {
      for (int kk = k; kk < k + 8 && kk < K; ++kk) {
        const int gr = kk / gs;
        const float ss = f16_to_f32_dev(my_s[gr]);
        const float zzv = is_awq ? 0.f : f16_to_f32_dev(my_z[gr]);
        const unsigned char b = qrow[kk >> 1];
        const int qi = (kk & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
        const int off = is_awq ? 7 : 0;
        acc += x[kk] * (((float)(qi - off)) * ss + zzv);
      }
    }
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) yout[loc] = acc;
}
)CUDA";

// GDN kernel: separate source to avoid interfering with GEMV JIT compilation.
const char* kGdnSrc = R"CUDA(
extern "C" __global__ void gated_delta_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ state,
    float* __restrict__ out, int dk, int dv, float scale) {
  const int h = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= dv) return;
  const float* qh = q + h * dk;
  const float* kh = k + h * dk;
  const float* vh = v + h * dv;
  float g_log = g[h];
  float beta_t = beta[h];
  float* st = state + (size_t)h * dk * dv;
  float qn = 0.f, kn = 0.f;
  for (int i = 0; i < dk; ++i) { qn += qh[i]*qh[i]; kn += kh[i]*kh[i]; }
  qn = rsqrtf(qn + 1e-12f); kn = rsqrtf(kn + 1e-12f);
  g_log = fmaxf(-80.f, fminf(0.f, g_log));
  const float g_t = expf(g_log);
  beta_t = fminf(1.f, fmaxf(0.f, beta_t));
  float kv_j = 0.f;
  for (int i = 0; i < dk; ++i) {
    float s = st[i * dv + j] * g_t;
    st[i * dv + j] = s;
    kv_j += kh[i] * kn * s;
  }
  const float delta_j = beta_t * (vh[j] - kv_j);
  float out_j = 0.f;
  for (int i = 0; i < dk; ++i) {
    float s = st[i * dv + j] + kh[i] * kn * delta_j;
    s = fminf(1e4f, fmaxf(-1e4f, s));
    st[i * dv + j] = s;
    out_j += qh[i] * qn * scale * s;
  }
  out[h * dv + j] = out_j;
}
)CUDA";

// Prefill causal attention (GPU modes only). Validated vs CPU in prefill_ops_bench (~13x @1064).
const char* kAttnPrefillSrc = R"CUDA(
extern "C" __global__ void attn_prefill_naive(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int seq, int n_heads, int n_kv, int hd, float scale) {
  const int tq = blockIdx.x;
  const int h = blockIdx.y;
  if (tq >= seq || h >= n_heads) return;
  const int g = n_heads / n_kv;
  const int hkv = h / g;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* scores = smem;
  const float* qh = q + ((size_t)tq * n_heads + h) * hd;
  for (int tk = tid; tk <= tq; tk += blockDim.x) {
    const float* kt = k + ((size_t)tk * n_kv + hkv) * hd;
    float dot = 0.f;
    for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
    scores[tk] = dot * scale;
  }
  __syncthreads();
  if (tid == 0) {
    float m = -1e30f;
    for (int tk = 0; tk <= tq; ++tk) m = fmaxf(m, scores[tk]);
    float sum = 0.f;
    for (int tk = 0; tk <= tq; ++tk) {
      scores[tk] = expf(scores[tk] - m);
      sum += scores[tk];
    }
    const float inv = 1.f / sum;
    for (int tk = 0; tk <= tq; ++tk) scores[tk] *= inv;
  }
  __syncthreads();
  float* oh = out + ((size_t)tq * n_heads + h) * hd;
  for (int d = tid; d < hd; d += blockDim.x) {
    float acc = 0.f;
    for (int tk = 0; tk <= tq; ++tk) {
      const float* vt = v + ((size_t)tk * n_kv + hkv) * hd;
      acc += scores[tk] * vt[d];
    }
    oh[d] = acc;
  }
}
)CUDA";

// kernel 名→句柄缓存(避免每 token 重编译; 与 g_jit_kernels 共享生命周期, disable 时清空)
void* get_jit_kernel(const char* src, const char* name) {
  const auto it = g_jit_kernels.find(name);
  if (it != g_jit_kernels.end()) return it->second;
  void* fn = nullptr;
  if (!jit_compile(src, name, &fn)) return nullptr;
  g_jit_kernels[name] = fn;
  return fn;
}

}  // namespace

bool jit_gemv_int4(const uint8_t* d_qweight, const uint16_t* d_scales, const uint16_t* d_zeros,
                   const float* d_x, float* d_y, int M, int K, int ng, int gs, bool is_awq) {
  if (!d_qweight || !d_scales || !d_x || !d_y || M <= 0 || K <= 0 || ng <= 0 || gs <= 0)
    return false;
  if (!is_awq && !d_zeros) return false;
  void* fn = get_jit_kernel(kGemvInt4Src, "gemv_int4");
  if (!fn) return false;
  int is_awq_i = is_awq ? 1 : 0;
  void* params[] = {&d_qweight, &d_scales, &d_zeros, &d_x, &d_y, &M, &K, &ng, &gs, &is_awq_i};
  constexpr int ROWS_PER_BLOCK = 8;
  constexpr int BLOCK_DIM = ROWS_PER_BLOCK * 32;  // 8 warps × 32 = 256
  const int blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  const unsigned shmem = sizeof(unsigned short) * ROWS_PER_BLOCK * ng *
                         (is_awq ? 1 : 2);
  return jit_launch(fn, static_cast<unsigned>(blocks), 1, 1,
                    static_cast<unsigned>(BLOCK_DIM), 1, 1, shmem, params);
}

bool jit_gemm_int4(const uint8_t* d_qweight, const uint16_t* d_scales, const uint16_t* d_zeros,
                   const float* d_X, float* d_Y, int M, int K, int n, int ng, int gs, bool is_awq) {
  if (!d_qweight || !d_scales || !d_X || !d_Y || M <= 0 || K <= 0 || n <= 0 || ng <= 0 || gs <= 0)
    return false;
  if (!is_awq && !d_zeros) return false;
  if (n == 1)
    return jit_gemv_int4(d_qweight, d_scales, d_zeros, d_X, d_Y, M, K, ng, gs, is_awq);
  void* fn = get_jit_kernel(kGemvInt4Src, "gemm_int4");
  if (!fn) return false;
  int is_awq_i = is_awq ? 1 : 0;
  void* params[] = {&d_qweight, &d_scales, &d_zeros, &d_X, &d_Y, &M, &K, &n, &ng, &gs, &is_awq_i};
  constexpr int ROWS_PER_BLOCK = 8;
  constexpr int BLOCK_DIM = ROWS_PER_BLOCK * 32;
  const int blocks_x = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  const unsigned shmem =
      sizeof(unsigned short) * ROWS_PER_BLOCK * ng * (is_awq ? 1u : 2u);
  return jit_launch(fn, static_cast<unsigned>(blocks_x), 1, 1,
                    static_cast<unsigned>(BLOCK_DIM), 1, 1, shmem, params);
}

bool try_gemm_w16(const float* x, const uint16_t* W, float* y, int M, int K, bool is_f16) {
  if (!g_enabled || !x || !W || !y || M <= 0 || K <= 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_cache.find(W);
  if (it == g_cache.end()) {
    const size_t nbytes = sizeof(float) * static_cast<size_t>(M) * K;
    if (g_budget > 0 && g_used + nbytes > g_budget) return false;
    if (M >= kMaxGpuInt4Rows) return false;
    std::vector<float> host(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < host.size(); ++i)
      host[i] = is_f16 ? f16_to_f32(W[i]) : bf16_to_f32(W[i]);
    void* dW = nullptr;
    if (g_api.cudaMalloc(&dW, nbytes) != kCudaSuccess) return false;
    if (g_api.cudaMemcpy(dW, host.data(), nbytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(dW);
      return false;
    }
    CacheEntry e;
    e.d_W = dW;
    e.M = M;
    e.K = K;
    e.bytes = nbytes;
    g_cache[W] = e;
    g_used += nbytes;
    it = g_cache.find(W);
  }
  if (it->second.M != M || it->second.K != K) return false;
  return gemm_dev(reinterpret_cast<const float*>(it->second.d_W), x, y, M, K);
}

bool prefetch_w16(const uint16_t* W, int M, int K, bool is_f16) {
  if (!g_enabled || !W || M <= 0 || K <= 0) return false;
  std::vector<float> dummy(static_cast<size_t>(K), 0.f);
  std::vector<float> out(static_cast<size_t>(M));
  return try_gemm_w16(dummy.data(), W, out.data(), M, K, is_f16);
}

bool try_gemm_int4(const float* x, const qlwc::Int4View& W, float* y) {
  if (!g_enabled || !x || !y) return false;
  const Int4Resident* res = nullptr;
  bool x_uploaded = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    res = ensure_int4_resident(W);
    const bool use_jit = res && jit_available();
    if (!use_jit) {
      const float* dW = ensure_int4_device(W);
      if (!dW) return false;
      return gemm_dev(dW, x, y, W.M, W.K);
    }
    // 锁内: 准备 GPU 端输入(x H2D)。之后解锁再调 jit_gemv_int4(其内部 jit_compile 会再锁 g_mu, 不可重入)。
    if (!ensure_xy(W.M, W.K, 1)) return false;
    if (g_api.cudaMemcpy(g_dx, x, sizeof(float) * static_cast<size_t>(W.K),
                         kCudaMemcpyH2D) != kCudaSuccess) return false;
    x_uploaded = true;
  }
  if (!x_uploaded) return false;
  auto tk0 = std::chrono::steady_clock::now();
  const bool ok = jit_gemv_int4(static_cast<const uint8_t*>(res->d_qweight),
                                static_cast<const uint16_t*>(res->d_scales),
                                static_cast<const uint16_t*>(res->d_zeros),
                                reinterpret_cast<const float*>(g_dx),
                                reinterpret_cast<float*>(g_dy),
                                res->M, res->K, res->ng, res->gs, res->is_awq);
  auto tk1 = std::chrono::steady_clock::now();
  if (!ok) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  auto tm0 = std::chrono::steady_clock::now();
  const bool mok = g_api.cudaMemcpy(y, g_dy, sizeof(float) * static_cast<size_t>(W.M),
                                    kCudaMemcpyD2H) == kCudaSuccess;
  auto tm1 = std::chrono::steady_clock::now();
  g_prof_h2d_us += 0;  // accumulate later
  g_prof_kernel_us += std::chrono::duration<double, std::micro>(tk1 - tk0).count();
  g_prof_d2h_us += std::chrono::duration<double, std::micro>(tm1 - tm0).count();
  g_prof_calls++;
  return mok;
}

bool try_gemm_int4_batch(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  if (!g_enabled || !X || !Y || n <= 0) return false;
  if (n == 1) {
    const Int4Resident* res = nullptr;
    bool x_uploaded = false;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      res = ensure_int4_resident(W);
      const bool use_jit = res && jit_available();
      if (!use_jit) {
        const float* dW = ensure_int4_device(W);
        if (!dW) return false;
        return gemm_dev(dW, X, Y, W.M, W.K);
      }
      if (!ensure_xy(W.M, W.K, 1)) return false;
      if (g_api.cudaMemcpy(g_dx, X, sizeof(float) * static_cast<size_t>(W.K),
                           kCudaMemcpyH2D) != kCudaSuccess)
        return false;
      x_uploaded = true;
    }
    if (!x_uploaded) return false;
    const bool ok = jit_gemv_int4(static_cast<const uint8_t*>(res->d_qweight),
                                  static_cast<const uint16_t*>(res->d_scales),
                                  static_cast<const uint16_t*>(res->d_zeros),
                                  reinterpret_cast<const float*>(g_dx),
                                  reinterpret_cast<float*>(g_dy), res->M, res->K, res->ng,
                                  res->gs, res->is_awq);
    if (!ok) return false;
    std::lock_guard<std::mutex> lock(g_mu);
    return g_api.cudaMemcpy(Y, g_dy, sizeof(float) * static_cast<size_t>(W.M),
                            kCudaMemcpyD2H) == kCudaSuccess;
  }

  // n>1 策略：
  // - 短 prefill：cuBLAS SGEMM（小 batch 远快于逐 token 风格的 JIT gemm_int4）
  // - 长 prefill：优先 INT4 resident JIT（避免再克隆 FP32 权重导致 OOM→回落 CPU）
  constexpr int kLongPrefillN = 64;

  auto try_int4_jit_batch = [&]() -> bool {
    const Int4Resident* res = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      res = ensure_int4_resident(W);
    }
    if (!res || !jit_available()) return false;
    const int M = res->M;
    const int K = res->K;
    constexpr size_t kMaxFloats = 64ull << 20;  // ~256MiB per X/Y buffer
    int chunk = n;
    auto fits = [&](int c) {
      return static_cast<size_t>(c) * static_cast<size_t>(K) <= kMaxFloats &&
             static_cast<size_t>(c) * static_cast<size_t>(M) <= kMaxFloats;
    };
    while (chunk > 1 && !fits(chunk)) chunk = (chunk + 1) / 2;
    if (chunk < 1) chunk = 1;

    for (int b0 = 0; b0 < n; b0 += chunk) {
      const int c = chunk < (n - b0) ? chunk : (n - b0);
      const float* Xp = X + static_cast<size_t>(b0) * K;
      float* Yp = Y + static_cast<size_t>(b0) * M;
      {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!ensure_xy(M, K, c)) return false;
        if (g_api.cudaMemcpy(g_dx, Xp, sizeof(float) * static_cast<size_t>(c) * K,
                             kCudaMemcpyH2D) != kCudaSuccess)
          return false;
      }
      if (!jit_gemm_int4(static_cast<const uint8_t*>(res->d_qweight),
                         static_cast<const uint16_t*>(res->d_scales),
                         static_cast<const uint16_t*>(res->d_zeros),
                         reinterpret_cast<const float*>(g_dx), reinterpret_cast<float*>(g_dy), M,
                         K, c, res->ng, res->gs, res->is_awq))
        return false;
      std::lock_guard<std::mutex> lock(g_mu);
      if (g_api.cudaMemcpy(Yp, g_dy, sizeof(float) * static_cast<size_t>(c) * M,
                           kCudaMemcpyD2H) != kCudaSuccess)
        return false;
    }
    return true;
  };

  auto try_cublas_fp32_batch = [&]() -> bool {
    std::lock_guard<std::mutex> lock(g_mu);
    const float* dW = ensure_int4_device(W);
    if (!dW) return false;
    return gemm_dev_batch(dW, X, n, Y, W.M, W.K);
  };

  if (n >= kLongPrefillN) {
    if (try_int4_jit_batch()) return true;
    if (try_cublas_fp32_batch()) return true;
    return false;
  }
  // 短序列：先 cuBLAS，再 INT4 JIT（显存不够装 FP32 副本时）
  if (try_cublas_fp32_batch()) return true;
  if (try_int4_jit_batch()) return true;
  return false;
}

// Fused multi-GEMV: up to 4 weight views sharing same x. One H2D + one kernel launch + N D2H.
// Returns false if any weight not resident or JIT unavailable (caller falls back to sequential).
bool try_gemm_int4_multi(const float* x, const qlwc::Int4View* const* Ws, float* const* ys, int n) {
  if (!g_enabled || !x || !Ws || !ys || n < 2 || n > 4) return false;
  if (!jit_available()) return false;
  const Int4Resident* res[4] = {};
  int total_m = 0;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    for (int i = 0; i < n; ++i) {
      res[i] = ensure_int4_resident(*Ws[i]);
      if (!res[i]) return false;
      total_m += res[i]->M;
    }
    // All must share same K, ng, gs, is_awq
    for (int i = 1; i < n; ++i) {
      if (res[i]->K != res[0]->K || res[i]->ng != res[0]->ng ||
          res[i]->gs != res[0]->gs || res[i]->is_awq != res[0]->is_awq)
        return false;
    }
    if (!ensure_xy(total_m, res[0]->K, 1)) return false;
    if (g_api.cudaMemcpy(g_dx, x, sizeof(float) * static_cast<size_t>(res[0]->K),
                         kCudaMemcpyH2D) != kCudaSuccess) return false;
  }
  // Build params for gemv_multi4_int4
  void* fn = get_jit_kernel(kGemvInt4Src, "gemv_multi4_int4");
  if (!fn) return false;
  const uint8_t* q[4] = {}; const uint16_t* s[4] = {}; const uint16_t* z[4] = {};
  float* ydev[4] = {}; int m[4] = {};
  // We write all outputs into g_dy (contiguous), then D2H slices
  int offset = 0;
  for (int i = 0; i < n; ++i) {
    q[i] = static_cast<const uint8_t*>(res[i]->d_qweight);
    s[i] = static_cast<const uint16_t*>(res[i]->d_scales);
    z[i] = static_cast<const uint16_t*>(res[i]->d_zeros);
    ydev[i] = reinterpret_cast<float*>(g_dy) + offset;
    m[i] = res[i]->M;
    offset += res[i]->M;
  }
  // Fill unused slots with task 0 (m=0 so they produce no rows)
  for (int i = n; i < 4; ++i) { q[i] = q[0]; s[i] = s[0]; z[i] = z[0]; ydev[i] = ydev[0]; m[i] = 0; }
  int K = res[0]->K, ng = res[0]->ng, gs = res[0]->gs;
  int is_awq_i = res[0]->is_awq ? 1 : 0;
  const float* dx = reinterpret_cast<const float*>(g_dx);
  void* params[] = {
    &q[0], &s[0], &z[0], &ydev[0], &m[0],
    &q[1], &s[1], &z[1], &ydev[1], &m[1],
    &q[2], &s[2], &z[2], &ydev[2], &m[2],
    &q[3], &s[3], &z[3], &ydev[3], &m[3],
    &n, &K, &ng, &gs, &is_awq_i, &dx
  };
  constexpr int RPB = 8;
  const int blocks = (total_m + RPB - 1) / RPB;
  const unsigned shmem = sizeof(unsigned short) * RPB * ng * 2;
  if (!jit_launch(fn, static_cast<unsigned>(blocks), 1, 1, RPB * 32, 1, 1, shmem, params))
    return false;
  // D2H: copy each task's slice from g_dy to host
  std::lock_guard<std::mutex> lock(g_mu);
  int off = 0;
  for (int i = 0; i < n; ++i) {
    if (g_api.cudaMemcpy(ys[i], reinterpret_cast<float*>(g_dy) + off,
                         sizeof(float) * static_cast<size_t>(m[i]),
                         kCudaMemcpyD2H) != kCudaSuccess) return false;
    off += m[i];
  }
  return true;
}

// GPU gated_delta_recurrent: state persists on device between calls.
// q/k: [n_heads, dk], v: [n_heads, dv], g/beta: [n_heads], state: [n_heads, dk, dv], out: [n_heads, dv]
bool try_attn_prefill(const float* q, const float* k, const float* v, float* out, int seq,
                      int n_heads, int n_kv_heads, int head_dim, float scale) {
  if (!g_enabled || !q || !k || !v || !out) return false;
  {
    const char* e = std::getenv("LLMOC_GPU_ATTN");
    if (e && e[0] == '0') return false;  // A/B: force CPU attn
  }
  if (seq <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) return false;
  if (n_heads % n_kv_heads != 0) return false;
  if (!jit_available()) return false;
  // scores[seq] in shared memory; default limit ~48KiB
  const size_t shmem = sizeof(float) * static_cast<size_t>(seq);
  if (shmem > 48ull * 1024ull) return false;

  void* fn = get_jit_kernel(kAttnPrefillSrc, "attn_prefill_naive");
  if (!fn) return false;

  const size_t qb = sizeof(float) * static_cast<size_t>(seq) * n_heads * head_dim;
  const size_t kb = sizeof(float) * static_cast<size_t>(seq) * n_kv_heads * head_dim;
  const size_t vb = kb;
  const size_t ob = qb;

  auto ensure_buf = [&](void*& p, size_t& cap, size_t need) -> bool {
    std::lock_guard<std::mutex> lock(g_mu);
    if (need <= cap && p) return true;
    if (p) {
      g_api.cudaFree(p);
      p = nullptr;
      cap = 0;
    }
    if (g_api.cudaMalloc(&p, need) != kCudaSuccess) {
      p = nullptr;
      return false;
    }
    cap = need;
    return true;
  };
  if (!ensure_buf(g_attn_q, g_attn_q_bytes, qb)) return false;
  if (!ensure_buf(g_attn_k, g_attn_k_bytes, kb)) return false;
  if (!ensure_buf(g_attn_v, g_attn_v_bytes, vb)) return false;
  if (!ensure_buf(g_attn_o, g_attn_o_bytes, ob)) return false;

  if (!h2d(g_attn_q, q, qb)) return false;
  if (!h2d(g_attn_k, k, kb)) return false;
  if (!h2d(g_attn_v, v, vb)) return false;

  float scale_mut = scale;
  int seq_i = seq, nh = n_heads, nkv = n_kv_heads, hd = head_dim;
  void* dq = g_attn_q;
  void* dk = g_attn_k;
  void* dv = g_attn_v;
  void* dout = g_attn_o;
  void* params[] = {&dq, &dk, &dv, &dout, &seq_i, &nh, &nkv, &hd, &scale_mut};
  if (!jit_launch(fn, static_cast<unsigned>(seq), static_cast<unsigned>(n_heads), 1, 256, 1, 1,
                  static_cast<unsigned>(shmem), params))
    return false;
  return d2h(out, g_attn_o, ob);
}

bool try_gated_delta_gpu(const float* q, const float* k, const float* v, const float* g,
                         const float* beta, float* state, float* out,
                         int n_heads, int dk, int dv) {
  if (!g_enabled || !jit_available()) return false;
  if (!q || !k || !v || !g || !beta || !state || !out) return false;
  const size_t state_bytes = sizeof(float) * n_heads * dk * dv;
  const size_t io_bytes = sizeof(float) * (n_heads * dk * 2 + n_heads * dv + n_heads * 2 + n_heads * dv);
  std::lock_guard<std::mutex> lock(g_mu);
  // Ensure device state buffer for this host state pointer
  auto it = g_gdn_state.find(state);
  if (it == g_gdn_state.end()) {
    void* d_state_v = nullptr;
    if (g_api.cudaMalloc(&d_state_v, state_bytes) != kCudaSuccess) return false;
    // Zero-init via H2D of a zero buffer
    std::vector<float> zeros(state_bytes / sizeof(float), 0.f);
    if (g_api.cudaMemcpy(d_state_v, zeros.data(), state_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(d_state_v); return false;
    }
    g_gdn_state[state] = static_cast<float*>(d_state_v);
    it = g_gdn_state.find(state);
  }
  float* d_state = it->second;
  // Ensure scratch buffer for q/k/v/g/beta/out
  if (io_bytes > g_gdn_buf_cap) {
    if (g_gdn_buf) g_api.cudaFree(g_gdn_buf);
    void* buf_v = nullptr;
    if (g_api.cudaMalloc(&buf_v, io_bytes) != kCudaSuccess) { g_gdn_buf = nullptr; return false; }
    g_gdn_buf = static_cast<float*>(buf_v);
    g_gdn_buf_cap = io_bytes;
  }
  // Layout in g_gdn_buf: [q | k | v | g | beta | out]
  float* d_q = g_gdn_buf;
  float* d_k = d_q + n_heads * dk;
  float* d_v = d_k + n_heads * dk;
  float* d_g = d_v + n_heads * dv;
  float* d_beta = d_g + n_heads;
  float* d_out = d_beta + n_heads;
  // H2D uploads
  if (g_api.cudaMemcpy(d_q, q, sizeof(float)*n_heads*dk, kCudaMemcpyH2D) != kCudaSuccess) return false;
  if (g_api.cudaMemcpy(d_k, k, sizeof(float)*n_heads*dk, kCudaMemcpyH2D) != kCudaSuccess) return false;
  if (g_api.cudaMemcpy(d_v, v, sizeof(float)*n_heads*dv, kCudaMemcpyH2D) != kCudaSuccess) return false;
  if (g_api.cudaMemcpy(d_g, g, sizeof(float)*n_heads, kCudaMemcpyH2D) != kCudaSuccess) return false;
  if (g_api.cudaMemcpy(d_beta, beta, sizeof(float)*n_heads, kCudaMemcpyH2D) != kCudaSuccess) return false;
  // Launch kernel
  void* fn = get_jit_kernel(kGdnSrc, "gated_delta_kernel");
  if (!fn) return false;
  float scale = 1.f / sqrtf((float)dk);
  void* params[] = {&d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_out, &dk, &dv, &scale};
  if (!jit_launch(fn, n_heads, 1, 1, dv, 1, 1, 0, params)) return false;
  // D2H output
  if (g_api.cudaMemcpy(out, d_out, sizeof(float)*n_heads*dv, kCudaMemcpyD2H) != kCudaSuccess) return false;
  return true;
}

bool prefetch_int4_weight(const qlwc::Int4View& W) {
  if (!g_enabled) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (ensure_int4_resident(W)) return true;
  return ensure_int4_device(W) != nullptr;
}

bool try_gemm_awq(const float* x, const AwqView& W, float* y) {
  if (!g_enabled || !x || !y) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  const float* dW = ensure_awq_device(W);
  if (!dW) return false;
  return gemm_dev(dW, x, y, W.M, W.K);
}

bool prefetch_awq_weight(const AwqView& W) {
  if (!g_enabled) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return ensure_awq_device(W) != nullptr;
}

bool try_gemm_nvfp4(const float* x, const Nvfp4View& W, float* y) {
  if (!g_enabled || !x || !y) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  const float* dW = ensure_nvfp4_device(W);
  if (!dW) return false;
  return gemm_dev(dW, x, y, W.M, W.K);
}

bool prefetch_nvfp4_weight(const Nvfp4View& W) {
  if (!g_enabled) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return ensure_nvfp4_device(W) != nullptr;
}

void log_status() {
  LOG_INFO("hal.cuda: %s enabled=%d used=%.2fGiB budget=%.2fGiB", g_status.c_str(),
           g_enabled ? 1 : 0, g_used / double(1ull << 30), g_budget / double(1ull << 30));
  if (g_prof_calls > 0) {
    LOG_INFO(
        "hal.cuda.prof: calls=%llu kernel_avg=%.1fus d2h_avg=%.1fus (per-call wall; H2D hidden in kernel span)",
        static_cast<unsigned long long>(g_prof_calls),
        g_prof_kernel_us / static_cast<double>(g_prof_calls),
        g_prof_d2h_us / static_cast<double>(g_prof_calls));
  }
}

}  // namespace llmoc::hal::cuda
