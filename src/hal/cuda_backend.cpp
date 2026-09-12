// llm-on-cpu :: hal/cuda_backend.cpp — M5 dynamic CUDA (no nvcc)
#include "hal/cuda_backend.h"

#include <algorithm>
#include <atomic>
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
constexpr int kCudaMemcpyD2D = 3;
constexpr int kCublasOpN = 0;
constexpr int kCublasOpT = 1;

using cudaMalloc_t = int (*)(void**, size_t);
using cudaFree_t = int (*)(void*);
using cudaMemcpy_t = int (*)(void*, const void*, size_t, int);
using cudaGetDeviceCount_t = int (*)(int*);
using cudaSetDevice_t = int (*)(int);
using cudaGetDeviceProperties_t = int (*)(void*, int);
using cudaMemGetInfo_t = int (*)(size_t*, size_t*);
using cudaHostRegister_t = int (*)(void*, size_t, unsigned);
using cudaHostUnregister_t = int (*)(void*);
constexpr unsigned kCudaHostRegisterPortable = 1u;
using cublasCreate_t = int (*)(void**);
using cublasDestroy_t = int (*)(void*);
using cublasSgemm_t = int (*)(void*, int, int, int, int, int, const float*, const float*, int,
                              const float*, int, const float*, float*, int);
// Mixed-precision GEMM (BF16/FP16 A × FP32 x → FP32 y). Optional; falls back to tiled path.
using cublasGemmEx_t = int (*)(void*, int, int, int, int, int, const void*, const void*, int, int,
                               const void*, int, int, const void*, void*, int, int, int, int);

constexpr int kCudaR_32F = 0;
constexpr int kCudaR_16F = 2;
constexpr int kCudaR_16BF = 14;
constexpr int kCublasCompute32F = 68;
constexpr int kCublasGemmDefault = -1;

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
  cudaMemGetInfo_t cudaMemGetInfo = nullptr;
  cudaHostRegister_t cudaHostRegister = nullptr;
  cudaHostUnregister_t cudaHostUnregister = nullptr;
  cublasCreate_t cublasCreate = nullptr;
  cublasDestroy_t cublasDestroy = nullptr;
  cublasSgemm_t cublasSgemm = nullptr;
  cublasGemmEx_t cublasGemmEx = nullptr;

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

// Packed BF16/F16 weights on device (Path A S2 — half the VRAM of FP32 inflate).
struct W16Pack {
  void* d_w = nullptr;  // uint16_t[M*K]
  int M = 0;
  int K = 0;
  bool is_f16 = false;
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
  int awq_zp = 7;  // from Int4View.awq_zp; GPTQ ignores
  bool pinned = false;  // warm attn/shared: MoE expert LRU must not evict
  size_t bytes = 0;
  uint64_t last_use = 0;
};

Api g_api;
std::mutex g_mu;
bool g_probed = false;
bool g_probe_ok = false;
bool g_enabled = false;
bool g_resident = false;
size_t g_resident_reserve = 0;  // accounting hold so weight uploads don't starve GDN workspace
std::string g_status = "off";
size_t g_budget = 0;
size_t g_used = 0;
void* g_cublas = nullptr;
void* g_dx = nullptr;
void* g_dy = nullptr;
int g_cap_k = 0;
int g_cap_m = 0;
int g_cap_n = 0;  // batch columns for X/Y
const float* g_sticky_x = nullptr;  // skip repeat H2D of same host x when resident_gpu
int g_sticky_k = 0;
std::unordered_map<const void*, CacheEntry> g_cache;
std::unordered_map<const void*, W16Pack> g_w16_pack;
// Leave headroom for workspace / KV / act (Path A S2).
// Only enforced for large W16 packs (lm_head); small out/a/b use FP32+cublas.
constexpr size_t kVramHeadroom = 1536ull << 20;  // 1.5 GiB
constexpr int kW16PackMinRows = 65536;           // lm_head-scale → pack; else FP32 inflate
constexpr int kW16TileRows = 2048;               // FP32 inflate tile for vocab GEMV via cublas
float* g_w16_tile = nullptr;                     // device FP32[tile_rows * K]
int g_w16_tile_cap = 0;                          // floats capacity
float* g_dequant_scratch = nullptr;              // reusable INT4→FP32 for prefill cublas
size_t g_dequant_scratch_cap = 0;
std::string g_act_lin_last_err;
std::unordered_map<const void*, Int4Resident> g_int4_cache;
uint64_t g_lru_tick = 0;
std::unordered_map<void*, void*> g_jit_modules;  // CUfunction -> CUmodule (JIT 句柄, disable 时卸载)
std::unordered_map<std::string, void*> g_jit_kernels;  // kernel 名→CUfunction 缓存(disable 时清空)

// GDN device state: host state ptr → device state ptr (persists between decode steps)
std::unordered_map<const float*, float*> g_gdn_state;
float* g_gdn_buf = nullptr;   // device scratch for q/k/v/g/beta/out uploads
size_t g_gdn_buf_cap = 0;
// Tensor-core prefill GEMM fp16 scratch
unsigned short* g_tc_w_f16 = nullptr;  // W fp16 [M,K]
size_t g_tc_w_f16_cap = 0;
unsigned short* g_tc_x_f16 = nullptr;  // X fp16 [n,K]
size_t g_tc_x_f16_cap = 0;
// 单 query decode attention buffers
float* g_dec_q = nullptr; size_t g_dec_q_cap = 0;
float* g_dec_k = nullptr; size_t g_dec_k_cap = 0;
float* g_dec_v = nullptr; size_t g_dec_v_cap = 0;
float* g_dec_o = nullptr; size_t g_dec_o_cap = 0;
uint64_t g_gdn_ok = 0;
uint64_t g_gdn_fail = 0;
uint64_t g_act_ffn_ok = 0;
uint64_t g_act_ffn_try = 0;
uint64_t g_act_lm_ok = 0;
std::string g_gdn_last_err;

// Resident MLP decode scratch (device)
float* g_mlp_x = nullptr;
float* g_mlp_norm = nullptr;
float* g_mlp_g = nullptr;
float* g_mlp_u = nullptr;
float* g_mlp_mid = nullptr;
float* g_mlp_down = nullptr;
float* g_mlp_core = nullptr;  // GDN core / attn intermediate
uint16_t* g_mlp_ln = nullptr;
int g_mlp_cap_h = 0;
int g_mlp_cap_i = 0;
int g_mlp_cap_core = 0;

// Cross-layer decode residual on device
float* g_act_h = nullptr;
int g_act_h_dim = 0;
bool g_act_valid = false;

// Linear-attn decode workspace (Path A: stay on device)
float* g_lin_ws = nullptr;
size_t g_lin_ws_cap = 0;
std::unordered_map<const float*, float*> g_conv_state;  // host conv ptr → device state
std::unordered_map<const float*, float*> g_conv_w_dev;   // host conv_w → device
float* g_dwconv_io = nullptr;  // xin|xout scratch for seq dwconv
size_t g_dwconv_io_cap = 0;
std::string g_dwconv_last_err;
uint64_t g_dwconv_ok = 0;
uint64_t g_dwconv_fail = 0;
uint64_t g_act_lin_ok = 0;
uint64_t g_act_lin_try = 0;
uint64_t g_act_full_ok = 0;
uint64_t g_act_full_try = 0;

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
  g_api.cublasGemmEx = reinterpret_cast<cublasGemmEx_t>(sym(g_api.cublas, "cublasGemmEx"));
  if (!g_api.cudaMalloc || !g_api.cudaMemcpy || !g_api.cudaGetDeviceCount || !g_api.cublasCreate ||
      !g_api.cublasSgemm) {
    err = "missing CUDA symbols";
    return false;
  }
  g_api.cudaGetDeviceProperties =
      reinterpret_cast<cudaGetDeviceProperties_t>(sym(g_api.cudart, "cudaGetDeviceProperties"));
  g_api.cudaMemGetInfo = reinterpret_cast<cudaMemGetInfo_t>(sym(g_api.cudart, "cudaMemGetInfo"));
  g_api.cudaHostRegister =
      reinterpret_cast<cudaHostRegister_t>(sym(g_api.cudart, "cudaHostRegister"));
  g_api.cudaHostUnregister =
      reinterpret_cast<cudaHostUnregister_t>(sym(g_api.cudart, "cudaHostUnregister"));

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
  // Use size_t to avoid overflow; reject absurd batches early.
  const size_t need_k = static_cast<size_t>(K) * static_cast<size_t>(n);
  const size_t need_m = static_cast<size_t>(M) * static_cast<size_t>(n);
  constexpr size_t kMaxFloats = 64ull << 20;  // ~256MiB
  if (need_k == 0 || need_m == 0 || need_k > kMaxFloats || need_m > kMaxFloats) return false;

  if (static_cast<int>(need_k) > g_cap_k || !g_dx) {
    if (g_dx) {
      g_api.cudaFree(g_dx);
      g_dx = nullptr;
    }
    g_cap_k = 0;
    g_sticky_x = nullptr;
    g_sticky_k = 0;
    if (g_api.cudaMalloc(&g_dx, sizeof(float) * need_k) != kCudaSuccess) {
      g_dx = nullptr;
      return false;
    }
    g_cap_k = static_cast<int>(need_k);
  }
  if (static_cast<int>(need_m) > g_cap_m || !g_dy) {
    if (g_dy) {
      g_api.cudaFree(g_dy);
      g_dy = nullptr;
    }
    g_cap_m = 0;
    if (g_api.cudaMalloc(&g_dy, sizeof(float) * need_m) != kCudaSuccess) {
      g_dy = nullptr;
      return false;
    }
    g_cap_m = static_cast<int>(need_m);
  }
  g_cap_n = n;
  return true;
}

// Upload host x→g_dx.
// NOTE: Do NOT skip H2D based on pointer equality. Decode scratch (normed/mid/…) is reused
// in-place across layers/experts; same address with new contents caused stale GPU x and
// sticky garbage tokens (ici/endah/…) on MoE INT4 fallback paths.
bool upload_x_sticky(const float* x, int K) {
  if (g_api.cudaMemcpy(g_dx, x, sizeof(float) * static_cast<size_t>(K), kCudaMemcpyH2D) !=
      kCudaSuccess)
    return false;
  g_sticky_x = x;
  g_sticky_k = K;
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
    // LRU 腾出空间（跳过 pinned：attn/shared/router 热权重）
    std::vector<std::pair<uint64_t, const void*>> items;
    items.reserve(g_int4_cache.size());
    for (auto& kv : g_int4_cache) {
      if (kv.second.pinned) continue;
      items.push_back({kv.second.last_use, kv.first});
    }
    std::sort(items.begin(), items.end());
    for (auto& p : items) {
      if (g_used + total <= g_budget) break;
      auto it2 = g_int4_cache.find(p.second);
      if (it2 == g_int4_cache.end() || it2->second.pinned) continue;
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
  e.awq_zp = W.awq_zp > 0 ? W.awq_zp : qlwc::kLocalAwqSymZero;
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

void set_vram_budget(size_t bytes) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_budget = bytes;
}

bool device_mem_info(size_t* free_bytes, size_t* total_bytes) {
  if (free_bytes) *free_bytes = 0;
  if (total_bytes) *total_bytes = 0;
  std::string err;
  if (!load_apis(err)) return false;
  if (!g_api.cudaMemGetInfo) return false;
  if (g_api.cudaSetDevice) (void)g_api.cudaSetDevice(0);
  size_t free_b = 0, total_b = 0;
  if (g_api.cudaMemGetInfo(&free_b, &total_b) != kCudaSuccess) return false;
  if (free_bytes) *free_bytes = free_b;
  if (total_bytes) *total_bytes = total_b;
  return true;
}

bool host_register(void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return false;
  std::string err;
  if (!load_apis(err)) return false;
  if (!g_api.cudaHostRegister) return false;
  // Need a device context; enable() or setDevice may not have run yet.
  if (g_api.cudaSetDevice) (void)g_api.cudaSetDevice(0);
  return g_api.cudaHostRegister(ptr, bytes, kCudaHostRegisterPortable) == kCudaSuccess;
}

bool host_unregister(void* ptr) {
  if (!ptr) return false;
  if (!g_api.cudaHostUnregister) return false;
  return g_api.cudaHostUnregister(ptr) == kCudaSuccess;
}

bool resident_gpu_enabled() { return g_resident && g_enabled; }

bool try_enable_resident_gpu(size_t workspace_bytes) {
  const char* env = std::getenv("LLMOC_RESIDENT_GPU");
  if (env && env[0] == '0' && env[1] == '\0') {
    g_resident = false;
    g_status = "resident_gpu forced off (LLMOC_RESIDENT_GPU=0)";
    return false;
  }
  const bool force = env && env[0] == '1' && env[1] == '\0';
  if (!g_enabled) {
    g_status = "resident_gpu: cuda not enabled";
    return false;
  }
  if (!jit_available()) {
    g_status = "resident_gpu: jit unavailable";
    return false;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  size_t need_add = 0;
  if (workspace_bytes > g_resident_reserve) need_add = workspace_bytes - g_resident_reserve;
  if (!force && g_budget > 0 && g_used + need_add > g_budget) {
    g_status = "resident_gpu: insufficient VRAM headroom";
    return false;
  }
  if (need_add > 0) {
    g_used += need_add;
    g_resident_reserve = workspace_bytes;
  }
  g_resident = true;
  g_status = "resident_gpu on";
  return true;
}

void log_resident_stats() {
  LOG_INFO("resident_gdn: ok=%llu fail=%llu last_err=%s act=%d ffn=%llu/%llu lm=%llu lin=%llu/%llu full=%llu/%llu lin_err=%s",
           static_cast<unsigned long long>(g_gdn_ok),
           static_cast<unsigned long long>(g_gdn_fail),
           g_gdn_last_err.empty() ? "-" : g_gdn_last_err.c_str(), g_act_valid ? 1 : 0,
           static_cast<unsigned long long>(g_act_ffn_ok),
           static_cast<unsigned long long>(g_act_ffn_try),
           static_cast<unsigned long long>(g_act_lm_ok),
           static_cast<unsigned long long>(g_act_lin_ok),
           static_cast<unsigned long long>(g_act_lin_try),
           static_cast<unsigned long long>(g_act_full_ok),
           static_cast<unsigned long long>(g_act_full_try),
           g_act_lin_last_err.empty() ? "-" : g_act_lin_last_err.c_str());
}

bool decode_act_begin(const float* h_host, int H) {
  if (!g_enabled || !g_resident || !h_host || H <= 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_act_h || g_act_h_dim < H) {
    if (g_act_h) g_api.cudaFree(g_act_h);
    g_act_h = nullptr;
    g_act_h_dim = 0;
    void* v = nullptr;
    if (g_api.cudaMalloc(&v, sizeof(float) * static_cast<size_t>(H)) != kCudaSuccess) return false;
    g_act_h = static_cast<float*>(v);
    g_act_h_dim = H;
  }
  if (g_api.cudaMemcpy(g_act_h, h_host, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyH2D) !=
      kCudaSuccess) {
    g_act_valid = false;
    return false;
  }
  g_act_valid = true;
  return true;
}

bool decode_act_sync_to_host(float* h_host, int H) {
  if (!g_enabled || !g_act_valid || !g_act_h || !h_host || H <= 0 || H > g_act_h_dim) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(h_host, g_act_h, sizeof(float) * static_cast<size_t>(H),
                          kCudaMemcpyD2H) == kCudaSuccess;
}

bool decode_act_load_from_host(const float* h_host, int H) {
  return decode_act_begin(h_host, H);
}

float* decode_act_ptr() { return (g_act_valid && g_act_h) ? g_act_h : nullptr; }
bool decode_act_valid() { return g_act_valid && g_act_h != nullptr; }
void decode_act_invalidate() { g_act_valid = false; }

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
  g_status = g_api.cublasGemmEx ? "cuda+cublas+gemmEx enabled" : "cuda+cublas enabled";

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
  for (auto& kv : g_w16_pack) {
    if (kv.second.d_w) g_api.cudaFree(kv.second.d_w);
  }
  g_w16_pack.clear();
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
  if (g_tc_w_f16) g_api.cudaFree(g_tc_w_f16);
  if (g_tc_x_f16) g_api.cudaFree(g_tc_x_f16);
  g_tc_w_f16 = g_tc_x_f16 = nullptr;
  g_tc_w_f16_cap = g_tc_x_f16_cap = 0;
  if (g_dec_q) g_api.cudaFree(g_dec_q);
  if (g_dec_k) g_api.cudaFree(g_dec_k);
  if (g_dec_v) g_api.cudaFree(g_dec_v);
  if (g_dec_o) g_api.cudaFree(g_dec_o);
  g_dec_q = g_dec_k = g_dec_v = g_dec_o = nullptr;
  g_dec_q_cap = g_dec_k_cap = g_dec_v_cap = g_dec_o_cap = 0;
  g_gdn_ok = g_gdn_fail = 0;
  g_act_ffn_ok = g_act_ffn_try = g_act_lm_ok = 0;
  g_act_lin_ok = g_act_lin_try = 0;
  g_act_full_ok = g_act_full_try = 0;
  g_act_lin_last_err.clear();
  g_gdn_last_err.clear();
  g_dwconv_last_err.clear();
  g_dwconv_ok = g_dwconv_fail = 0;
  auto free_f = [&](float*& p) {
    if (p) {
      g_api.cudaFree(p);
      p = nullptr;
    }
  };
  free_f(g_mlp_x);
  free_f(g_mlp_norm);
  free_f(g_mlp_g);
  free_f(g_mlp_u);
  free_f(g_mlp_mid);
  free_f(g_mlp_down);
  free_f(g_mlp_core);
  if (g_mlp_ln) {
    g_api.cudaFree(g_mlp_ln);
    g_mlp_ln = nullptr;
  }
  g_mlp_cap_h = g_mlp_cap_i = g_mlp_cap_core = 0;
  if (g_act_h) {
    g_api.cudaFree(g_act_h);
    g_act_h = nullptr;
  }
  g_act_h_dim = 0;
  g_act_valid = false;
  for (auto& kv : g_conv_state) {
    if (kv.second) g_api.cudaFree(kv.second);
  }
  g_conv_state.clear();
  for (auto& kv : g_conv_w_dev) {
    if (kv.second) g_api.cudaFree(kv.second);
  }
  g_conv_w_dev.clear();
  if (g_dwconv_io) {
    g_api.cudaFree(g_dwconv_io);
    g_dwconv_io = nullptr;
    g_dwconv_io_cap = 0;
  }
  if (g_lin_ws) {
    g_api.cudaFree(g_lin_ws);
    g_lin_ws = nullptr;
    g_lin_ws_cap = 0;
  }
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
  if (g_w16_tile) {
    g_api.cudaFree(g_w16_tile);
    g_w16_tile = nullptr;
  }
  g_w16_tile_cap = 0;
  g_sticky_x = nullptr;
  g_sticky_k = 0;
  if (g_cublas && g_api.cublasDestroy) g_api.cublasDestroy(g_cublas);
  g_cublas = nullptr;
  g_used = 0;
  g_resident = false;
  g_resident_reserve = 0;
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
// awq: w=(q-awq_zp)*scale (Int4View.awq_zp); gptq: w=q*scale+zero。
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

// 多行/块的 GEMV: 每 block ROWS_PER_BLOCK 行; x 尽量进 shared(避免每行重复读 global)。
#define ROWS_PER_BLOCK 8
extern "C" __global__ void gemv_int4(
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int ng, int gs, int is_awq, int awq_zp, int use_sx) {
  const int row0 = blockIdx.x * ROWS_PER_BLOCK;
  const int tid = threadIdx.x;
  const int rb = (K + 1) >> 1;
  extern __shared__ unsigned char smem_raw[];
  unsigned short* s_scales = (unsigned short*)smem_raw;
  const int scale_words = ROWS_PER_BLOCK * ng * (is_awq ? 1 : 2);
  unsigned short* s_zeros = is_awq ? nullptr : s_scales + ROWS_PER_BLOCK * ng;
  float* sx = (float*)(s_scales + ((scale_words + 1) & ~1));  // 4-byte align
  for (int row_off = 0; row_off < ROWS_PER_BLOCK; ++row_off) {
    const int m = row0 + row_off;
    if (m >= M) break;
    for (int g = tid; g < ng; g += blockDim.x)
      s_scales[row_off * ng + g] = scales[m * ng + g];
    if (!is_awq) {
      for (int g = tid; g < ng; g += blockDim.x)
        s_zeros[row_off * ng + g] = zeros[m * ng + g];
    }
  }
  if (use_sx) {
    for (int i = tid; i < K; i += blockDim.x) sx[i] = x[i];
  }
  __syncthreads();
  const float* xr = use_sx ? sx : x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  if (warp_id >= ROWS_PER_BLOCK) return;
  const int m = row0 + warp_id;
  if (m >= M) return;
  const unsigned char* qrow = qweight + (size_t)m * (size_t)rb;
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 32 * 8) {
    const int g = k / gs;
    const float s = f16_to_f32_dev(s_scales[warp_id * ng + g]);
    const float z = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + g]);
    const int kp = k >> 1;
    if (kp + 3 < rb) {
      const uchar4 v = *reinterpret_cast<const uchar4*>(qrow + kp);
      const float x0 = xr[k + 0], x1 = xr[k + 1], x2 = xr[k + 2], x3 = xr[k + 3];
      const float x4 = xr[k + 4], x5 = xr[k + 5], x6 = xr[k + 6], x7 = xr[k + 7];
      const float w0 = ((float)((v.x & 0xF) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w1 = ((float)((v.x >> 4) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w2 = ((float)((v.y & 0xF) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w3 = ((float)((v.y >> 4) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w4 = ((float)((v.z & 0xF) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w5 = ((float)((v.z >> 4) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w6 = ((float)((v.w & 0xF) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      const float w7 = ((float)((v.w >> 4) - (is_awq ? awq_zp : 0))) * s + (is_awq ? 0.f : z);
      acc += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + x5 * w5 + x6 * w6 + x7 * w7;
    } else {
      for (int kk = k; kk < k + 8 && kk < K; ++kk) {
        const int gg = kk / gs;
        const float ss = f16_to_f32_dev(s_scales[warp_id * ng + gg]);
        const float zz = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + gg]);
        const unsigned char b = qrow[kk >> 1];
        const int qi = (kk & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
        const float w = ((float)qi - (is_awq ? awq_zp : 0)) * ss + (is_awq ? 0.f : zz);
        acc += xr[kk] * w;
      }
    }
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[m] = acc;
}

// Prefill batch GEMM: Y[n,M] = X[n,K] @ W[M,K]^T with on-the-fly INT4 dequant.
// grid.x = ceil(M / ROWS_PER_BLOCK), grid.y = ceil(n / BT).
// Each block owns ROWS_PER_BLOCK output rows × BT batch rows.
// X is staged through shared memory in K-tiles (old kernel re-read full X per M-block
// ≈ M/8 times — multi-GB traffic for a ~MB activation).
extern "C" __global__ void gemm_int4(
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int n, int ng, int gs, int is_awq, int awq_zp) {
  constexpr int BT = 8;
  constexpr int KT = 256;  // X tile along K; BT*KT*4 ≈ 8KiB
  const int row0 = blockIdx.x * ROWS_PER_BLOCK;
  const int b0 = blockIdx.y * BT;
  if (b0 >= n) return;
  const int bn = (b0 + BT <= n) ? BT : (n - b0);
  const int tid = threadIdx.x;
  const int rb = (K + 1) >> 1;

  extern __shared__ unsigned char smem_raw[];
  unsigned short* s_scales = (unsigned short*)smem_raw;
  unsigned short* s_zeros = is_awq ? nullptr : s_scales + ROWS_PER_BLOCK * ng;
  const int scale_words = ROWS_PER_BLOCK * ng * (is_awq ? 1 : 2);
  float* sx = (float*)(s_scales + ((scale_words + 1) & ~1));

  for (int row_off = 0; row_off < ROWS_PER_BLOCK; ++row_off) {
    const int m = row0 + row_off;
    if (m >= M) break;
    for (int g = tid; g < ng; g += blockDim.x)
      s_scales[row_off * ng + g] = scales[m * ng + g];
    if (!is_awq) {
      for (int g = tid; g < ng; g += blockDim.x)
        s_zeros[row_off * ng + g] = zeros[m * ng + g];
    }
  }
  __syncthreads();

  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int m = row0 + warp_id;
  const bool active = (warp_id < ROWS_PER_BLOCK) && (m < M);
  const unsigned char* qrow = active ? (qweight + (size_t)m * (size_t)rb) : nullptr;
  const int off0 = is_awq ? awq_zp : 0;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

  for (int k0 = 0; k0 < K; k0 += KT) {
    const int kn = (k0 + KT <= K) ? KT : (K - k0);
    // Stage X[b0:b0+bn, k0:k0+kn] into sx[bi * KT + ki]
    for (int i = tid; i < bn * kn; i += blockDim.x) {
      const int bi = i / kn;
      const int ki = i - bi * kn;
      sx[bi * KT + ki] = X[(size_t)(b0 + bi) * (size_t)K + (k0 + ki)];
    }
    __syncthreads();

    if (active) {
      for (int k = lane * 8; k < kn; k += 32 * 8) {
        const int gk = k0 + k;
        const int g = gk / gs;
        const float s = f16_to_f32_dev(s_scales[warp_id * ng + g]);
        const float z = is_awq ? 0.f : f16_to_f32_dev(s_zeros[warp_id * ng + g]);
        const int kp = gk >> 1;
        float w0, w1, w2, w3, w4, w5, w6, w7;
        if (gk + 7 < K && kp + 3 < rb) {
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
            const int kk = gk + t;
            if (kk >= K || k + t >= kn) break;
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
        for (int bi = 0; bi < bn; ++bi) {
          const float* xr = sx + bi * KT;
          float partial = 0.f;
          if (k + 7 < kn) {
            partial = xr[k]*w0 + xr[k+1]*w1 + xr[k+2]*w2 + xr[k+3]*w3
                    + xr[k+4]*w4 + xr[k+5]*w5 + xr[k+6]*w6 + xr[k+7]*w7;
          } else {
            if (k + 0 < kn) partial += xr[k + 0] * w0;
            if (k + 1 < kn) partial += xr[k + 1] * w1;
            if (k + 2 < kn) partial += xr[k + 2] * w2;
            if (k + 3 < kn) partial += xr[k + 3] * w3;
            if (k + 4 < kn) partial += xr[k + 4] * w4;
            if (k + 5 < kn) partial += xr[k + 5] * w5;
            if (k + 6 < kn) partial += xr[k + 6] * w6;
            if (k + 7 < kn) partial += xr[k + 7] * w7;
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
    }
    __syncthreads();  // before next X tile overwrite
  }

  if (active) {
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

// Expand full INT4 matrix to FP32 for one-shot cuBLAS (prefill). out[M*K] row-major.
extern "C" __global__ void dequant_int4_to_f32(
    const unsigned char* __restrict__ qweight, const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros, float* __restrict__ out, int M, int K, int ng,
    int gs, int is_awq, int awq_zp) {
  const size_t n = (size_t)M * (size_t)K;
  const int rb = (K + 1) >> 1;
  const int off0 = is_awq ? awq_zp : 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const int m = (int)(i / (size_t)K);
    const int k = (int)(i % (size_t)K);
    const int g = k / gs;
    const float s = f16_to_f32_dev(scales[m * ng + g]);
    const float z = is_awq ? 0.f : f16_to_f32_dev(zeros[m * ng + g]);
    const unsigned char packed = qweight[(size_t)m * (size_t)rb + (size_t)(k >> 1)];
    const int qi = (k & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
    out[i] = ((float)(qi - off0)) * s + z;
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
    int nt, int K, int ng, int gs, int is_awq, int awq_zp,
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
      const int off = is_awq ? awq_zp : 0;
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
        const int off = is_awq ? awq_zp : 0;
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

// Prefill: one launch per head runs the full token window (no per-token launch storm).
extern "C" __global__ void gated_delta_seq_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ state,
    float* __restrict__ out, int seq, int n_heads, int dk, int dv, float scale) {
  const int h = blockIdx.x;
  const int j = threadIdx.x;
  if (h >= n_heads || j >= dv) return;
  float* st = state + (size_t)h * dk * dv;
  for (int t = 0; t < seq; ++t) {
    const float* qh = q + ((size_t)t * n_heads + h) * dk;
    const float* kh = k + ((size_t)t * n_heads + h) * dk;
    const float* vh = v + ((size_t)t * n_heads + h) * dv;
    float g_log = g[t * n_heads + h];
    float beta_t = beta[t * n_heads + h];
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
    out[((size_t)t * n_heads + h) * dv + j] = out_j;
  }
}
)CUDA";

// Tensor-core prefill GEMM: INT4 resident → device FP16 weights, cuBLAS GemmEx (FP16×FP16→FP32).
const char* kTcSrc = R"CUDA(
__device__ __forceinline__ float tc_f16_to_f32(unsigned short h) {
  unsigned int sign = (h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int man = h & 0x3FFu;
  unsigned int bits;
  if (exp == 0u) {
    if (man == 0u) bits = sign;
    else { exp = 1u; while ((man & 0x400u) == 0u) { man <<= 1; exp -= 1u; } man &= 0x3FFu; bits = sign | ((exp + 112u) << 23) | (man << 13); }
  } else if (exp == 31u) bits = sign | 0x7F800000u | (man << 13);
  else bits = sign | ((exp + 112u) << 23) | (man << 13);
  return __uint_as_float(bits);
}
__device__ __forceinline__ unsigned short tc_f32_to_f16(float f) {
  // 手动 round-to-nearest-even（不依赖 __half 类型，NVRTC 通用）
  unsigned int i = __float_as_uint(f);
  unsigned int s = (i >> 16) & 0x8000u;
  int e = (int)((i >> 23) & 0xff) - 127 + 15;
  unsigned int m = i & 0x7fffffu;
  unsigned short out;
  if (e >= 30) {
    out = (unsigned short)(s | 0x7c00u);  // inf/nan
  } else if (e <= 0) {
    if (e < -10) {
      out = (unsigned short)s;  // 下溢→0
    } else {
      m |= 0x800000u;           // 规格化尾数
      int sh = 14 - e;
      unsigned int m10 = m >> (sh + 13);
      unsigned int rem = m & ((1u << (sh + 13)) - 1u);
      if (rem > (1u << (sh + 12)) ||
          (rem == (1u << (sh + 12)) && (m10 & 1u)))
        m10++;
      out = (unsigned short)(s | m10);
    }
  } else {
    unsigned int m10 = m >> 13;
    unsigned int rem = m & 0x1fffu;
    unsigned int round_up = rem > 0x1000u || (rem == 0x1000u && (m10 & 1u));
    if (round_up) {
      m10++;
      if (m10 == 0x400u) { m10 = 0; e++; }
    }
    out = (unsigned short)(s | ((unsigned int)e << 10) | m10);
  }
  return out;
}
// W [M,K] row-major INT4 → FP16。语义与 gemv_int4 完全一致。
extern "C" __global__ void int4_to_f16(const unsigned char* __restrict__ qw,
                                       const unsigned short* __restrict__ sc,
                                       const unsigned short* __restrict__ z,
                                       unsigned short* __restrict__ w16, int M, int K, int ng,
                                       int gs, int is_awq, int awq_zp) {
  const int rb = (K + 1) >> 1;
  const size_t total = (size_t)M * K;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t m = i / K;
    const size_t k = i - m * K;
    const int g = (int)(k / gs);
    const unsigned char b = qw[m * rb + (k >> 1)];
    const int qi = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    const float s = tc_f16_to_f32(sc[m * ng + g]);
    float w;
    if (is_awq) w = (float)(qi - awq_zp) * s;
    else {
      const float zz = tc_f16_to_f32(z[m * ng + g]);
      w = (float)qi * s + zz;
    }
    w16[i] = tc_f32_to_f16(w);
  }
}
extern "C" __global__ void f32_to_f16_buf(const float* __restrict__ in,
                                          unsigned short* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tc_f32_to_f16(in[i]);
}
)CUDA";

// 单 query flash decode attention：每个 query head 一个 block，seq 并行算分数+并行 softmax+V。
// k_cache/v_cache 布局 [hkv*stride + t]*hd，stride = cache_cap(max_seq)。CPU 回退。
const char* kAttnDecodeSrc = R"CUDA(
__device__ __forceinline__ float dd_wp_max(float x) {
  for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}
__device__ __forceinline__ float dd_wp_sum(float x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
extern "C" __global__ void attn_decode_flash(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ out, int seq_len, int stride,
    int n_heads, int n_kv, int hd, float scale) {
  const int h = blockIdx.x;
  if (h >= n_heads) return;
  const int g = n_heads / n_kv, hkv = h / g;
  const int tid = threadIdx.x, nt = blockDim.x;
  extern __shared__ float sm[];
  float* sh_q = sm;                     // hd
  float* sc = sm + hd;                  // seq_len
  float* red = sm + hd + ((seq_len + 3u) & ~3u);  // reduce scratch
  const float* qh = q + (size_t)h * hd;
  for (int d = tid; d < hd; d += nt) sh_q[d] = qh[d];
  __syncthreads();
  for (int t = tid; t < seq_len; t += nt) {
    const float* kt = k_cache + ((size_t)hkv * stride + (size_t)t) * hd;
    float dot = 0.f;
    for (int d = 0; d < hd; ++d) dot += sh_q[d] * kt[d];
    sc[t] = dot * scale;
  }
  __syncthreads();
  float m = -1e30f;
  for (int t = tid; t < seq_len; t += nt) m = fmaxf(m, sc[t]);
  m = dd_wp_max(m);
  if ((tid & 31) == 0) red[tid >> 5] = m;
  __syncthreads();
  if (tid == 0) { float M = red[0]; for (int i = 1; i < nt / 32; ++i) M = fmaxf(M, red[i]); red[0] = M; }
  __syncthreads();
  m = red[0];
  float s = 0.f;
  for (int t = tid; t < seq_len; t += nt) { sc[t] = expf(sc[t] - m); s += sc[t]; }
  __syncthreads();
  s = dd_wp_sum(s);
  if ((tid & 31) == 0) red[tid >> 5] = s;
  __syncthreads();
  if (tid == 0) { float S = 0.f; for (int i = 0; i < nt / 32; ++i) S += red[i]; red[0] = S; }
  __syncthreads();
  const float inv = 1.f / red[0];
  float* oh = out + (size_t)h * hd;
  for (int d = tid; d < hd; d += nt) {
    float acc = 0.f;
    for (int t = 0; t < seq_len; ++t) {
      const float* vt = v_cache + ((size_t)hkv * stride + (size_t)t) * hd;
      acc += sc[t] * vt[d];
    }
    oh[d] = acc * inv;
  }
}
)CUDA";

// Depthwise conv+SiLU — separate from kActSrc so seq kernel JIT can't fail with the huge act TU.
const char* kDwconvSrc = R"CUDA(
extern "C" __global__ void dwconv_silu_k4(const float* __restrict__ xin, float* __restrict__ state,
                                         const float* __restrict__ w, float* __restrict__ xout,
                                         int conv_dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  float* st = state + (size_t)c * 4;
  const float* wk = w + (size_t)c * 4;
  st[3] = st[2];
  st[2] = st[1];
  st[1] = st[0];
  st[0] = xin[c];
  const float acc = st[0] * wk[3] + st[1] * wk[2] + st[2] * wk[1] + st[3] * wk[0];
  xout[c] = acc / (1.f + expf(-acc));
}

// Prefill: one thread per channel runs the full token window.
extern "C" __global__ void dwconv_silu_k4_seq(const float* __restrict__ xin, float* __restrict__ state,
                                             const float* __restrict__ w, float* __restrict__ xout,
                                             int seq, int conv_dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  float* st = state + (size_t)c * 4;
  const float* wk = w + (size_t)c * 4;
  float s0 = st[0], s1 = st[1], s2 = st[2], s3 = st[3];
  for (int t = 0; t < seq; ++t) {
    s3 = s2;
    s2 = s1;
    s1 = s0;
    s0 = xin[(size_t)t * conv_dim + c];
    const float acc = s0 * wk[3] + s1 * wk[2] + s2 * wk[1] + s3 * wk[0];
    xout[(size_t)t * conv_dim + c] = acc / (1.f + expf(-acc));
  }
  st[0] = s0;
  st[1] = s1;
  st[2] = s2;
  st[3] = s3;
}
)CUDA";

// Decode activation helpers (rmsnorm / silu× / add) for resident MLP path.
const char* kActSrc = R"CUDA(
__device__ __forceinline__ float w16_to_f32(unsigned short h, int is_f16) {
  if (is_f16) {
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int man = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
      if (man == 0u) bits = sign;
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
    return __int_as_float(bits);
  }
  // bf16
  unsigned int bits = ((unsigned int)h) << 16;
  return __int_as_float(bits);
}

extern "C" __global__ void rmsnorm_w16(const float* __restrict__ x, const unsigned short* __restrict__ w,
                                      float* __restrict__ y, int n, float eps, int is_f16) {
  extern __shared__ float smem[];
  float* partial = smem;
  const int tid = threadIdx.x;
  float sum = 0.f;
  for (int i = tid; i < n; i += blockDim.x) {
    float v = x[i];
    sum += v * v;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  // Match CPU rmsnorm(..., one_plus_weight=true): y = x * rsqrt(mean(x^2)+eps) * (1+w)
  const float inv = rsqrtf(partial[0] / (float)n + eps);
  for (int i = tid; i < n; i += blockDim.x)
    y[i] = x[i] * inv * (1.f + w16_to_f32(w[i], is_f16));
}

extern "C" __global__ void silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                                   float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = g[i];
  y[i] = (x / (1.f + expf(-x))) * u[i];
}

extern "C" __global__ void vec_add(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = a[i] + b[i];
}

extern "C" __global__ void vec_fill(float* __restrict__ y, float v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = v;
}

extern "C" __global__ void vec_axpy(const float* __restrict__ x, float* __restrict__ y, float a,
                                   int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] += a * x[i];
}

// Pack GDN q/k (repeat nk→nv) and copy v from mixed_c.
extern "C" __global__ void gdn_pack_qkv(const float* __restrict__ mixed_c, float* __restrict__ q,
                                       float* __restrict__ k, float* __restrict__ v, int nk, int nv,
                                       int dk, int dv) {
  const int key_dim = nk * dk;
  const int value_dim = nv * dv;
  const int rep = nv / nk;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nq = nv * dk;
  if (tid < nq) {
    const int hh = tid / dk;
    const int d = tid % dk;
    const int h = hh / rep;
    q[tid] = mixed_c[h * dk + d];
    k[tid] = mixed_c[key_dim + h * dk + d];
  }
  if (tid < value_dim) v[tid] = mixed_c[2 * key_dim + tid];
}

// beta = sigmoid(b); g = -clamp(exp(A_log)) * softplus(a + dt_bias)
extern "C" __global__ void gdn_prep_gb(const float* __restrict__ b, const float* __restrict__ a,
                                      const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                      float* __restrict__ beta, float* __restrict__ g, int nv) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nv) return;
  beta[h] = 1.f / (1.f + expf(-b[h]));
  float A = expf(A_log[h]);
  if (!(A == A) || A > 1e4f) A = 1e4f;
  if (A < 1e-6f) A = 1e-6f;
  const float x = a[h] + dt_bias[h];
  // softplus
  float sp;
  if (x > 20.f) sp = x;
  else if (x < -20.f) sp = expf(x);
  else sp = logf(1.f + expf(x));
  if (!(sp == sp)) sp = 0.f;
  g[h] = -A * sp;
}

// Per-head rmsnorm_gated (CPU: scale = w, NOT 1+w).
// Weight w is SHARED across heads and has length hd (matches CPU: always pass lp.nrm base).
extern "C" __global__ void rmsnorm_gated_heads_v2(const float* __restrict__ x,
                                               const float* __restrict__ gate,
                                               const unsigned short* __restrict__ w,
                                               float* __restrict__ y, int hd, float eps,
                                               int is_f16) {
  extern __shared__ float smem[];
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const float* xh = x + (size_t)h * hd;
  const float* gh = gate + (size_t)h * hd;
  float* yh = y + (size_t)h * hd;
  float sum = 0.f;
  for (int i = tid; i < hd; i += blockDim.x) {
    float v = xh[i];
    if (!(v == v)) v = 0.f;
    sum += v * v;
  }
  smem[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / (float)hd + eps);
  for (int i = tid; i < hd; i += blockDim.x) {
    float v = xh[i];
    if (!(v == v)) v = 0.f;
    float gg = gh[i];
    if (!(gg == gg)) gg = 0.f;
    const float silu = gg / (1.f + expf(-gg));
    // Shared norm weight (not per-head): matches host rmsnorm_gated(lp.nrm, ..., dv).
    yh[i] = v * inv * w16_to_f32(w[i], is_f16) * silu;
  }
}

// GEMV with packed BF16/F16 weights (no FP32 inflate). y[M] = W[M,K] @ x[K]
// Prefer tiled cublas for vocab-scale M; this kernel remains for small M fallback.
extern "C" __global__ void gemv_w16(const unsigned short* __restrict__ W, const float* __restrict__ x,
                                    float* __restrict__ y, int M, int K, int is_f16) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  for (int m = (int)blockIdx.x; m < M; m += (int)gridDim.x) {
    const unsigned short* row = W + (size_t)m * (size_t)K;
    float sum = 0.f;
    for (int k = tid; k < K; k += blockDim.x) sum += w16_to_f32(row[k], is_f16) * x[k];
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) smem[tid] += smem[tid + s];
      __syncthreads();
    }
    if (tid == 0) {
      float o = smem[0];
      y[m] = (o == o) ? o : 0.f;
    }
    __syncthreads();
  }
}

// Expand a contiguous row tile of W16 → FP32 for cublas (lm_head path).
// out[rows, K] row-major from W[m0 : m0+rows, :].
extern "C" __global__ void w16_tile_to_f32(const unsigned short* __restrict__ W, float* __restrict__ out,
                                          int M, int K, int m0, int rows, int is_f16) {
  const size_t n = (size_t)rows * (size_t)K;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const int r = (int)(i / (size_t)K);
    const int c = (int)(i % (size_t)K);
    const int m = m0 + r;
    if (m >= M) {
      out[i] = 0.f;
      continue;
    }
    out[i] = w16_to_f32(W[(size_t)m * (size_t)K + (size_t)c], is_f16);
  }
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

// FlashPrefill-V2 风格 kernel（算子重写+均值校正稀疏）：
//  - scores 计算：每 warp 一个 K 位置、lane 按 float4 分 hd 维，warp-reduce 得点积（向量化）
//  - softmax：多线程并行 max/sum（而非单线程串行）
//  - V 累加：lane 按 float4 分 dim，逐 K 流式
//  - 稀疏跳块：softmax 后按块最大概率 < exp(-tau)*(1+mean_corr/n) 跳过 V 访存（FlashPrefill V2 动态阈值+均值校正）
//  tau=+inf（默认）→ 与 dense 完全一致；head_dim % 4 == 0 时走 float4 路径。
__device__ __forceinline__ float wp_reduce_max(float x) {
  for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}
__device__ __forceinline__ float wp_reduce_sum(float x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
__device__ __forceinline__ float blk_reduce_max(float x, float* r, int tid) {
  x = wp_reduce_max(x);
  if ((tid & 31) == 0) r[tid >> 5] = x;
  __syncthreads();
  if (tid == 0) {
    float m = r[0];
    for (int i = 1; i < 128 / 32; ++i) m = fmaxf(m, r[i]);
    r[0] = m;
  }
  __syncthreads();
  return r[0];
}
__device__ __forceinline__ float blk_reduce_sum(float x, float* r, int tid) {
  x = wp_reduce_sum(x);
  if ((tid & 31) == 0) r[tid >> 5] = x;
  __syncthreads();
  if (tid == 0) {
    float s = r[0];
    for (int i = 1; i < 128 / 32; ++i) s += r[i];
    r[0] = s;
  }
  __syncthreads();
  return r[0];
}
extern "C" __global__ void attn_prefill_flash(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, int seq, int n_heads, int n_kv, int hd, float scale,
    float tau, float mean_corr, int sparse) {
  const int tq = blockIdx.x, h = blockIdx.y;
  if (tq >= seq || h >= n_heads) return;
  const int g = n_heads / n_kv;
  const int hkv = h / g;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nt = blockDim.x;  // 128
  extern __shared__ float sm[];
  float* score = sm;
  float* rb = sm + seq + 4;           // reduce scratch
  const float* qh = q + ((size_t)tq * n_heads + h) * hd;
  // hd may be 256 (Qwen3.5-4B/9B). One warp only covers 32*4=128 dims — tile along hd.
  const int d_stride = 32 * 4;
  // Phase 1: scores (4 warps 并行 4 个 K 位置, float4 点积 + warp reduce)
  for (int base = 0; base <= tq; base += 4) {
    const int tk = base + warp;
    if (tk <= tq) {
      const float* kt = k + ((size_t)tk * n_kv + hkv) * hd;
      float dot = 0.f;
      for (int d4 = lane * 4; d4 + 3 < hd; d4 += d_stride)
        dot += qh[d4] * kt[d4] + qh[d4 + 1] * kt[d4 + 1] + qh[d4 + 2] * kt[d4 + 2] +
               qh[d4 + 3] * kt[d4 + 3];
      dot = wp_reduce_sum(dot);
      if (lane == 0) score[tk] = dot * scale;
    }
  }
  __syncthreads();
  const int n = tq + 1;
  // Phase 2: 并行 softmax
  float lm = -1e30f;
  for (int i = tid; i < n; i += nt) lm = fmaxf(lm, score[i]);
  lm = blk_reduce_max(lm, rb, tid);
  float ls = 0.f;
  for (int i = tid; i < n; i += nt) {
    score[i] = expf(score[i] - lm);
    ls += score[i];
  }
  ls = blk_reduce_sum(ls, rb, tid);
  const float inv = 1.f / ls;
  for (int i = tid; i < n; i += nt) score[i] *= inv;
  __syncthreads();  // phase3 需读全部 score[]
  // Phase 2.5: 每块最大概率（稀疏跳块判定依据）
  const int B = 256;
  const int nb = (n + B - 1) / B;
  float* bmax = sm + seq + 16;
  if (sparse) {
    for (int b = tid; b < nb; b += nt) {
      const int lo = b * B, hi = (b + 1) * B < n ? (b + 1) * B : n;
      float m = 0.f;
      for (int i = lo; i < hi; ++i) m = fmaxf(m, score[i]);
      bmax[b] = m;
    }
    __syncthreads();
  }
  const float gate = sparse ? expf(-tau) * (1.f + mean_corr / (float)n) : 0.f;
  // Phase 3: V 累加（float4 tiles along hd, 稀疏跳块）
  float* oh = out + ((size_t)tq * n_heads + h) * hd;
  for (int d4 = lane * 4; d4 + 3 < hd; d4 += d_stride) {
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    for (int tk = 0; tk <= tq; ++tk) {
      if (sparse && bmax[tk / B] < gate) continue;
      const float w = score[tk];
      const float* vt = v + ((size_t)tk * n_kv + hkv) * hd;
      o0 += w * vt[d4];
      o1 += w * vt[d4 + 1];
      o2 += w * vt[d4 + 2];
      o3 += w * vt[d4 + 3];
    }
    oh[d4] = o0;
    oh[d4 + 1] = o1;
    oh[d4 + 2] = o2;
    oh[d4 + 3] = o3;
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
                   const float* d_x, float* d_y, int M, int K, int ng, int gs, bool is_awq,
                   int awq_zp) {
  if (!d_qweight || !d_scales || !d_x || !d_y || M <= 0 || K <= 0 || ng <= 0 || gs <= 0)
    return false;
  if (!is_awq && !d_zeros) return false;
  void* fn = get_jit_kernel(kGemvInt4Src, "gemv_int4");
  if (!fn) return false;
  int is_awq_i = is_awq ? 1 : 0;
  int awq_zp_i = awq_zp;
  constexpr int RPB = 8;
  constexpr int BLOCK_DIM = RPB * 32;
  const int scale_words = RPB * ng * (is_awq ? 1 : 2);
  const size_t scale_bytes = sizeof(unsigned short) * static_cast<size_t>((scale_words + 1) & ~1);
  // Prefer caching x in shared when it fits (~48KiB default limit).
  constexpr size_t kMaxSmem = 48ull * 1024ull;
  int use_sx = (scale_bytes + sizeof(float) * static_cast<size_t>(K) <= kMaxSmem) ? 1 : 0;
  const unsigned shmem =
      static_cast<unsigned>(scale_bytes + (use_sx ? sizeof(float) * static_cast<size_t>(K) : 0));
  void* params[] = {&d_qweight, &d_scales, &d_zeros, &d_x, &d_y, &M, &K, &ng, &gs, &is_awq_i, &awq_zp_i,
                    &use_sx};
  const int blocks = (M + RPB - 1) / RPB;
  return jit_launch(fn, static_cast<unsigned>(blocks), 1, 1, static_cast<unsigned>(BLOCK_DIM), 1, 1,
                    shmem, params);
}

bool jit_gemm_int4(const uint8_t* d_qweight, const uint16_t* d_scales, const uint16_t* d_zeros,
                   const float* d_X, float* d_Y, int M, int K, int n, int ng, int gs, bool is_awq,
                   int awq_zp) {
  if (!d_qweight || !d_scales || !d_X || !d_Y || M <= 0 || K <= 0 || n <= 0 || ng <= 0 || gs <= 0)
    return false;
  if (!is_awq && !d_zeros) return false;
  if (n == 1)
    return jit_gemv_int4(d_qweight, d_scales, d_zeros, d_X, d_Y, M, K, ng, gs, is_awq, awq_zp);
  void* fn = get_jit_kernel(kGemvInt4Src, "gemm_int4");
  if (!fn) return false;
  int is_awq_i = is_awq ? 1 : 0;
  int awq_zp_i = awq_zp;
  void* params[] = {&d_qweight, &d_scales, &d_zeros, &d_X, &d_Y, &M, &K, &n, &ng, &gs, &is_awq_i, &awq_zp_i};
  constexpr int ROWS_PER_BLOCK = 8;
  constexpr int BT = 8;
  constexpr int KT = 256;
  constexpr int BLOCK_DIM = ROWS_PER_BLOCK * 32;
  const int blocks_x = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  const int blocks_y = (n + BT - 1) / BT;
  const int scale_words = ROWS_PER_BLOCK * ng * (is_awq ? 1 : 2);
  const unsigned shmem =
      sizeof(unsigned short) * static_cast<unsigned>((scale_words + 1) & ~1) +
      sizeof(float) * static_cast<unsigned>(BT * KT);
  return jit_launch(fn, static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y), 1,
                    static_cast<unsigned>(BLOCK_DIM), 1, 1, shmem, params);
}


bool weight_budget_ok(size_t nbytes, bool need_headroom) {
  if (g_budget == 0) return true;
  const size_t pad = need_headroom ? kVramHeadroom : (256ull << 20);
  return g_used + nbytes + pad <= g_budget;
}

// Upload packed BF16/F16 weights (no FP32 inflate). Key = host pass pointer.
static uint16_t bf16_to_f16_bits_host(uint16_t h);  // 见 ensure_w16_pack 下方实现
const W16Pack* ensure_w16_pack(const uint16_t* W, int M, int K, bool is_f16) {
  if (!g_enabled || !W || M <= 0 || K <= 0) return nullptr;
  if (M >= kMaxGpuInt4Rows) return nullptr;
  auto it = g_w16_pack.find(W);
  if (it != g_w16_pack.end()) {
    if (it->second.M != M || it->second.K != K) return nullptr;
    return &it->second;
  }
  const size_t nbytes = sizeof(uint16_t) * static_cast<size_t>(M) * static_cast<size_t>(K);
  // Large packs (lm_head) keep 1.5GiB headroom; smaller packs are uncommon (prefer FP32).
  if (!weight_budget_ok(nbytes, M >= kW16PackMinRows)) return nullptr;
  void* d = nullptr;
  if (g_api.cudaMalloc(&d, nbytes) != kCudaSuccess) return nullptr;
  std::vector<uint16_t> hbuf;
  const uint16_t* upload = W;
  if (!is_f16) {
    // bf16 → fp16：正常段精确移位（fp16 尾数 10 位含 bf16 的 8 位），次正规段经 float+llrintf 舍入
    hbuf.resize(static_cast<size_t>(M) * static_cast<size_t>(K));
    for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] = bf16_to_f16_bits_host(W[i]);
    upload = hbuf.data();
  }
  if (g_api.cudaMemcpy(d, upload, nbytes, kCudaMemcpyH2D) != kCudaSuccess) {
    g_api.cudaFree(d);
    return nullptr;
  }
  W16Pack e;
  e.d_w = d;
  e.M = M;
  e.K = K;
  e.is_f16 = true;  // 统一按 fp16 存储（bf16 已正确转 fp16）
  e.bytes = nbytes;
  g_w16_pack[W] = e;
  g_used += nbytes;
  return &g_w16_pack[W];
}

// host: bf16(16bit) → fp16(16bit)。正常段精确；次正规用 float 舍入，避免位运算 UB。
static uint16_t bf16_to_f16_bits_host(uint16_t h) {
  const uint32_t s = h & 0x8000u;
  const uint32_t e8 = (h >> 7) & 0xffu;
  const uint32_t m8 = h & 0x7fu;
  if (e8 == 0u) return (uint16_t)s;                     // ±0 / bf16 次正规（<2^-126）→ fp16 0
  if (e8 == 0xffu) return (uint16_t)(s | 0x7c00u);      // inf/nan
  const int e16 = (int)e8 - 127 + 15;
  if (e16 >= 31) return (uint16_t)(s | 0x7c00u);        // 溢出 → inf
  if (e16 >= 1) return (uint16_t)(s | ((uint32_t)e16 << 10) | (m8 << 3));  // 正常：精确
  // 次正规：value = (128|m8)*2^(e8-135)；m10 = value*2^24
  const float val = (float)(128u | m8) * std::exp2f((float)((int)e8 - 135));
  uint32_t m10 = (uint32_t)std::llrintf(val * 16777216.0f);
  if (m10 == 0u) return (uint16_t)s;
  if (m10 > 1024u) m10 = 1023u;
  return (uint16_t)(s | m10);
}

// FP32 inflate for cublas (out/a/b). Faster than naive gemv_w16; uses more VRAM.
const float* ensure_w16_fp32(const uint16_t* W, int M, int K, bool is_f16) {
  if (!g_enabled || !W || M <= 0 || K <= 0) return nullptr;
  if (M >= kMaxGpuInt4Rows) return nullptr;
  auto it = g_cache.find(W);
  if (it != g_cache.end()) {
    if (it->second.M != M || it->second.K != K) return nullptr;
    return reinterpret_cast<const float*>(it->second.d_W);
  }
  const size_t nbytes = sizeof(float) * static_cast<size_t>(M) * static_cast<size_t>(K);
  if (!weight_budget_ok(nbytes, /*need_headroom=*/false)) return nullptr;
  std::vector<float> host(static_cast<size_t>(M) * static_cast<size_t>(K));
  for (size_t i = 0; i < host.size(); ++i)
    host[i] = is_f16 ? f16_to_f32(W[i]) : bf16_to_f32(W[i]);
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
  g_cache[W] = e;
  g_used += nbytes;
  return reinterpret_cast<const float*>(dW);
}

bool jit_gemv_w16(const uint16_t* d_W, const float* d_x, float* d_y, int M, int K, bool is_f16) {
  if (!d_W || !d_x || !d_y || M <= 0 || K <= 0) return false;
  void* fn = get_jit_kernel(kActSrc, "gemv_w16");
  if (!fn) return false;
  int is_f16_i = is_f16 ? 1 : 0;
  void* params[] = {&d_W, &d_x, &d_y, &M, &K, &is_f16_i};
  constexpr unsigned BLOCK = 256;
  const unsigned grid = static_cast<unsigned>(M < 4096 ? M : 4096);
  return jit_launch(fn, grid, 1, 1, BLOCK, 1, 1, BLOCK * sizeof(float), params);
}

bool ensure_w16_tile(int rows, int K) {
  const size_t need = static_cast<size_t>(rows) * static_cast<size_t>(K);
  if (need == 0) return false;
  if (static_cast<int>(need) <= g_w16_tile_cap && g_w16_tile) return true;
  if (g_w16_tile) {
    g_api.cudaFree(g_w16_tile);
    g_w16_tile = nullptr;
    g_w16_tile_cap = 0;
  }
  if (g_dequant_scratch) {
    g_api.cudaFree(g_dequant_scratch);
    g_dequant_scratch = nullptr;
    g_dequant_scratch_cap = 0;
  }
  void* p = nullptr;
  if (g_api.cudaMalloc(&p, sizeof(float) * need) != kCudaSuccess) return false;
  g_w16_tile = static_cast<float*>(p);
  g_w16_tile_cap = static_cast<int>(need);
  return true;
}

// Vocab-scale lm_head: tiled FP32 inflate + cublasSgemm.
// cublasGemmEx(BF16×FP32) was tried but measured slower here (lm_head ~26ms vs ~18ms tiled).
bool gemv_w16_tiled_cublas(const uint16_t* d_W, const float* d_x, float* d_y, int M, int K,
                           bool is_f16) {
  if (!d_W || !d_x || !d_y || M <= 0 || K <= 0 || !g_cublas) return false;
  // FP16 权重 + cublasGemmEx → tensor-core GEMV（打包已统一为 fp16 且验证正确；LLMOC_TC_LM=0 关闭）
  const char* tc_lm_on = std::getenv("LLMOC_TC_LM");
  const bool tc_lm = !(tc_lm_on && tc_lm_on[0] == '0');
  if (is_f16 && tc_lm && g_api.cublasGemmEx) {
    void* fx = get_jit_kernel(kTcSrc, "f32_to_f16_buf");
    if (fx) {
      std::lock_guard<std::mutex> lock(g_mu);
      const size_t need = sizeof(unsigned short) * static_cast<size_t>(K);
      if (!g_tc_x_f16 || need > g_tc_x_f16_cap) {
        if (g_tc_x_f16) g_api.cudaFree(g_tc_x_f16);
        void* p = nullptr;
        if (g_api.cudaMalloc(&p, need) == kCudaSuccess) {
          g_tc_x_f16 = static_cast<unsigned short*>(p);
          g_tc_x_f16_cap = need;
        } else {
          g_tc_x_f16 = nullptr;
          g_tc_x_f16_cap = 0;
        }
      }
      if (g_tc_x_f16) {
        int nk = K;
        const unsigned nxt = static_cast<unsigned>((K + 255) / 256);
        void* xp[] = {(void*)&d_x, (void*)&g_tc_x_f16, (void*)&nk};
        if (jit_launch(fx, nxt < 65535u ? nxt : 65535u, 1, 1, 256, 1, 1, 0, xp)) {
          const float alpha = 1.f, beta = 0.f;
          if (g_api.cublasGemmEx(g_cublas, kCublasOpT, kCublasOpN, M, 1, K, &alpha, d_W,
                                 kCudaR_16F, K, g_tc_x_f16, kCudaR_16F, K, &beta, d_y,
                                 kCudaR_32F, M, kCublasCompute32F, kCublasGemmDefault) ==
              kCublasSuccess)
            return true;
        }
      }
    }
    // 失败则落到下方 tiled FP32 路径
  }
  if (!g_api.cublasSgemm) return false;
  void* fn = get_jit_kernel(kActSrc, "w16_tile_to_f32");
  if (!fn) return false;
  const int tile = (M < kW16TileRows) ? M : kW16TileRows;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_w16_tile(tile, K)) return false;
  }
  const float alpha = 1.f, beta = 0.f;
  int is_f16_i = is_f16 ? 1 : 0;
  int M_i = M, K_i = K;
  constexpr unsigned BLOCK = 256;
  for (int m0 = 0; m0 < M; m0 += tile) {
    int rows = (m0 + tile <= M) ? tile : (M - m0);
    void* params[] = {&d_W, &g_w16_tile, &M_i, &K_i, &m0, &rows, &is_f16_i};
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(K);
    const unsigned grid = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    const unsigned grid_cap = grid < 65535u ? grid : 65535u;
    if (!jit_launch(fn, grid_cap, 1, 1, BLOCK, 1, 1, 0, params)) return false;
    float* d_yt = d_y + m0;
    if (g_api.cublasSgemm(g_cublas, kCublasOpT, kCublasOpN, rows, 1, K, &alpha, g_w16_tile, K, d_x,
                          K, &beta, d_yt, rows) != kCublasSuccess)
      return false;
  }
  return true;
}

// Device GEMV for W16: mid-size → FP32+cublas; vocab-scale → pack tile+cublas.
bool gemv_w16_dev_x(const uint16_t* pass, bool is_f16w, const float* d_x, float* d_y, int M,
                    int K) {
  if (!pass || !d_x || !d_y || M <= 0 || K <= 0) return false;
  if (M < kW16PackMinRows) {
    const float* dW = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      dW = ensure_w16_fp32(pass, M, K, is_f16w);
    }
    if (!dW) return false;
    const float alpha = 1.f, beta0 = 0.f;
    return g_api.cublasSgemm(g_cublas, kCublasOpT, kCublasOpN, M, 1, K, &alpha, dW, K, d_x, K,
                             &beta0, d_y, M) == kCublasSuccess;
  }
  const W16Pack* pack = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    pack = ensure_w16_pack(pass, M, K, is_f16w);
    if (!pack) return false;
  }
  return gemv_w16_tiled_cublas(static_cast<const uint16_t*>(pack->d_w), d_x, d_y, M, K, pack->is_f16);
}

bool try_gemm_w16(const float* x, const uint16_t* W, float* y, int M, int K, bool is_f16) {
  if (!g_enabled || !x || !W || !y || M <= 0 || K <= 0) return false;
  // Mid-size (out/a/b): full FP32 mirror + cublas.
  if (M < kW16PackMinRows) {
    const float* dW = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      dW = ensure_w16_fp32(W, M, K, is_f16);
      if (!dW) return false;
      return gemm_dev(dW, x, y, M, K);
    }
  }
  // Vocab-scale lm_head: pack + tiled FP32 cublas (not naive gemv_w16).
  const W16Pack* pack = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    pack = ensure_w16_pack(W, M, K, is_f16);
    if (!pack) return false;
    if (!ensure_xy(M, K, 1)) return false;
    if (!upload_x_sticky(x, K)) return false;
  }
  if (!gemv_w16_tiled_cublas(static_cast<const uint16_t*>(pack->d_w),
                             reinterpret_cast<const float*>(g_dx), reinterpret_cast<float*>(g_dy), M,
                             K, pack->is_f16))
    return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(y, g_dy, sizeof(float) * static_cast<size_t>(M), kCudaMemcpyD2H) ==
         kCudaSuccess;
}

bool try_gemm_w16_batch(const float* X, int n, const uint16_t* W, float* Y, int M, int K,
                        bool is_f16) {
  if (!g_enabled || !X || !W || !Y || M <= 0 || K <= 0 || n <= 0) return false;
  if (n == 1) return try_gemm_w16(X, W, Y, M, K, is_f16);
  constexpr int kMaxGpuBatch = 128;
  if (n > kMaxGpuBatch) return false;

  if (M < kW16PackMinRows) {
    const float* dW = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      dW = ensure_w16_fp32(W, M, K, is_f16);
      if (!dW) return false;
      return gemm_dev_batch(dW, X, n, Y, M, K);
    }
  }

  // Vocab × batch: reuse single-vector tiled path per column (rare for lm_head).
  for (int i = 0; i < n; ++i) {
    if (!try_gemm_w16(X + static_cast<size_t>(i) * K, W, Y + static_cast<size_t>(i) * M, M, K,
                      is_f16))
      return false;
  }
  return true;
}

bool prefetch_w16(const uint16_t* W, int M, int K, bool is_f16) {
  if (!g_enabled || !W || M <= 0 || K <= 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (M >= kW16PackMinRows) return ensure_w16_pack(W, M, K, is_f16) != nullptr;
  return ensure_w16_fp32(W, M, K, is_f16) != nullptr;
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
    if (!upload_x_sticky(x, W.K)) return false;
    x_uploaded = true;
  }
  if (!x_uploaded) return false;
  auto tk0 = std::chrono::steady_clock::now();
  const bool ok = jit_gemv_int4(static_cast<const uint8_t*>(res->d_qweight),
                                static_cast<const uint16_t*>(res->d_scales),
                                static_cast<const uint16_t*>(res->d_zeros),
                                reinterpret_cast<const float*>(g_dx),
                                reinterpret_cast<float*>(g_dy),
                                res->M, res->K, res->ng, res->gs, res->is_awq, res->awq_zp);
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

bool ensure_dequant_scratch(size_t floats) {
  if (floats == 0) return false;
  if (floats <= g_dequant_scratch_cap && g_dequant_scratch) return true;
  if (g_dequant_scratch) {
    g_api.cudaFree(g_dequant_scratch);
    g_dequant_scratch = nullptr;
    g_dequant_scratch_cap = 0;
  }
  void* p = nullptr;
  if (g_api.cudaMalloc(&p, sizeof(float) * floats) != kCudaSuccess) return false;
  g_dequant_scratch = static_cast<float*>(p);
  g_dequant_scratch_cap = floats;
  return true;
}

bool dequant_int4_resident_to_scratch(const Int4Resident* res) {
  if (!res || !g_dequant_scratch) return false;
  void* fn = get_jit_kernel(kGemvInt4Src, "dequant_int4_to_f32");
  if (!fn) return false;
  const size_t n = static_cast<size_t>(res->M) * static_cast<size_t>(res->K);
  if (n > g_dequant_scratch_cap) return false;
  int M = res->M, K = res->K, ng = res->ng, gs = res->gs;
  int is_awq_i = res->is_awq ? 1 : 0;
  int awq_zp_i = res->awq_zp;
  const uint8_t* q = static_cast<const uint8_t*>(res->d_qweight);
  const uint16_t* s = static_cast<const uint16_t*>(res->d_scales);
  const uint16_t* z = static_cast<const uint16_t*>(res->d_zeros);
  // zeros may be null for AWQ; pass scales as dummy (kernel ignores when is_awq).
  if (!z) z = s;
  void* params[] = {&q, &s, &z, &g_dequant_scratch, &M, &K, &ng, &gs, &is_awq_i, &awq_zp_i};
  constexpr unsigned BLOCK = 256;
  unsigned grid = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
  if (grid > 65535u) grid = 65535u;
  return jit_launch(fn, grid, 1, 1, BLOCK, 1, 1, 0, params);
}

// Prefill: GPU-dequant INT4 → scratch once, then cuBLAS SGEMM (usually >> on-the-fly JIT).
bool try_gemm_int4_batch_cublas_scratch(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  if (!g_enabled || !X || !Y || n <= 1 || !g_cublas) return false;
  const Int4Resident* res = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    res = ensure_int4_resident(W);
    if (!res || !jit_available()) return false;
    const size_t need = static_cast<size_t>(res->M) * static_cast<size_t>(res->K);
    if (!ensure_dequant_scratch(need)) return false;
  }
  if (!dequant_int4_resident_to_scratch(res)) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  return gemm_dev_batch(g_dequant_scratch, X, n, Y, res->M, res->K);
}

// Tensor-core FP16 prefill GEMM: upload X(fp32) → dequant W resident int4→fp16 → convert X→fp16 →
// cublasGemmEx(FP16×FP16→FP32)。仅长 batch（FLOP 密集）值得。
bool tc_gemm_int4_batch_f16(const float* X, int n, const qlwc::Int4View& W, float* Y) {
  if (!g_enabled || !X || !Y || n <= 0) return false;
  if (!g_api.cublasGemmEx) return false;
  {
    // tc FP16 GEMM 已在真实权重验证正确（长 prefill n>=64 首选）；LLMOC_TC_GEMM=0 关闭
    const char* e = std::getenv("LLMOC_TC_GEMM");
    if (e && e[0] == '0') return false;
  }
  const Int4Resident* res = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    res = ensure_int4_resident(W);
  }
  if (!res || !jit_available()) return false;
  if (!res->d_qweight || !res->d_scales) return false;
  const int M = res->M, K = res->K;
  if (M <= 0 || K <= 0 || res->ng <= 0 || res->gs <= 0) return false;
  const size_t wb = sizeof(unsigned short) * static_cast<size_t>(M) * K;
  const size_t xb = sizeof(unsigned short) * static_cast<size_t>(n) * K;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_tc_w_f16 || wb > g_tc_w_f16_cap) {
      if (g_tc_w_f16) g_api.cudaFree(g_tc_w_f16);
      void* p = nullptr;
      if (g_api.cudaMalloc(&p, wb) != kCudaSuccess) { g_tc_w_f16 = nullptr; g_tc_w_f16_cap = 0; return false; }
      g_tc_w_f16 = static_cast<unsigned short*>(p);
      g_tc_w_f16_cap = wb;
    }
    if (!g_tc_x_f16 || xb > g_tc_x_f16_cap) {
      if (g_tc_x_f16) g_api.cudaFree(g_tc_x_f16);
      void* p = nullptr;
      if (g_api.cudaMalloc(&p, xb) != kCudaSuccess) { g_tc_x_f16 = nullptr; g_tc_x_f16_cap = 0; return false; }
      g_tc_x_f16 = static_cast<unsigned short*>(p);
      g_tc_x_f16_cap = xb;
    }
    if (!ensure_xy(M, K, n)) return false;
    if (g_api.cudaMemcpy(g_dx, X, sizeof(float) * static_cast<size_t>(n) * K, kCudaMemcpyH2D) !=
        kCudaSuccess)
      return false;
  }
  void* dq = get_jit_kernel(kTcSrc, "int4_to_f16");
  void* cx = get_jit_kernel(kTcSrc, "f32_to_f16_buf");
  if (!dq || !cx) return false;
  const int ng = res->ng, gs = res->gs, is_awq = res->is_awq ? 1 : 0, awq_zp = res->awq_zp;
  const unsigned wtotal = static_cast<unsigned>((M * (unsigned long)K + 255u) / 256u);
  {
    std::lock_guard<std::mutex> lock(g_mu);
    void* dp[] = {(void*)&res->d_qweight, (void*)&res->d_scales, (void*)&res->d_zeros,
                  (void*)&g_tc_w_f16, (void*)&M, (void*)&K, (void*)&ng, (void*)&gs,
                  (void*)&is_awq, (void*)&awq_zp};
    if (!jit_launch(dq, wtotal < 65535u ? wtotal : 65535u, 1, 1, 256, 1, 1, 0, dp)) return false;
    const int nx = n * K;
    const unsigned nxt = static_cast<unsigned>((nx + 255) / 256);
    void* xp[] = {(void*)&g_dx, (void*)&g_tc_x_f16, (void*)&nx};
    if (!jit_launch(cx, nxt < 65535u ? nxt : 65535u, 1, 1, 256, 1, 1, 0, xp)) return false;
  }
  const float alpha = 1.f, beta = 0.f;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    int st = kCublasCompute32F;
    if (g_api.cublasGemmEx(g_cublas, kCublasOpT, kCublasOpN, M, n, K, &alpha, g_tc_w_f16,
                           kCudaR_16F, K, g_tc_x_f16, kCudaR_16F, K, &beta,
                           reinterpret_cast<float*>(g_dy), kCudaR_32F, M, st, kCublasGemmDefault) !=
        kCublasSuccess)
      return false;
    if (g_api.cudaMemcpy(Y, g_dy, sizeof(float) * static_cast<size_t>(n) * M, kCudaMemcpyD2H) !=
        kCudaSuccess)
      return false;
  }
  return true;
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
      if (!upload_x_sticky(X, W.K)) return false;
      x_uploaded = true;
    }
    if (!x_uploaded) return false;
    const bool ok = jit_gemv_int4(static_cast<const uint8_t*>(res->d_qweight),
                                  static_cast<const uint16_t*>(res->d_scales),
                                  static_cast<const uint16_t*>(res->d_zeros),
                                  reinterpret_cast<const float*>(g_dx),
                                  reinterpret_cast<float*>(g_dy), res->M, res->K, res->ng,
                                  res->gs, res->is_awq, res->awq_zp);
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
                         K, c, res->ng, res->gs, res->is_awq, res->awq_zp))
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
    // 很长 prefill：优先 FP32（INT4→FP32 scratch / device），减轻 32 层累积的 FP16 误差；
    // 短于阈值仍先试 Tensor-core（吞吐优先）。
    auto log_gemm_once = [&](const char* path) {
      static std::atomic<int> once{1};
      if (once.fetch_sub(1) > 0)
        LOG_INFO("gemm prefill: path=%s n=%d M=%d K=%d (LLMOC_TC_GEMM=0 forces no TC)", path, n,
                 W.M, W.K);
    };
    if (n >= 512) {
      if (try_gemm_int4_batch_cublas_scratch(X, n, W, Y)) {
        log_gemm_once("fp32_scratch");
        return true;
      }
      if (try_cublas_fp32_batch()) {
        log_gemm_once("fp32_device");
        return true;
      }
      if (try_int4_jit_batch()) {
        log_gemm_once("int4_jit");
        return true;
      }
      if (tc_gemm_int4_batch_f16(X, n, W, Y)) {
        log_gemm_once("tc_fp16_fallback");
        return true;
      }
      return false;
    }
    // 长 prefill：Tensor-core FP16 (INT4→FP16 + cublasGemmEx) 最优先；JIT 与 FP32 兜底。
    if (tc_gemm_int4_batch_f16(X, n, W, Y)) return true;
    if (try_int4_jit_batch()) return true;
    if (try_gemm_int4_batch_cublas_scratch(X, n, W, Y)) return true;
    if (try_cublas_fp32_batch()) return true;
    return false;
  }
  // 短序列：先 cuBLAS，再 INT4 JIT（显存不够装 FP32 副本时）
  if (try_cublas_fp32_batch()) return true;
  if (try_int4_jit_batch()) return true;
  return false;
}

// Prefill: shared-X multi-output INT4 batch. Uploads each X chunk once, then runs N GEMMs.
bool try_gemm_int4_batch_multi(const float* X, int n, const qlwc::Int4View* const* Ws,
                               float* const* Ys, int nW) {
  if (!g_enabled || !X || !Ws || !Ys || n <= 0 || nW < 2 || nW > 4) return false;
  if (!jit_available()) return false;
  if (n == 1) return try_gemm_int4_multi(X, Ws, Ys, nW);

  const Int4Resident* res[4] = {};
  int max_m = 0;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    for (int i = 0; i < nW; ++i) {
      if (!Ws[i] || !Ys[i]) return false;
      res[i] = ensure_int4_resident(*Ws[i]);
      if (!res[i]) return false;
      if (res[i]->M > max_m) max_m = res[i]->M;
    }
    for (int i = 1; i < nW; ++i) {
      if (res[i]->K != res[0]->K || res[i]->ng != res[0]->ng || res[i]->gs != res[0]->gs ||
          res[i]->is_awq != res[0]->is_awq)
        return false;
    }
  }
  const int K = res[0]->K;
  constexpr size_t kMaxFloats = 64ull << 20;
  int chunk = n;
  auto fits = [&](int c) {
    return static_cast<size_t>(c) * static_cast<size_t>(K) <= kMaxFloats &&
           static_cast<size_t>(c) * static_cast<size_t>(max_m) <= kMaxFloats;
  };
  while (chunk > 1 && !fits(chunk)) chunk = (chunk + 1) / 2;
  if (chunk < 1) chunk = 1;

  for (int b0 = 0; b0 < n; b0 += chunk) {
    const int c = chunk < (n - b0) ? chunk : (n - b0);
    const float* Xp = X + static_cast<size_t>(b0) * K;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      if (!ensure_xy(max_m, K, c)) return false;
      if (g_api.cudaMemcpy(g_dx, Xp, sizeof(float) * static_cast<size_t>(c) * K, kCudaMemcpyH2D) !=
          kCudaSuccess)
        return false;
    }
    for (int i = 0; i < nW; ++i) {
      const int M = res[i]->M;
      float* Yp = Ys[i] + static_cast<size_t>(b0) * M;
      if (!jit_gemm_int4(static_cast<const uint8_t*>(res[i]->d_qweight),
                         static_cast<const uint16_t*>(res[i]->d_scales),
                         static_cast<const uint16_t*>(res[i]->d_zeros),
                         reinterpret_cast<const float*>(g_dx), reinterpret_cast<float*>(g_dy), M, K,
                         c, res[i]->ng, res[i]->gs, res[i]->is_awq, res[i]->awq_zp))
        return false;
      std::lock_guard<std::mutex> lock(g_mu);
      if (g_api.cudaMemcpy(Yp, g_dy, sizeof(float) * static_cast<size_t>(c) * M, kCudaMemcpyD2H) !=
          kCudaSuccess)
        return false;
    }
  }
  return true;
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
    if (!upload_x_sticky(x, res[0]->K)) return false;
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
  int awq_zp_i = res[0]->awq_zp;
  const float* dx = reinterpret_cast<const float*>(g_dx);
  void* params[] = {
    &q[0], &s[0], &z[0], &ydev[0], &m[0],
    &q[1], &s[1], &z[1], &ydev[1], &m[1],
    &q[2], &s[2], &z[2], &ydev[2], &m[2],
    &q[3], &s[3], &z[3], &ydev[3], &m[3],
    &n, &K, &ng, &gs, &is_awq_i, &awq_zp_i, &dx
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
  {
    const char* e = std::getenv("LLMOC_GPU_ATTN");
    if (e && e[0] == '0') return false;  // A/B: force CPU attn
  }
  const float tau = []
  {
    const char* e = std::getenv("LLMOC_PREFILL_TAU");
    return e && e[0] ? static_cast<float>(std::atof(e)) : 1e9f;
  }();
  const float mean_corr = []
  {
    const char* e = std::getenv("LLMOC_PREFILL_MEANCORR");
    return e && e[0] ? static_cast<float>(std::atof(e)) : 0.f;
  }();
  return try_attn_prefill_sparse(q, k, v, out, seq, n_heads, n_kv_heads, head_dim, scale, tau,
                                 mean_corr);
}

bool try_attn_prefill_sparse(const float* q, const float* k, const float* v, float* out, int seq,
                             int n_heads, int n_kv_heads, int head_dim, float scale, float tau,
                             float mean_corr) {
  if (!g_enabled || !q || !k || !v || !out) return false;
  if (seq <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) return false;
  if (n_heads % n_kv_heads != 0) return false;
  if (!jit_available()) return false;

  // FlashPrefill: default on for hd<=128. Qwen3.5 full-attn uses hd=256 — tiled flash exists
  // but long-prefill still shows early-EOS quality issues; prefer naive (full-hd loops) unless
  // LLMOC_ATTN_PREFILL=flash. Force naive: =naive|n.
  bool use_flash = true;
  {
    const char* e = std::getenv("LLMOC_ATTN_PREFILL");
    if (e && (e[0] == 'n' || e[0] == 'N'))
      use_flash = false;
    else if (e && (e[0] == 'f' || e[0] == 'F'))
      use_flash = true;
    else if (head_dim > 128)
      use_flash = false;
  }
  const int sparse = (tau < 50.f) ? 1 : 0;

  // Flash kernel 需要 hd%4==0；naive 对所有形状可用。共享上限 ~48KiB（score+nb）。
  const int nb = (seq + 255) / 256;
  const size_t shmem_flash = sizeof(float) * static_cast<size_t>(seq + 16 + nb);
  const size_t shmem_naive = sizeof(float) * static_cast<size_t>(seq);
  if (use_flash && shmem_flash > 48ull * 1024ull) use_flash = false;
  if (shmem_naive > 48ull * 1024ull) return false;
  const bool flash_ok = use_flash && (head_dim % 4 == 0) && (head_dim > 0);
  if (flash_ok && seq >= 1024 && head_dim >= 256) {
    static std::atomic<int> once{1};
    if (once.fetch_sub(1) > 0)
      LOG_INFO("attn prefill: flash hd=%d seq=%d (set LLMOC_ATTN_PREFILL=naive to compare)",
               head_dim, seq);
  } else if (!flash_ok && head_dim > 128 && seq >= 256) {
    static std::atomic<int> once{1};
    if (once.fetch_sub(1) > 0)
      LOG_INFO("attn prefill: naive hd=%d seq=%d (safer for Qwen3.5 hd=256; FLASH to force flash)",
               head_dim, seq);
  }
  void* fn = get_jit_kernel(kAttnPrefillSrc, flash_ok ? "attn_prefill_flash" : "attn_prefill_naive");
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
  if (flash_ok) {
    float tau_mut = tau, mc_mut = mean_corr;
    int sparse_i = sparse;
    void* params[] = {&dq, &dk, &dv, &dout, &seq_i, &nh, &nkv, &hd, &scale_mut,
                      &tau_mut, &mc_mut, &sparse_i};
    if (!jit_launch(fn, static_cast<unsigned>(seq), static_cast<unsigned>(n_heads), 1, 128, 1, 1,
                    static_cast<unsigned>(shmem_flash), params))
      return false;
  } else {
    void* params[] = {&dq, &dk, &dv, &dout, &seq_i, &nh, &nkv, &hd, &scale_mut};
    if (!jit_launch(fn, static_cast<unsigned>(seq), static_cast<unsigned>(n_heads), 1, 256, 1, 1,
                    static_cast<unsigned>(shmem_naive), params))
      return false;
  }
  return d2h(out, g_attn_o, ob);
}

// 单 query decode attention：flash kernel（seq 并行）。k/v_cache 上传当前使用的 nkv*seq_len*stride 切片。
bool try_attn_decode_gpu(const float* q, const float* k_cache, const float* v_cache, float* out,
                         int seq_len, int stride, int n_heads, int n_kv, int hd, float scale) {
  if (!g_enabled || !jit_available()) return false;
  {
    const char* e = std::getenv("LLMOC_GPU_DECODE");
    if (e && e[0] == '0') return false;  // A/B：force CPU attn_decode_one
  }
  if (!q || !k_cache || !v_cache || !out) return false;
  if (seq_len <= 0 || n_heads <= 0 || n_kv <= 0 || hd <= 0 || n_heads % n_kv) return false;
  const size_t smem = sizeof(float) * (static_cast<size_t>(hd) + seq_len + 16);
  if (smem > 48ull * 1024ull) return false;
  void* fn = get_jit_kernel(kAttnDecodeSrc, "attn_decode_flash");
  if (!fn) return false;
  const size_t qb = sizeof(float) * static_cast<size_t>(n_heads) * hd;
  const size_t kvb = sizeof(float) * static_cast<size_t>(n_kv) * static_cast<size_t>(seq_len) * hd;
  const size_t kv_row = sizeof(float) * static_cast<size_t>(hd) * static_cast<size_t>(seq_len);
  const size_t ob = qb;
  auto ensure = [&](float*& p, size_t& cap, size_t need) -> bool {
    std::lock_guard<std::mutex> lock(g_mu);
    if (need <= cap && p) return true;
    if (p) g_api.cudaFree(p);
    void* v = nullptr;
    if (g_api.cudaMalloc(&v, need) != kCudaSuccess) { p = nullptr; cap = 0; return false; }
    p = static_cast<float*>(v);
    cap = need;
    return true;
  };
  if (!ensure(g_dec_q, g_dec_q_cap, qb)) return false;
  if (!ensure(g_dec_k, g_dec_k_cap, kvb)) return false;
  if (!ensure(g_dec_v, g_dec_v_cap, kvb)) return false;
  if (!ensure(g_dec_o, g_dec_o_cap, ob)) return false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_api.cudaMemcpy(g_dec_q, q, qb, kCudaMemcpyH2D) != kCudaSuccess) return false;
    // 只上传实际使用的 seq_len 切片：逐 kv 行（原 layout [hkv*stride+t]*hd → 设备 packed [hkv*seq_len+t]*hd）
    for (int hkv = 0; hkv < n_kv; ++hkv) {
      const float* ksrc = k_cache + ((size_t)hkv * stride) * hd;
      const float* vsrc = v_cache + ((size_t)hkv * stride) * hd;
      float* kdst = g_dec_k + ((size_t)hkv * seq_len) * hd;
      float* vdst = g_dec_v + ((size_t)hkv * seq_len) * hd;
      if (g_api.cudaMemcpy(kdst, ksrc, kv_row, kCudaMemcpyH2D) != kCudaSuccess) return false;
      if (g_api.cudaMemcpy(vdst, vsrc, kv_row, kCudaMemcpyH2D) != kCudaSuccess) return false;
    }
  }
  float scale_m = scale;
  int sl = seq_len, st = seq_len /*设备 packed  row stride=seq_len*/, nh = n_heads, nkv = n_kv, hdd = hd;
  void* dp[] = {(void*)&g_dec_q, (void*)&g_dec_k, (void*)&g_dec_v, (void*)&g_dec_o,
                (void*)&sl, (void*)&st, (void*)&nh, (void*)&nkv, (void*)&hdd, (void*)&scale_m};
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!jit_launch(fn, n_heads, 1, 1, 256, 1, 1, static_cast<unsigned>(smem), dp))
      return false;
    return g_api.cudaMemcpy(out, g_dec_o, ob, kCudaMemcpyD2H) == kCudaSuccess;
  }
}

bool try_gated_delta_gpu(const float* q, const float* k, const float* v, const float* g,
                         const float* beta, float* state, float* out,
                         int n_heads, int dk, int dv) {
  auto fail = [&](const char* why) {
    g_gdn_last_err = why;
    ++g_gdn_fail;
    return false;
  };
  if (!g_enabled || !jit_available() || !g_resident) return false;
  if (!q || !k || !v || !g || !beta || !state || !out) return fail("null_arg");
  if (n_heads <= 0 || dk <= 0 || dv <= 0) return fail("bad_dims");
  void* fn = get_jit_kernel(kGdnSrc, "gated_delta_kernel");
  if (!fn) return fail(g_status.c_str());

  const size_t state_bytes = sizeof(float) * static_cast<size_t>(n_heads) * dk * dv;
  const size_t io_bytes =
      sizeof(float) * (static_cast<size_t>(n_heads) * dk * 2 + static_cast<size_t>(n_heads) * dv +
                       static_cast<size_t>(n_heads) * 2 + static_cast<size_t>(n_heads) * dv);
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_gdn_state.find(state);
  if (it == g_gdn_state.end()) {
    void* d_state_v = nullptr;
    if (g_api.cudaMalloc(&d_state_v, state_bytes) != kCudaSuccess) return fail("state_malloc");
    if (g_api.cudaMemcpy(d_state_v, state, state_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(d_state_v);
      return fail("state_h2d");
    }
    g_gdn_state[state] = static_cast<float*>(d_state_v);
    it = g_gdn_state.find(state);
  }
  float* d_state = it->second;
  if (io_bytes > g_gdn_buf_cap) {
    if (g_gdn_buf) g_api.cudaFree(g_gdn_buf);
    void* buf_v = nullptr;
    if (g_api.cudaMalloc(&buf_v, io_bytes) != kCudaSuccess) {
      g_gdn_buf = nullptr;
      g_gdn_buf_cap = 0;
      return fail("io_malloc");
    }
    g_gdn_buf = static_cast<float*>(buf_v);
    g_gdn_buf_cap = io_bytes;
  }
  float* d_q = g_gdn_buf;
  float* d_k = d_q + n_heads * dk;
  float* d_v = d_k + n_heads * dk;
  float* d_g = d_v + n_heads * dv;
  float* d_beta = d_g + n_heads;
  float* d_out = d_beta + n_heads;
  if (g_api.cudaMemcpy(d_q, q, sizeof(float) * n_heads * dk, kCudaMemcpyH2D) != kCudaSuccess)
    return fail("q_h2d");
  if (g_api.cudaMemcpy(d_k, k, sizeof(float) * n_heads * dk, kCudaMemcpyH2D) != kCudaSuccess)
    return fail("k_h2d");
  if (g_api.cudaMemcpy(d_v, v, sizeof(float) * n_heads * dv, kCudaMemcpyH2D) != kCudaSuccess)
    return fail("v_h2d");
  if (g_api.cudaMemcpy(d_g, g, sizeof(float) * n_heads, kCudaMemcpyH2D) != kCudaSuccess)
    return fail("g_h2d");
  if (g_api.cudaMemcpy(d_beta, beta, sizeof(float) * n_heads, kCudaMemcpyH2D) != kCudaSuccess)
    return fail("beta_h2d");
  float scale = 1.f / sqrtf(static_cast<float>(dk));
  void* params[] = {&d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_out, &dk, &dv, &scale};
  if (!jit_launch(fn, static_cast<unsigned>(n_heads), 1, 1, static_cast<unsigned>(dv), 1, 1, 0,
                  params))
    return fail("launch");
  if (g_api.cudaMemcpy(out, d_out, sizeof(float) * n_heads * dv, kCudaMemcpyD2H) != kCudaSuccess)
    return fail("out_d2h");
  ++g_gdn_ok;
  return true;
}

bool try_gated_delta_gpu_seq(const float* q, const float* k, const float* v, const float* g,
                             const float* beta, float* state, float* out, int seq, int n_heads,
                             int dk, int dv) {
  auto fail = [&](const char* why) {
    g_gdn_last_err = why;
    ++g_gdn_fail;
    return false;
  };
  if (!g_enabled || !jit_available() || !g_resident) return false;
  if (!q || !k || !v || !g || !beta || !state || !out) return fail("null_arg");
  if (seq <= 0 || n_heads <= 0 || dk <= 0 || dv <= 0) return fail("bad_dims");
  if (seq == 1)
    return try_gated_delta_gpu(q, k, v, g, beta, state, out, n_heads, dk, dv);

  void* fn_seq = get_jit_kernel(kGdnSrc, "gated_delta_seq_kernel");
  void* fn_step = nullptr;
  if (!fn_seq) {
    fn_step = get_jit_kernel(kGdnSrc, "gated_delta_kernel");
    if (!fn_step) return fail(g_status.c_str());
  }

  // ~256 tokens/chunk: fewer H2D rounds; seq kernel = 1 launch per chunk.
  constexpr int kChunk = 256;
  const size_t state_bytes = sizeof(float) * static_cast<size_t>(n_heads) * dk * dv;
  const size_t per_tok = sizeof(float) * (static_cast<size_t>(n_heads) * dk * 2 +
                                          static_cast<size_t>(n_heads) * dv * 2 +
                                          static_cast<size_t>(n_heads) * 2);
  const size_t io_bytes = per_tok * static_cast<size_t>(kChunk);

  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_gdn_state.find(state);
  if (it == g_gdn_state.end()) {
    void* d_state_v = nullptr;
    if (g_api.cudaMalloc(&d_state_v, state_bytes) != kCudaSuccess) return fail("state_malloc");
    if (g_api.cudaMemcpy(d_state_v, state, state_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(d_state_v);
      return fail("state_h2d");
    }
    g_gdn_state[state] = static_cast<float*>(d_state_v);
    it = g_gdn_state.find(state);
  }
  float* d_state = it->second;
  if (io_bytes > g_gdn_buf_cap) {
    if (g_gdn_buf) g_api.cudaFree(g_gdn_buf);
    void* buf_v = nullptr;
    if (g_api.cudaMalloc(&buf_v, io_bytes) != kCudaSuccess) {
      g_gdn_buf = nullptr;
      g_gdn_buf_cap = 0;
      return fail("io_malloc");
    }
    g_gdn_buf = static_cast<float*>(buf_v);
    g_gdn_buf_cap = io_bytes;
  }

  const float scale = 1.f / sqrtf(static_cast<float>(dk));
  for (int t0 = 0; t0 < seq; t0 += kChunk) {
    const int n = (t0 + kChunk <= seq) ? kChunk : (seq - t0);
    float* d_q = g_gdn_buf;
    float* d_k = d_q + static_cast<size_t>(n) * n_heads * dk;
    float* d_v = d_k + static_cast<size_t>(n) * n_heads * dk;
    float* d_g = d_v + static_cast<size_t>(n) * n_heads * dv;
    float* d_beta = d_g + static_cast<size_t>(n) * n_heads;
    float* d_out = d_beta + static_cast<size_t>(n) * n_heads;

    const size_t q_bytes = sizeof(float) * static_cast<size_t>(n) * n_heads * dk;
    const size_t v_bytes = sizeof(float) * static_cast<size_t>(n) * n_heads * dv;
    const size_t g_bytes = sizeof(float) * static_cast<size_t>(n) * n_heads;
    if (g_api.cudaMemcpy(d_q, q + static_cast<size_t>(t0) * n_heads * dk, q_bytes,
                         kCudaMemcpyH2D) != kCudaSuccess)
      return fail("q_h2d");
    if (g_api.cudaMemcpy(d_k, k + static_cast<size_t>(t0) * n_heads * dk, q_bytes,
                         kCudaMemcpyH2D) != kCudaSuccess)
      return fail("k_h2d");
    if (g_api.cudaMemcpy(d_v, v + static_cast<size_t>(t0) * n_heads * dv, v_bytes,
                         kCudaMemcpyH2D) != kCudaSuccess)
      return fail("v_h2d");
    if (g_api.cudaMemcpy(d_g, g + static_cast<size_t>(t0) * n_heads, g_bytes, kCudaMemcpyH2D) !=
        kCudaSuccess)
      return fail("g_h2d");
    if (g_api.cudaMemcpy(d_beta, beta + static_cast<size_t>(t0) * n_heads, g_bytes,
                         kCudaMemcpyH2D) != kCudaSuccess)
      return fail("beta_h2d");

    if (fn_seq) {
      int seq_i = n, nh_i = n_heads, dk_i = dk, dv_i = dv;
      float scale_mut = scale;
      void* params[] = {&d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_out,
                        &seq_i, &nh_i, &dk_i, &dv_i, &scale_mut};
      if (!jit_launch(fn_seq, static_cast<unsigned>(n_heads), 1, 1, static_cast<unsigned>(dv), 1, 1,
                      0, params))
        return fail("launch_seq");
    } else {
      for (int t = 0; t < n; ++t) {
        float* qt = d_q + static_cast<size_t>(t) * n_heads * dk;
        float* kt = d_k + static_cast<size_t>(t) * n_heads * dk;
        float* vt = d_v + static_cast<size_t>(t) * n_heads * dv;
        float* gt = d_g + static_cast<size_t>(t) * n_heads;
        float* bt = d_beta + static_cast<size_t>(t) * n_heads;
        float* ot = d_out + static_cast<size_t>(t) * n_heads * dv;
        int dk_i = dk, dv_i = dv;
        float scale_mut = scale;
        void* params[] = {&qt, &kt, &vt, &gt, &bt, &d_state, &ot, &dk_i, &dv_i, &scale_mut};
        if (!jit_launch(fn_step, static_cast<unsigned>(n_heads), 1, 1, static_cast<unsigned>(dv), 1,
                        1, 0, params))
          return fail("launch");
      }
    }
    if (g_api.cudaMemcpy(out + static_cast<size_t>(t0) * n_heads * dv, d_out, v_bytes,
                         kCudaMemcpyD2H) != kCudaSuccess)
      return fail("out_d2h");
  }
  g_gdn_ok += static_cast<uint64_t>(seq);
  return true;
}

const char* gdn_last_error() { return g_gdn_last_err.empty() ? "" : g_gdn_last_err.c_str(); }
uint64_t gdn_fail_count() { return g_gdn_fail; }

void flush_gdn_state_to_host(float* host_state, int n_heads, int dk, int dv) {
  if (!g_enabled || !host_state || n_heads <= 0 || dk <= 0 || dv <= 0) return;
  const size_t state_bytes = sizeof(float) * static_cast<size_t>(n_heads) * dk * dv;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_gdn_state.find(host_state);
  if (it == g_gdn_state.end() || !it->second) return;
  if (g_api.cudaMemcpy(host_state, it->second, state_bytes, kCudaMemcpyD2H) != kCudaSuccess) return;
  g_api.cudaFree(it->second);
  g_gdn_state.erase(it);
}

void invalidate_gdn_state(float* host_state) {
  if (!g_enabled || !host_state) return;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_gdn_state.find(host_state);
  if (it == g_gdn_state.end() || !it->second) return;
  g_api.cudaFree(it->second);
  g_gdn_state.erase(it);
}

void flush_conv_state_to_host(float* host_conv, int conv_dim, int conv_k) {
  if (!g_enabled || !host_conv || conv_dim <= 0 || conv_k <= 0) return;
  const size_t bytes = sizeof(float) * static_cast<size_t>(conv_dim) * conv_k;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_conv_state.find(host_conv);
  if (it == g_conv_state.end() || !it->second) return;
  if (g_api.cudaMemcpy(host_conv, it->second, bytes, kCudaMemcpyD2H) != kCudaSuccess) return;
  g_api.cudaFree(it->second);
  g_conv_state.erase(it);
}

void invalidate_conv_state(float* host_conv) {
  if (!g_enabled || !host_conv) return;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_conv_state.find(host_conv);
  if (it == g_conv_state.end() || !it->second) return;
  g_api.cudaFree(it->second);
  g_conv_state.erase(it);
}

bool try_dwconv_silu_k4_seq(const float* xin, float* host_state, const float* w, float* xout,
                            int seq, int conv_dim) {
  auto fail = [&](const char* why) {
    g_dwconv_last_err = why;
    ++g_dwconv_fail;
    return false;
  };
  if (!g_enabled || !jit_available()) return fail("cuda_or_jit_off");
  if (!g_resident) return fail("not_resident");
  if (!xin || !host_state || !w || !xout || seq <= 0 || conv_dim <= 0) return fail("bad_arg");

  void* fn = get_jit_kernel(kDwconvSrc, "dwconv_silu_k4_seq");
  if (!fn) return fail(g_status.empty() ? "jit_dwconv_seq" : g_status.c_str());

  // Chunk tokens so xin|xout scratch stays bounded (~same idea as GDN seq).
  constexpr int kChunk = 256;
  const size_t state_bytes = sizeof(float) * static_cast<size_t>(conv_dim) * 4;
  const size_t w_bytes = state_bytes;
  const size_t io_bytes = sizeof(float) * static_cast<size_t>(kChunk) * conv_dim * 2;

  std::lock_guard<std::mutex> lock(g_mu);
  auto sit = g_conv_state.find(host_state);
  if (sit == g_conv_state.end()) {
    void* ds = nullptr;
    if (g_api.cudaMalloc(&ds, state_bytes) != kCudaSuccess) return fail("state_malloc");
    if (g_api.cudaMemcpy(ds, host_state, state_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(ds);
      return fail("state_h2d");
    }
    g_conv_state[host_state] = static_cast<float*>(ds);
    sit = g_conv_state.find(host_state);
  }
  float* d_state = sit->second;

  auto wit = g_conv_w_dev.find(w);
  if (wit == g_conv_w_dev.end()) {
    void* dw = nullptr;
    if (g_api.cudaMalloc(&dw, w_bytes) != kCudaSuccess) return fail("w_malloc");
    if (g_api.cudaMemcpy(dw, w, w_bytes, kCudaMemcpyH2D) != kCudaSuccess) {
      g_api.cudaFree(dw);
      return fail("w_h2d");
    }
    g_conv_w_dev[w] = static_cast<float*>(dw);
    wit = g_conv_w_dev.find(w);
  }
  float* d_w = wit->second;

  if (io_bytes > g_dwconv_io_cap) {
    if (g_dwconv_io) g_api.cudaFree(g_dwconv_io);
    void* io = nullptr;
    if (g_api.cudaMalloc(&io, io_bytes) != kCudaSuccess) {
      g_dwconv_io = nullptr;
      g_dwconv_io_cap = 0;
      return fail("io_malloc");
    }
    g_dwconv_io = static_cast<float*>(io);
    g_dwconv_io_cap = io_bytes;
  }
  float* d_xin = g_dwconv_io;
  float* d_xout = g_dwconv_io + static_cast<size_t>(kChunk) * conv_dim;
  const unsigned blk = 256;
  const unsigned grid = (static_cast<unsigned>(conv_dim) + blk - 1) / blk;

  for (int t0 = 0; t0 < seq; t0 += kChunk) {
    const int c = kChunk < (seq - t0) ? kChunk : (seq - t0);
    const size_t x_bytes = sizeof(float) * static_cast<size_t>(c) * conv_dim;
    if (g_api.cudaMemcpy(d_xin, xin + static_cast<size_t>(t0) * conv_dim, x_bytes,
                         kCudaMemcpyH2D) != kCudaSuccess)
      return fail("xin_h2d");
    int seq_i = c, cd_i = conv_dim;
    void* params[] = {&d_xin, &d_state, &d_w, &d_xout, &seq_i, &cd_i};
    if (!jit_launch(fn, grid, 1, 1, blk, 1, 1, 0, params)) return fail("launch");
    if (g_api.cudaMemcpy(xout + static_cast<size_t>(t0) * conv_dim, d_xout, x_bytes,
                         kCudaMemcpyD2H) != kCudaSuccess)
      return fail("xout_d2h");
  }
  // Mirror state so CPU fallback / snapshot stays coherent.
  if (g_api.cudaMemcpy(host_state, d_state, state_bytes, kCudaMemcpyD2H) != kCudaSuccess)
    return fail("state_d2h");
  g_dwconv_last_err.clear();
  ++g_dwconv_ok;
  return true;
}

const char* dwconv_last_error() {
  return g_dwconv_last_err.empty() ? "" : g_dwconv_last_err.c_str();
}

namespace {

bool ensure_mlp_caps(int H, int I) {
  auto grow = [&](void** p, size_t bytes) -> bool {
    if (*p) g_api.cudaFree(*p);
    *p = nullptr;
    return g_api.cudaMalloc(p, bytes) == kCudaSuccess;
  };
  if (H > g_mlp_cap_h) {
    if (!grow(reinterpret_cast<void**>(&g_mlp_x), sizeof(float) * static_cast<size_t>(H)))
      return false;
    if (!grow(reinterpret_cast<void**>(&g_mlp_norm), sizeof(float) * static_cast<size_t>(H)))
      return false;
    if (!grow(reinterpret_cast<void**>(&g_mlp_down), sizeof(float) * static_cast<size_t>(H)))
      return false;
    if (!grow(reinterpret_cast<void**>(&g_mlp_ln), sizeof(uint16_t) * static_cast<size_t>(H)))
      return false;
    g_mlp_cap_h = H;
  }
  if (I > g_mlp_cap_i) {
    if (!grow(reinterpret_cast<void**>(&g_mlp_g), sizeof(float) * static_cast<size_t>(I)))
      return false;
    if (!grow(reinterpret_cast<void**>(&g_mlp_u), sizeof(float) * static_cast<size_t>(I)))
      return false;
    if (!grow(reinterpret_cast<void**>(&g_mlp_mid), sizeof(float) * static_cast<size_t>(I)))
      return false;
    g_mlp_cap_i = I;
  }
  return g_mlp_x && g_mlp_norm && g_mlp_down && g_mlp_ln && g_mlp_g && g_mlp_u && g_mlp_mid;
}

bool ensure_mlp_core(int n) {
  if (n <= g_mlp_cap_core && g_mlp_core) return true;
  if (g_mlp_core) g_api.cudaFree(g_mlp_core);
  g_mlp_core = nullptr;
  g_mlp_cap_core = 0;
  void* v = nullptr;
  if (g_api.cudaMalloc(&v, sizeof(float) * static_cast<size_t>(n)) != kCudaSuccess) return false;
  g_mlp_core = static_cast<float*>(v);
  g_mlp_cap_core = n;
  return true;
}

}  // namespace

bool try_mlp_decode_resident(const float* x, const uint16_t* ln2, const qlwc::Int4View& wgate,
                             const qlwc::Int4View& wup, const qlwc::Int4View& wdown, float* y, int H,
                             int I, float eps, bool ln_is_f16) {
  if (!g_enabled || !g_resident || !jit_available()) return false;
  if (!x || !ln2 || !y || H <= 0 || I <= 0) return false;
  if (wgate.K != H || wup.K != H || wdown.K != I || wgate.M != I || wup.M != I || wdown.M != H)
    return false;

  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  void* fn_silu = get_jit_kernel(kActSrc, "silu_mul");
  void* fn_add = get_jit_kernel(kActSrc, "vec_add");
  if (!fn_rms || !fn_silu || !fn_add) return false;

  const Int4Resident* rg = nullptr;
  const Int4Resident* ru = nullptr;
  const Int4Resident* rd = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, I)) return false;
    rg = ensure_int4_resident(wgate);
    ru = ensure_int4_resident(wup);
    rd = ensure_int4_resident(wdown);
    if (!rg || !ru || !rd) return false;
    if (g_api.cudaMemcpy(g_mlp_x, x, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyH2D) !=
        kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(g_mlp_ln, ln2, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  const unsigned blk = 256;
  void* prm_rms[] = {&g_mlp_x, &g_mlp_ln, &g_mlp_norm, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;

  if (!jit_gemv_int4(static_cast<const uint8_t*>(rg->d_qweight),
                     static_cast<const uint16_t*>(rg->d_scales),
                     static_cast<const uint16_t*>(rg->d_zeros), g_mlp_norm, g_mlp_g, rg->M, rg->K,
                     rg->ng, rg->gs, rg->is_awq, rg->awq_zp))
    return false;
  if (!jit_gemv_int4(static_cast<const uint8_t*>(ru->d_qweight),
                     static_cast<const uint16_t*>(ru->d_scales),
                     static_cast<const uint16_t*>(ru->d_zeros), g_mlp_norm, g_mlp_u, ru->M, ru->K,
                     ru->ng, ru->gs, ru->is_awq, ru->awq_zp))
    return false;

  const unsigned silu_grid = (static_cast<unsigned>(I) + blk - 1) / blk;
  void* prm_silu[] = {&g_mlp_g, &g_mlp_u, &g_mlp_mid, &I};
  if (!jit_launch(fn_silu, silu_grid, 1, 1, blk, 1, 1, 0, prm_silu)) return false;

  if (!jit_gemv_int4(static_cast<const uint8_t*>(rd->d_qweight),
                     static_cast<const uint16_t*>(rd->d_scales),
                     static_cast<const uint16_t*>(rd->d_zeros), g_mlp_mid, g_mlp_down, rd->M, rd->K,
                     rd->ng, rd->gs, rd->is_awq, rd->awq_zp))
    return false;

  const unsigned add_grid = (static_cast<unsigned>(H) + blk - 1) / blk;
  void* prm_add[] = {&g_mlp_x, &g_mlp_down, &g_mlp_norm, &H};
  if (!jit_launch(fn_add, add_grid, 1, 1, blk, 1, 1, 0, prm_add)) return false;

  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(y, g_mlp_norm, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyD2H) ==
         kCudaSuccess;
}

bool try_rmsnorm_gemm_multi_resident(const float* x, const uint16_t* ln1,
                                     const qlwc::Int4View* const* Ws, float* const* ys, int n, int H,
                                     float eps, bool ln_is_f16) {
  if (!g_enabled || !g_resident || !jit_available()) return false;
  if (!x || !ln1 || !Ws || !ys || n < 2 || n > 4 || H <= 0) return false;

  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  if (!fn_rms) return false;
  void* fn_multi = get_jit_kernel(kGemvInt4Src, "gemv_multi4_int4");
  if (!fn_multi) return false;

  const Int4Resident* res[4] = {};
  int total_m = 0;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, H)) return false;  // I unused; keep mid sized at least H
    for (int i = 0; i < n; ++i) {
      if (!Ws[i] || Ws[i]->K != H) return false;
      res[i] = ensure_int4_resident(*Ws[i]);
      if (!res[i]) return false;
      total_m += res[i]->M;
    }
    for (int i = 1; i < n; ++i) {
      if (res[i]->K != res[0]->K || res[i]->ng != res[0]->ng || res[i]->gs != res[0]->gs ||
          res[i]->is_awq != res[0]->is_awq)
        return false;
    }
    if (!ensure_xy(total_m, H, 1)) return false;
    if (g_api.cudaMemcpy(g_mlp_x, x, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyH2D) !=
        kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(g_mlp_ln, ln1, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  const unsigned blk = 256;
  void* prm_rms[] = {&g_mlp_x, &g_mlp_ln, &g_mlp_norm, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;

  const uint8_t* q[4] = {};
  const uint16_t* s[4] = {};
  const uint16_t* z[4] = {};
  float* ydev[4] = {};
  int m[4] = {};
  int offset = 0;
  for (int i = 0; i < n; ++i) {
    q[i] = static_cast<const uint8_t*>(res[i]->d_qweight);
    s[i] = static_cast<const uint16_t*>(res[i]->d_scales);
    z[i] = static_cast<const uint16_t*>(res[i]->d_zeros);
    ydev[i] = reinterpret_cast<float*>(g_dy) + offset;
    m[i] = res[i]->M;
    offset += res[i]->M;
  }
  for (int i = n; i < 4; ++i) {
    q[i] = q[0];
    s[i] = s[0];
    z[i] = z[0];
    ydev[i] = ydev[0];
    m[i] = 0;
  }
  int K = res[0]->K, ng = res[0]->ng, gs = res[0]->gs;
  int is_awq_i = res[0]->is_awq ? 1 : 0;
  int awq_zp_i = res[0]->awq_zp;
  const float* dx = g_mlp_norm;
  void* params[] = {&q[0], &s[0], &z[0], &ydev[0], &m[0], &q[1], &s[1], &z[1], &ydev[1], &m[1],
                    &q[2], &s[2], &z[2], &ydev[2], &m[2], &q[3], &s[3], &z[3], &ydev[3], &m[3],
                    &n,    &K,    &ng,   &gs,      &is_awq_i, &awq_zp_i, &dx};
  constexpr int RPB = 8;
  const int blocks = (total_m + RPB - 1) / RPB;
  const unsigned shmem = sizeof(unsigned short) * RPB * ng * 2;
  if (!jit_launch(fn_multi, static_cast<unsigned>(blocks), 1, 1, RPB * 32, 1, 1, shmem, params))
    return false;

  std::lock_guard<std::mutex> lock(g_mu);
  int off = 0;
  for (int i = 0; i < n; ++i) {
    if (g_api.cudaMemcpy(ys[i], reinterpret_cast<float*>(g_dy) + off,
                         sizeof(float) * static_cast<size_t>(m[i]), kCudaMemcpyD2H) != kCudaSuccess)
      return false;
    off += m[i];
  }
  return true;
}

bool try_out_mlp_resident(const float* residual, const float* core, const qlwc::Int4View& wout,
                          const uint16_t* ln2, const qlwc::Int4View& wgate,
                          const qlwc::Int4View& wup, const qlwc::Int4View& wdown, float* y, int H,
                          int I, float eps, bool ln_is_f16) {
  if (!g_enabled || !g_resident || !jit_available()) return false;
  if (!residual || !core || !ln2 || !y || H <= 0 || I <= 0) return false;
  // wout: y[H] = W[H, K_core] @ core[K_core]
  if (wout.M != H || wgate.K != H || wup.K != H || wdown.K != I || wgate.M != I || wup.M != I ||
      wdown.M != H)
    return false;
  const int Kc = wout.K;
  if (Kc <= 0) return false;

  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  void* fn_silu = get_jit_kernel(kActSrc, "silu_mul");
  void* fn_add = get_jit_kernel(kActSrc, "vec_add");
  if (!fn_rms || !fn_silu || !fn_add) return false;

  const Int4Resident* ro = nullptr;
  const Int4Resident* rg = nullptr;
  const Int4Resident* ru = nullptr;
  const Int4Resident* rd = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, I)) return false;
    if (!ensure_mlp_core(Kc)) return false;
    ro = ensure_int4_resident(wout);
    rg = ensure_int4_resident(wgate);
    ru = ensure_int4_resident(wup);
    rd = ensure_int4_resident(wdown);
    if (!ro || !rg || !ru || !rd) return false;
    if (g_api.cudaMemcpy(g_mlp_x, residual, sizeof(float) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(g_mlp_core, core, sizeof(float) * static_cast<size_t>(Kc),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(g_mlp_ln, ln2, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
  }

  if (!jit_gemv_int4(static_cast<const uint8_t*>(ro->d_qweight),
                     static_cast<const uint16_t*>(ro->d_scales),
                     static_cast<const uint16_t*>(ro->d_zeros), g_mlp_core, g_mlp_down, ro->M, ro->K,
                     ro->ng, ro->gs, ro->is_awq, ro->awq_zp))
    return false;

  const unsigned blk = 256;
  const unsigned add_grid = (static_cast<unsigned>(H) + blk - 1) / blk;
  // post-attn residual in g_mlp_norm = residual + attn_out
  void* prm_add0[] = {&g_mlp_x, &g_mlp_down, &g_mlp_norm, &H};
  if (!jit_launch(fn_add, add_grid, 1, 1, blk, 1, 1, 0, prm_add0)) return false;

  // MLP from g_mlp_norm (copy to g_mlp_x as residual for final add)
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_api.cudaMemcpy(g_mlp_x, g_mlp_norm, sizeof(float) * static_cast<size_t>(H),
                         kCudaMemcpyD2D) != kCudaSuccess)
      return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  void* prm_rms[] = {&g_mlp_norm, &g_mlp_ln, &g_mlp_down, &H, &eps, &is_f16};  // normed in down tmp
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;

  if (!jit_gemv_int4(static_cast<const uint8_t*>(rg->d_qweight),
                     static_cast<const uint16_t*>(rg->d_scales),
                     static_cast<const uint16_t*>(rg->d_zeros), g_mlp_down, g_mlp_g, rg->M, rg->K,
                     rg->ng, rg->gs, rg->is_awq, rg->awq_zp))
    return false;
  if (!jit_gemv_int4(static_cast<const uint8_t*>(ru->d_qweight),
                     static_cast<const uint16_t*>(ru->d_scales),
                     static_cast<const uint16_t*>(ru->d_zeros), g_mlp_down, g_mlp_u, ru->M, ru->K,
                     ru->ng, ru->gs, ru->is_awq, ru->awq_zp))
    return false;

  const unsigned silu_grid = (static_cast<unsigned>(I) + blk - 1) / blk;
  void* prm_silu[] = {&g_mlp_g, &g_mlp_u, &g_mlp_mid, &I};
  if (!jit_launch(fn_silu, silu_grid, 1, 1, blk, 1, 1, 0, prm_silu)) return false;

  if (!jit_gemv_int4(static_cast<const uint8_t*>(rd->d_qweight),
                     static_cast<const uint16_t*>(rd->d_scales),
                     static_cast<const uint16_t*>(rd->d_zeros), g_mlp_mid, g_mlp_down, rd->M, rd->K,
                     rd->ng, rd->gs, rd->is_awq, rd->awq_zp))
    return false;

  void* prm_add1[] = {&g_mlp_x, &g_mlp_down, &g_mlp_norm, &H};
  if (!jit_launch(fn_add, add_grid, 1, 1, blk, 1, 1, 0, prm_add1)) return false;

  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(y, g_mlp_norm, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyD2H) ==
         kCudaSuccess;
}

bool try_rmsnorm_gemm_multi_from_act(const uint16_t* ln1, const qlwc::Int4View* const* Ws,
                                     float* const* ys, int n, float eps, bool ln_is_f16) {
  if (!g_enabled || !g_resident || !g_act_valid || !g_act_h || !ln1 || !Ws || !ys || n < 2 || n > 4)
    return false;
  int H = g_act_h_dim;
  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  void* fn_multi = get_jit_kernel(kGemvInt4Src, "gemv_multi4_int4");
  if (!fn_rms || !fn_multi) return false;

  const Int4Resident* res[4] = {};
  int total_m = 0;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, H)) return false;
    for (int i = 0; i < n; ++i) {
      if (!Ws[i] || Ws[i]->K != H) return false;
      res[i] = ensure_int4_resident(*Ws[i]);
      if (!res[i]) return false;
      total_m += res[i]->M;
    }
    for (int i = 1; i < n; ++i) {
      if (res[i]->K != res[0]->K || res[i]->ng != res[0]->ng || res[i]->gs != res[0]->gs ||
          res[i]->is_awq != res[0]->is_awq)
        return false;
    }
    if (!ensure_xy(total_m, H, 1)) return false;
    if (g_api.cudaMemcpy(g_mlp_ln, ln1, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  const unsigned blk = 256;
  void* prm_rms[] = {&g_act_h, &g_mlp_ln, &g_mlp_norm, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;

  const uint8_t* q[4] = {};
  const uint16_t* s[4] = {};
  const uint16_t* z[4] = {};
  float* ydev[4] = {};
  int m[4] = {};
  int offset = 0;
  for (int i = 0; i < n; ++i) {
    q[i] = static_cast<const uint8_t*>(res[i]->d_qweight);
    s[i] = static_cast<const uint16_t*>(res[i]->d_scales);
    z[i] = static_cast<const uint16_t*>(res[i]->d_zeros);
    ydev[i] = reinterpret_cast<float*>(g_dy) + offset;
    m[i] = res[i]->M;
    offset += res[i]->M;
  }
  for (int i = n; i < 4; ++i) {
    q[i] = q[0];
    s[i] = s[0];
    z[i] = z[0];
    ydev[i] = ydev[0];
    m[i] = 0;
  }
  int K = res[0]->K, ng = res[0]->ng, gs = res[0]->gs;
  int is_awq_i = res[0]->is_awq ? 1 : 0;
  int awq_zp_i = res[0]->awq_zp;
  const float* dx = g_mlp_norm;
  void* params[] = {&q[0], &s[0], &z[0], &ydev[0], &m[0], &q[1], &s[1], &z[1], &ydev[1], &m[1],
                    &q[2], &s[2], &z[2], &ydev[2], &m[2], &q[3], &s[3], &z[3], &ydev[3], &m[3],
                    &n,    &K,    &ng,   &gs,      &is_awq_i, &awq_zp_i, &dx};
  constexpr int RPB = 8;
  const int blocks = (total_m + RPB - 1) / RPB;
  const unsigned shmem = sizeof(unsigned short) * RPB * ng * 2;
  if (!jit_launch(fn_multi, static_cast<unsigned>(blocks), 1, 1, RPB * 32, 1, 1, shmem, params))
    return false;

  std::lock_guard<std::mutex> lock(g_mu);
  int off = 0;
  for (int i = 0; i < n; ++i) {
    if (g_api.cudaMemcpy(ys[i], reinterpret_cast<float*>(g_dy) + off,
                         sizeof(float) * static_cast<size_t>(m[i]), kCudaMemcpyD2H) != kCudaSuccess)
      return false;
    off += m[i];
  }
  return true;
}

bool try_linear_decode_on_act(const uint16_t* ln1, const qlwc::Int4View& wqkv,
                              const qlwc::Int4View& wz, const qlwc::Int4View* wb_i4,
                              const uint16_t* wb_pass, bool wb_is_f16, const qlwc::Int4View* wa_i4,
                              const uint16_t* wa_pass, bool wa_is_f16, const float* conv_w_host,
                              float* conv_state_host, int conv_k, const float* A_log_host,
                              const float* dt_bias_host, float* recurrent_host,
                              const uint16_t* nrm, const qlwc::Int4View* wout_i4,
                              const uint16_t* wout_pass, bool wout_is_f16, const uint16_t* ln2,
                              const qlwc::Int4View& wgate, const qlwc::Int4View& wup,
                              const qlwc::Int4View& wdown, int nk, int nv, int dk, int dv, int I,
                              float eps, bool ln_is_f16, bool nrm_is_f16) {
  if (!g_enabled || !g_resident || !g_act_valid || !g_act_h || !ln1 || !ln2 || !nrm) return false;
  if (!conv_w_host || !conv_state_host || !A_log_host || !dt_bias_host || !recurrent_host)
    return false;
  if (nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || I <= 0 || conv_k != 4) return false;
  if (nv % nk != 0) return false;
  if (!wb_i4 && !wb_pass) return false;
  if (!wa_i4 && !wa_pass) return false;
  if (!wout_i4 && !wout_pass) return false;
  ++g_act_lin_try;

  int H = g_act_h_dim;
  const int key_dim = nk * dk;
  const int value_dim = nv * dv;
  const int conv_dim = key_dim * 2 + value_dim;

  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  void* fn_multi = get_jit_kernel(kGemvInt4Src, "gemv_multi4_int4");
  void* fn_conv = get_jit_kernel(kDwconvSrc, "dwconv_silu_k4");
  void* fn_pack = get_jit_kernel(kActSrc, "gdn_pack_qkv");
  void* fn_prep = get_jit_kernel(kActSrc, "gdn_prep_gb");
  void* fn_gn = get_jit_kernel(kActSrc, "rmsnorm_gated_heads_v2");
  void* fn_gdn = get_jit_kernel(kGdnSrc, "gated_delta_kernel");
  if (!fn_rms || !fn_multi || !fn_conv || !fn_pack || !fn_prep || !fn_gn || !fn_gdn) {
    g_act_lin_last_err = "jit_kernels";
    return false;
  }

  // CPU always passes lp.nrm base with n=dv — shared weight of length dv, NOT nv*dv.
  const size_t nrm_bytes = sizeof(uint16_t) * static_cast<size_t>(dv);
  // Layout: mixed,z,b,a,mixed_c,q,k,v,g,beta,A,dt,cw,nrm
  const size_t floats_need =
      static_cast<size_t>(conv_dim) * 2 + static_cast<size_t>(value_dim) * 2 +
      static_cast<size_t>(nv) * 6 + static_cast<size_t>(nv) * dk * 2 +
      static_cast<size_t>(conv_dim) * static_cast<size_t>(conv_k) +
      (nrm_bytes + sizeof(float) - 1) / sizeof(float);
  const size_t bytes_need = floats_need * sizeof(float);

  const Int4Resident* rq = nullptr;
  const Int4Resident* rz = nullptr;
  const Int4Resident* rb = nullptr;
  const Int4Resident* ra = nullptr;
  float* d_conv_st = nullptr;
  float* d_state = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, I)) {
      g_act_lin_last_err = "mlp_caps";
      return false;
    }
    if (!ensure_mlp_core(value_dim)) {
      g_act_lin_last_err = "mlp_core";
      return false;
    }
    if (bytes_need > g_lin_ws_cap || !g_lin_ws) {
      if (g_lin_ws) g_api.cudaFree(g_lin_ws);
      g_lin_ws = nullptr;
      g_lin_ws_cap = 0;
      void* v = nullptr;
      if (g_api.cudaMalloc(&v, bytes_need) != kCudaSuccess) {
        g_act_lin_last_err = "lin_ws_malloc";
        return false;
      }
      g_lin_ws = static_cast<float*>(v);
      g_lin_ws_cap = bytes_need;
    }
    rq = ensure_int4_resident(wqkv);
    rz = ensure_int4_resident(wz);
    if (!rq || !rz) {
      g_act_lin_last_err = "qkv_z_resident";
      return false;
    }
    if (wb_i4) {
      rb = ensure_int4_resident(*wb_i4);
      if (!rb) return false;
    }
    if (wa_i4) {
      ra = ensure_int4_resident(*wa_i4);
      if (!ra) return false;
    }
    auto cit = g_conv_state.find(conv_state_host);
    if (cit == g_conv_state.end()) {
      void* cs = nullptr;
      const size_t csb = sizeof(float) * static_cast<size_t>(conv_dim) * conv_k;
      if (g_api.cudaMalloc(&cs, csb) != kCudaSuccess) return false;
      if (g_api.cudaMemcpy(cs, conv_state_host, csb, kCudaMemcpyH2D) != kCudaSuccess) {
        g_api.cudaFree(cs);
        return false;
      }
      g_conv_state[conv_state_host] = static_cast<float*>(cs);
      cit = g_conv_state.find(conv_state_host);
    }
    d_conv_st = cit->second;
    auto git = g_gdn_state.find(recurrent_host);
    if (git == g_gdn_state.end()) {
      void* ds = nullptr;
      const size_t sb = sizeof(float) * static_cast<size_t>(nv) * dk * dv;
      if (g_api.cudaMalloc(&ds, sb) != kCudaSuccess) return false;
      if (g_api.cudaMemcpy(ds, recurrent_host, sb, kCudaMemcpyH2D) != kCudaSuccess) {
        g_api.cudaFree(ds);
        return false;
      }
      g_gdn_state[recurrent_host] = static_cast<float*>(ds);
      git = g_gdn_state.find(recurrent_host);
    }
    d_state = git->second;
    if (g_api.cudaMemcpy(g_mlp_ln, ln1, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
  }

  float* d_mixed = g_lin_ws;
  float* d_z = d_mixed + conv_dim;
  float* d_b = d_z + value_dim;
  float* d_a = d_b + nv;
  float* d_mixed_c = d_a + nv;
  float* d_q = d_mixed_c + conv_dim;
  float* d_k = d_q + nv * dk;
  float* d_v = d_k + nv * dk;
  float* d_g = d_v + value_dim;
  float* d_beta = d_g + nv;
  float* d_A = d_beta + nv;
  float* d_dt = d_A + nv;
  float* d_cw = d_dt + nv;
  auto* d_nrm = reinterpret_cast<uint16_t*>(d_cw + conv_dim * conv_k);

  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_api.cudaMemcpy(d_A, A_log_host, sizeof(float) * static_cast<size_t>(nv),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(d_dt, dt_bias_host, sizeof(float) * static_cast<size_t>(nv),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(d_cw, conv_w_host,
                         sizeof(float) * static_cast<size_t>(conv_dim) * conv_k,
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (g_api.cudaMemcpy(d_nrm, nrm, nrm_bytes, kCudaMemcpyH2D) != kCudaSuccess) return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  const unsigned blk = 256;
  void* prm_rms[] = {&g_act_h, &g_mlp_ln, &g_mlp_norm, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) {
    g_act_lin_last_err = "rmsnorm";
    return false;
  }

  auto run_multi = [&](const Int4Resident* const* resv, float* const* ydev, int nproj,
                       int total_m) -> bool {
    const uint8_t* q[4] = {};
    const uint16_t* s[4] = {};
    const uint16_t* z[4] = {};
    float* yd[4] = {};
    int m[4] = {};
    for (int i = 0; i < nproj; ++i) {
      q[i] = static_cast<const uint8_t*>(resv[i]->d_qweight);
      s[i] = static_cast<const uint16_t*>(resv[i]->d_scales);
      z[i] = static_cast<const uint16_t*>(resv[i]->d_zeros);
      yd[i] = ydev[i];
      m[i] = resv[i]->M;
    }
    for (int i = nproj; i < 4; ++i) {
      q[i] = q[0];
      s[i] = s[0];
      z[i] = z[0];
      yd[i] = yd[0];
      m[i] = 0;
    }
    int K = resv[0]->K, ng = resv[0]->ng, gs = resv[0]->gs;
    int is_awq_i = resv[0]->is_awq ? 1 : 0;
  int awq_zp_i = resv[0]->awq_zp;
    const float* dx = g_mlp_norm;
    int nn = nproj;
    void* params[] = {&q[0], &s[0], &z[0], &yd[0], &m[0], &q[1], &s[1], &z[1], &yd[1], &m[1],
                      &q[2], &s[2], &z[2], &yd[2], &m[2], &q[3], &s[3], &z[3], &yd[3], &m[3],
                      &nn,   &K,    &ng,   &gs,      &is_awq_i, &awq_zp_i, &dx};
    constexpr int RPB = 8;
    const int blocks = (total_m + RPB - 1) / RPB;
    const unsigned shmem = sizeof(unsigned short) * RPB * ng * 2;
    return jit_launch(fn_multi, static_cast<unsigned>(blocks), 1, 1, RPB * 32, 1, 1, shmem, params);
  };

  auto gemv_w16_dev = [&](const uint16_t* pass, bool is_f16w, float* d_y, int M) -> bool {
    return gemv_w16_dev_x(pass, is_f16w, g_mlp_norm, d_y, M, H);
  };

  if (rb && ra) {
    if (rq->K != rz->K || rq->ng != rz->ng || rq->gs != rz->gs || rq->is_awq != rz->is_awq ||
        rb->K != rq->K || rb->ng != rq->ng || rb->gs != rq->gs || rb->is_awq != rq->is_awq ||
        ra->K != rq->K || ra->ng != rq->ng || ra->gs != rq->gs || ra->is_awq != rq->is_awq)
      return false;
    const Int4Resident* resv[4] = {rq, rz, rb, ra};
    float* ydev[4] = {d_mixed, d_z, d_b, d_a};
    if (!run_multi(resv, ydev, 4, rq->M + rz->M + rb->M + ra->M)) {
      g_act_lin_last_err = "multi4";
      return false;
    }
  } else {
    if (rq->K != rz->K || rq->ng != rz->ng || rq->gs != rz->gs || rq->is_awq != rz->is_awq)
      return false;
    const Int4Resident* resv[2] = {rq, rz};
    float* ydev[2] = {d_mixed, d_z};
    if (!run_multi(resv, ydev, 2, rq->M + rz->M)) {
      g_act_lin_last_err = "multi2";
      return false;
    }
    if (rb) {
      if (!jit_gemv_int4(static_cast<const uint8_t*>(rb->d_qweight),
                         static_cast<const uint16_t*>(rb->d_scales),
                         static_cast<const uint16_t*>(rb->d_zeros), g_mlp_norm, d_b, rb->M, rb->K,
                         rb->ng, rb->gs, rb->is_awq, rb->awq_zp)) {
        g_act_lin_last_err = "b_int4";
        return false;
      }
    } else if (!gemv_w16_dev(wb_pass, wb_is_f16, d_b, nv)) {
      g_act_lin_last_err = "b_w16";
      return false;
    }
    if (ra) {
      if (!jit_gemv_int4(static_cast<const uint8_t*>(ra->d_qweight),
                         static_cast<const uint16_t*>(ra->d_scales),
                         static_cast<const uint16_t*>(ra->d_zeros), g_mlp_norm, d_a, ra->M, ra->K,
                         ra->ng, ra->gs, ra->is_awq, ra->awq_zp)) {
        g_act_lin_last_err = "a_int4";
        return false;
      }
    } else if (!gemv_w16_dev(wa_pass, wa_is_f16, d_a, nv)) {
      g_act_lin_last_err = "a_w16";
      return false;
    }
  }

  const unsigned conv_grid = (static_cast<unsigned>(conv_dim) + blk - 1) / blk;
  int conv_dim_i = conv_dim;
  void* prm_conv[] = {&d_mixed, &d_conv_st, &d_cw, &d_mixed_c, &conv_dim_i};
  if (!jit_launch(fn_conv, conv_grid, 1, 1, blk, 1, 1, 0, prm_conv)) {
    g_act_lin_last_err = "conv";
    return false;
  }

  const int pack_n = (nv * dk > value_dim) ? nv * dk : value_dim;
  const unsigned pack_grid = (static_cast<unsigned>(pack_n) + blk - 1) / blk;
  int nk_i = nk, nv_i = nv, dk_i = dk, dv_i = dv;
  void* prm_pack[] = {&d_mixed_c, &d_q, &d_k, &d_v, &nk_i, &nv_i, &dk_i, &dv_i};
  if (!jit_launch(fn_pack, pack_grid, 1, 1, blk, 1, 1, 0, prm_pack)) {
    g_act_lin_last_err = "pack";
    return false;
  }

  const unsigned prep_grid = (static_cast<unsigned>(nv) + blk - 1) / blk;
  void* prm_prep[] = {&d_b, &d_a, &d_A, &d_dt, &d_beta, &d_g, &nv_i};
  if (!jit_launch(fn_prep, prep_grid, 1, 1, blk, 1, 1, 0, prm_prep)) {
    g_act_lin_last_err = "prep";
    return false;
  }

  float scale = 1.f / sqrtf(static_cast<float>(dk));
  float* d_out = g_mlp_core;
  void* prm_gdn[] = {&d_q, &d_k, &d_v, &d_g, &d_beta, &d_state, &d_out, &dk_i, &dv_i, &scale};
  if (!jit_launch(fn_gdn, static_cast<unsigned>(nv), 1, 1, static_cast<unsigned>(dv), 1, 1, 0,
                  prm_gdn)) {
    g_act_lin_last_err = "gdn";
    return false;
  }
  ++g_gdn_ok;

  int nrm_f16 = nrm_is_f16 ? 1 : 0;
  int hd = dv;
  void* prm_gn[] = {&d_out, &d_z, &d_nrm, &d_out, &hd, &eps, &nrm_f16};
  if (!jit_launch(fn_gn, static_cast<unsigned>(nv), 1, 1, blk, 1, 1, blk * sizeof(float), prm_gn)) {
    g_act_lin_last_err = "gated_norm";
    return false;
  }

  if (!try_ffn_on_act(nullptr, value_dim, wout_i4, wout_pass, wout_is_f16, ln2, wgate, wup, wdown, I,
                      eps, ln_is_f16)) {
    g_act_lin_last_err = "ffn";
    return false;
  }
  g_act_lin_last_err.clear();
  ++g_act_lin_ok;
  return true;
}


bool try_ffn_on_act(const float* host_core, int core_dim, const qlwc::Int4View* wout_i4,
                    const uint16_t* wout_pass, bool wout_is_f16, const uint16_t* ln2,
                    const qlwc::Int4View& wgate, const qlwc::Int4View& wup,
                    const qlwc::Int4View& wdown, int I, float eps, bool ln_is_f16) {
  if (!g_enabled || !g_resident || !g_act_valid || !g_act_h || !ln2) return false;
  // host_core==nullptr → g_mlp_core already holds device core (Path A linear decode).
  if (!host_core && !g_mlp_core) return false;
  int H = g_act_h_dim;
  if (H <= 0 || I <= 0 || core_dim <= 0) return false;
  if (wgate.K != H || wup.K != H || wdown.K != I || wgate.M != I || wup.M != I || wdown.M != H)
    return false;
  if (wout_i4) {
    if (wout_i4->M != H || wout_i4->K != core_dim) return false;
  } else if (!wout_pass) {
    return false;
  }
  ++g_act_ffn_try;

  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  void* fn_silu = get_jit_kernel(kActSrc, "silu_mul");
  void* fn_add = get_jit_kernel(kActSrc, "vec_add");
  if (!fn_rms || !fn_silu || !fn_add) return false;

  const Int4Resident* ro = nullptr;
  const Int4Resident* rg = nullptr;
  const Int4Resident* ru = nullptr;
  const Int4Resident* rd = nullptr;
  const float* d_wout_fp32 = nullptr;
  const W16Pack* d_wout_pack = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, I)) return false;
    if (!ensure_mlp_core(core_dim)) return false;
    if (wout_i4) {
      ro = ensure_int4_resident(*wout_i4);
      if (!ro) return false;
    } else {
      // Prefer FP32+cublas for out_proj; pack only if inflate won't fit.
      d_wout_fp32 = ensure_w16_fp32(wout_pass, H, core_dim, wout_is_f16);
      if (!d_wout_fp32) {
        d_wout_pack = ensure_w16_pack(wout_pass, H, core_dim, wout_is_f16);
        if (!d_wout_pack) return false;
      }
    }
    rg = ensure_int4_resident(wgate);
    ru = ensure_int4_resident(wup);
    rd = ensure_int4_resident(wdown);
    if (!rg || !ru || !rd) return false;
    if (host_core) {
      if (g_api.cudaMemcpy(g_mlp_core, host_core, sizeof(float) * static_cast<size_t>(core_dim),
                           kCudaMemcpyH2D) != kCudaSuccess)
        return false;
    }
    if (g_api.cudaMemcpy(g_mlp_ln, ln2, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
  }

  if (ro) {
    if (!jit_gemv_int4(static_cast<const uint8_t*>(ro->d_qweight),
                       static_cast<const uint16_t*>(ro->d_scales),
                       static_cast<const uint16_t*>(ro->d_zeros), g_mlp_core, g_mlp_down, ro->M,
                       ro->K, ro->ng, ro->gs, ro->is_awq, ro->awq_zp))
      return false;
  } else if (d_wout_fp32) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_xy(H, core_dim, 1)) return false;
    const float alpha = 1.f, beta = 0.f;
    if (g_api.cublasSgemm(g_cublas, kCublasOpT, kCublasOpN, H, 1, core_dim, &alpha, d_wout_fp32,
                          core_dim, g_mlp_core, core_dim, &beta, g_mlp_down, H) != kCublasSuccess)
      return false;
  } else {
    if (!d_wout_pack) return false;
    if (!jit_gemv_w16(static_cast<const uint16_t*>(d_wout_pack->d_w), g_mlp_core, g_mlp_down, H,
                      core_dim, d_wout_pack->is_f16))
      return false;
  }

  const unsigned blk = 256;
  const unsigned add_grid = (static_cast<unsigned>(H) + blk - 1) / blk;
  // g_mlp_norm = g_act_h + attn_out
  void* prm_add0[] = {&g_act_h, &g_mlp_down, &g_mlp_norm, &H};
  if (!jit_launch(fn_add, add_grid, 1, 1, blk, 1, 1, 0, prm_add0)) return false;

  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_api.cudaMemcpy(g_mlp_x, g_mlp_norm, sizeof(float) * static_cast<size_t>(H),
                         kCudaMemcpyD2D) != kCudaSuccess)
      return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  void* prm_rms[] = {&g_mlp_norm, &g_mlp_ln, &g_mlp_down, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;

  if (!jit_gemv_int4(static_cast<const uint8_t*>(rg->d_qweight),
                     static_cast<const uint16_t*>(rg->d_scales),
                     static_cast<const uint16_t*>(rg->d_zeros), g_mlp_down, g_mlp_g, rg->M, rg->K,
                     rg->ng, rg->gs, rg->is_awq, rg->awq_zp))
    return false;
  if (!jit_gemv_int4(static_cast<const uint8_t*>(ru->d_qweight),
                     static_cast<const uint16_t*>(ru->d_scales),
                     static_cast<const uint16_t*>(ru->d_zeros), g_mlp_down, g_mlp_u, ru->M, ru->K,
                     ru->ng, ru->gs, ru->is_awq, ru->awq_zp))
    return false;

  const unsigned silu_grid = (static_cast<unsigned>(I) + blk - 1) / blk;
  void* prm_silu[] = {&g_mlp_g, &g_mlp_u, &g_mlp_mid, &I};
  if (!jit_launch(fn_silu, silu_grid, 1, 1, blk, 1, 1, 0, prm_silu)) return false;

  if (!jit_gemv_int4(static_cast<const uint8_t*>(rd->d_qweight),
                     static_cast<const uint16_t*>(rd->d_scales),
                     static_cast<const uint16_t*>(rd->d_zeros), g_mlp_mid, g_mlp_down, rd->M, rd->K,
                     rd->ng, rd->gs, rd->is_awq, rd->awq_zp))
    return false;

  // Write back into persistent residual: g_act_h = g_mlp_x + down
  void* prm_add1[] = {&g_mlp_x, &g_mlp_down, &g_act_h, &H};
  if (!jit_launch(fn_add, add_grid, 1, 1, blk, 1, 1, 0, prm_add1)) return false;
  g_act_valid = true;
  ++g_act_ffn_ok;
  return true;
}

bool try_lm_head_w16_from_act(const uint16_t* final_norm, const uint16_t* lm_pass, int V,
                              float* logits_host, float eps, bool ln_is_f16, bool lm_is_f16) {
  if (!g_enabled || !g_resident || !g_act_valid || !g_act_h || !final_norm || !lm_pass ||
      !logits_host || V <= 0)
    return false;
  int H = g_act_h_dim;
  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  if (!fn_rms) return false;

  const W16Pack* pack = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, H)) return false;
    pack = ensure_w16_pack(lm_pass, V, H, lm_is_f16);
    if (!pack) return false;
    if (g_api.cudaMemcpy(g_mlp_ln, final_norm, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (!ensure_xy(V, H, 1)) return false;
  }

  int is_f16 = ln_is_f16 ? 1 : 0;
  const unsigned blk = 256;
  void* prm_rms[] = {&g_act_h, &g_mlp_ln, &g_mlp_norm, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;

  if (!gemv_w16_tiled_cublas(static_cast<const uint16_t*>(pack->d_w), g_mlp_norm,
                             reinterpret_cast<float*>(g_dy), V, H, pack->is_f16))
    return false;

  std::lock_guard<std::mutex> lock(g_mu);
  if (g_api.cudaMemcpy(logits_host, g_dy, sizeof(float) * static_cast<size_t>(V),
                       kCudaMemcpyD2H) != kCudaSuccess)
    return false;
  int bad = 0;
  const int probe = (V < 4096) ? V : 4096;
  for (int i = 0; i < probe; ++i) {
    const float v = logits_host[i];
    if (v != v || v > 1e30f || v < -1e30f) ++bad;
  }
  if (bad * 4 > probe) return false;
  ++g_act_lm_ok;
  return true;
}

bool try_lm_head_int4_from_act(const uint16_t* final_norm, const qlwc::Int4View& lm,
                               float* logits_host, float eps, bool ln_is_f16) {
  if (!g_enabled || !g_resident || !g_act_valid || !g_act_h || !final_norm || !logits_host)
    return false;
  int H = g_act_h_dim;
  if (lm.K != H) return false;
  void* fn_rms = get_jit_kernel(kActSrc, "rmsnorm_w16");
  if (!fn_rms) return false;
  const Int4Resident* res = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, H)) return false;
    res = ensure_int4_resident(lm);
    if (!res) return false;
    if (g_api.cudaMemcpy(g_mlp_ln, final_norm, sizeof(uint16_t) * static_cast<size_t>(H),
                         kCudaMemcpyH2D) != kCudaSuccess)
      return false;
    if (!ensure_xy(lm.M, H, 1)) return false;
  }
  int is_f16 = ln_is_f16 ? 1 : 0;
  const unsigned blk = 256;
  void* prm_rms[] = {&g_act_h, &g_mlp_ln, &g_mlp_norm, &H, &eps, &is_f16};
  if (!jit_launch(fn_rms, 1, 1, 1, blk, 1, 1, blk * sizeof(float), prm_rms)) return false;
  if (!jit_gemv_int4(static_cast<const uint8_t*>(res->d_qweight),
                     static_cast<const uint16_t*>(res->d_scales),
                     static_cast<const uint16_t*>(res->d_zeros), g_mlp_norm,
                     reinterpret_cast<float*>(g_dy), res->M, res->K, res->ng, res->gs, res->is_awq, res->awq_zp))
    return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_api.cudaMemcpy(logits_host, g_dy, sizeof(float) * static_cast<size_t>(lm.M),
                       kCudaMemcpyD2H) != kCudaSuccess)
    return false;
  ++g_act_lm_ok;
  return true;
}

void note_full_attn_try() { ++g_act_full_try; }
void note_full_attn_ok() { ++g_act_full_ok; }

bool prefetch_int4_weight(const qlwc::Int4View& W) {
  if (!g_enabled) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (ensure_int4_resident(W)) return true;
  return ensure_int4_device(W) != nullptr;
}

bool pin_int4_weight(const qlwc::Int4View& W) {
  if (!g_enabled || !W.qweight) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  const Int4Resident* res = ensure_int4_resident(W);
  if (!res) return false;
  auto it = g_int4_cache.find(W.qweight);
  if (it == g_int4_cache.end()) return false;
  it->second.pinned = true;
  return true;
}

bool try_moe_ffn_int4(const float* x, int H, int I, const MoeExpertInt4* experts, int n_experts,
                      const qlwc::Int4View* shared_gate, const qlwc::Int4View* shared_up,
                      const qlwc::Int4View* shared_down, float shared_scale, float* y) {
  if (!g_enabled || !jit_available() || !x || !y || !experts || n_experts <= 0 || H <= 0 || I <= 0)
    return false;
  const bool has_shared =
      shared_gate && shared_up && shared_down && shared_gate->qweight && shared_up->qweight &&
      shared_down->qweight && std::fabs(shared_scale) > 0.f;
  if (has_shared) {
    if (shared_gate->M != I || shared_up->M != I || shared_down->M != H || shared_gate->K != H ||
        shared_up->K != H || shared_down->K != I)
      return false;
  }
  for (int i = 0; i < n_experts; ++i) {
    const auto& e = experts[i];
    if (!e.gate || !e.up || !e.down || !e.gate->qweight || !e.up->qweight || !e.down->qweight)
      return false;
    if (e.gate->M != I || e.up->M != I || e.down->M != H || e.gate->K != H || e.up->K != H ||
        e.down->K != I)
      return false;
  }

  void* fn_silu = get_jit_kernel(kActSrc, "silu_mul");
  void* fn_fill = get_jit_kernel(kActSrc, "vec_fill");
  void* fn_axpy = get_jit_kernel(kActSrc, "vec_axpy");
  if (!fn_silu || !fn_fill || !fn_axpy) return false;

  const Int4Resident* rg[64] = {};
  const Int4Resident* ru[64] = {};
  const Int4Resident* rd[64] = {};
  if (n_experts > 64) return false;
  const Int4Resident* sg = nullptr;
  const Int4Resident* su = nullptr;
  const Int4Resident* sd = nullptr;

  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensure_mlp_caps(H, I)) return false;
    if (!ensure_mlp_core(H)) return false;  // per-expert down temp
    for (int i = 0; i < n_experts; ++i) {
      rg[i] = ensure_int4_resident(*experts[i].gate);
      ru[i] = ensure_int4_resident(*experts[i].up);
      rd[i] = ensure_int4_resident(*experts[i].down);
      if (!rg[i] || !ru[i] || !rd[i]) return false;
    }
    if (has_shared) {
      sg = ensure_int4_resident(*shared_gate);
      su = ensure_int4_resident(*shared_up);
      sd = ensure_int4_resident(*shared_down);
      if (!sg || !su || !sd) return false;
    }
    if (g_api.cudaMemcpy(g_mlp_x, x, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyH2D) !=
        kCudaSuccess)
      return false;
    g_sticky_x = x;
    g_sticky_k = H;
  }

  const unsigned blk = 256;
  const unsigned grid_h = (static_cast<unsigned>(H) + blk - 1) / blk;
  const unsigned grid_i = (static_cast<unsigned>(I) + blk - 1) / blk;
  float zero = 0.f;
  void* prm_fill[] = {&g_mlp_down, &zero, &H};
  if (!jit_launch(fn_fill, grid_h, 1, 1, blk, 1, 1, 0, prm_fill)) return false;

  auto run_one = [&](const Int4Resident* g, const Int4Resident* u, const Int4Resident* d,
                     float w) -> bool {
    if (std::fabs(w) < 1e-12f) return true;
    if (!jit_gemv_int4(static_cast<const uint8_t*>(g->d_qweight),
                       static_cast<const uint16_t*>(g->d_scales),
                       static_cast<const uint16_t*>(g->d_zeros), g_mlp_x, g_mlp_g, g->M, g->K, g->ng,
                       g->gs, g->is_awq, g->awq_zp))
      return false;
    if (!jit_gemv_int4(static_cast<const uint8_t*>(u->d_qweight),
                       static_cast<const uint16_t*>(u->d_scales),
                       static_cast<const uint16_t*>(u->d_zeros), g_mlp_x, g_mlp_u, u->M, u->K, u->ng,
                       u->gs, u->is_awq, u->awq_zp))
      return false;
    void* prm_silu[] = {&g_mlp_g, &g_mlp_u, &g_mlp_mid, &I};
    if (!jit_launch(fn_silu, grid_i, 1, 1, blk, 1, 1, 0, prm_silu)) return false;
    if (!jit_gemv_int4(static_cast<const uint8_t*>(d->d_qweight),
                       static_cast<const uint16_t*>(d->d_scales),
                       static_cast<const uint16_t*>(d->d_zeros), g_mlp_mid, g_mlp_core, d->M, d->K,
                       d->ng, d->gs, d->is_awq, d->awq_zp))
      return false;
    float aw = w;
    void* prm_axpy[] = {&g_mlp_core, &g_mlp_down, &aw, &H};
    return jit_launch(fn_axpy, grid_h, 1, 1, blk, 1, 1, 0, prm_axpy);
  };

  for (int i = 0; i < n_experts; ++i) {
    if (!run_one(rg[i], ru[i], rd[i], experts[i].weight)) return false;
  }
  if (has_shared) {
    if (!run_one(sg, su, sd, shared_scale)) return false;
  }

  std::lock_guard<std::mutex> lock(g_mu);
  return g_api.cudaMemcpy(y, g_mlp_down, sizeof(float) * static_cast<size_t>(H), kCudaMemcpyD2H) ==
         kCudaSuccess;
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
