// llm-on-cpu :: hal/cuda_backend.cpp — M5 dynamic CUDA (no nvcc)
#include "hal/cuda_backend.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/log.h"

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
using cublasCreate_t = int (*)(void**);
using cublasDestroy_t = int (*)(void*);
using cublasSgemm_t = int (*)(void*, int, int, int, int, int, const float*, const float*, int,
                              const float*, int, const float*, float*, int);

struct Api {
  void* cudart = nullptr;
  void* cublas = nullptr;
  cudaMalloc_t cudaMalloc = nullptr;
  cudaFree_t cudaFree = nullptr;
  cudaMemcpy_t cudaMemcpy = nullptr;
  cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
  cudaSetDevice_t cudaSetDevice = nullptr;
  cublasCreate_t cublasCreate = nullptr;
  cublasDestroy_t cublasDestroy = nullptr;
  cublasSgemm_t cublasSgemm = nullptr;
};

struct CacheEntry {
  void* d_W = nullptr;
  int M = 0;
  int K = 0;
  size_t bytes = 0;
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
std::unordered_map<const void*, CacheEntry> g_cache;

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
#if defined(_WIN32)
  const char* cudart_names[] = {"cudart64_12.dll", "cudart64_110.dll", nullptr};
  const char* cublas_names[] = {"cublas64_12.dll", "cublas64_11.dll", nullptr};
  const char* dirs[] = {
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin\\",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin\\",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin\\",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin\\",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin\\",
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\",
      "",
      nullptr};
  for (int d = 0; dirs[d]; ++d) {
    for (int i = 0; cudart_names[i] && !g_api.cudart; ++i) {
      std::string p = std::string(dirs[d]) + cudart_names[i];
      g_api.cudart = dirs[d][0] ? load_lib_path(p) : load_lib(cudart_names[i]);
    }
    for (int i = 0; cublas_names[i] && !g_api.cublas; ++i) {
      std::string p = std::string(dirs[d]) + cublas_names[i];
      g_api.cublas = dirs[d][0] ? load_lib_path(p) : load_lib(cublas_names[i]);
    }
    if (g_api.cudart && g_api.cublas) break;
  }
#else
  g_api.cudart = load_lib("libcudart.so.12");
  if (!g_api.cudart) g_api.cudart = load_lib("libcudart.so");
  g_api.cublas = load_lib("libcublas.so.12");
  if (!g_api.cublas) g_api.cublas = load_lib("libcublas.so");
#endif
  if (!g_api.cudart || !g_api.cublas) {
    err = "cudart/cublas not found";
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
  return true;
}

bool ensure_xy(int M, int K) {
  if (K > g_cap_k || !g_dx) {
    if (g_dx) g_api.cudaFree(g_dx);
    if (g_api.cudaMalloc(&g_dx, sizeof(float) * static_cast<size_t>(K)) != kCudaSuccess) return false;
    g_cap_k = K;
  }
  if (M > g_cap_m || !g_dy) {
    if (g_dy) g_api.cudaFree(g_dy);
    if (g_api.cudaMalloc(&g_dy, sizeof(float) * static_cast<size_t>(M)) != kCudaSuccess) return false;
    g_cap_m = M;
  }
  return true;
}

bool gemm_dev(const float* d_W, const float* x, float* y, int M, int K) {
  if (!ensure_xy(M, K)) return false;
  if (g_api.cudaMemcpy(g_dx, x, sizeof(float) * static_cast<size_t>(K), kCudaMemcpyH2D) !=
      kCudaSuccess)
    return false;
  const float alpha = 1.f, beta = 0.f;
  if (g_api.cublasSgemm(g_cublas, kCublasOpT, kCublasOpN, M, 1, K, &alpha, d_W, K,
                        reinterpret_cast<float*>(g_dx), 1, &beta, reinterpret_cast<float*>(g_dy),
                        M) != kCublasSuccess)
    return false;
  return g_api.cudaMemcpy(y, g_dy, sizeof(float) * static_cast<size_t>(M), kCudaMemcpyD2H) ==
         kCudaSuccess;
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
  for (auto& kv : g_cache) {
    if (kv.second.d_W) g_api.cudaFree(kv.second.d_W);
  }
  g_cache.clear();
  if (g_dx) g_api.cudaFree(g_dx);
  if (g_dy) g_api.cudaFree(g_dy);
  g_dx = g_dy = nullptr;
  g_cap_k = g_cap_m = 0;
  if (g_cublas && g_api.cublasDestroy) g_api.cublasDestroy(g_cublas);
  g_cublas = nullptr;
  g_used = 0;
  g_enabled = false;
  g_status = "disabled";
}

bool try_gemm_w16(const float* x, const uint16_t* W, float* y, int M, int K, bool is_f16) {
  if (!g_enabled || !x || !W || !y || M <= 0 || K <= 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_cache.find(W);
  if (it == g_cache.end()) {
    const size_t nbytes = sizeof(float) * static_cast<size_t>(M) * K;
    if (g_budget > 0 && g_used + nbytes > g_budget) return false;
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

void log_status() {
  LOG_INFO("hal.cuda: %s enabled=%d used=%.2fGiB budget=%.2fGiB", g_status.c_str(),
           g_enabled ? 1 : 0, g_used / double(1ull << 30), g_budget / double(1ull << 30));
}

}  // namespace llmoc::hal::cuda
