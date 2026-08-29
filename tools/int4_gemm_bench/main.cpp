// llm-on-cpu :: tools/int4_gemm_bench/main.cpp
// 微基准：INT4 / BF16 GEMM 吞吐 → 粗估 4B decode tok/s 上界

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "hal/cpu_ops.h"
#include "hal/int4_ops.h"

using SteadyClock = std::chrono::steady_clock;

static uint16_t f32_to_f16bits(float x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.f = x;
  uint32_t sign = (v.u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((v.u >> 23) & 0xFF) - 127 + 15;
  uint32_t man = v.u & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

static void bench_shape(int M, int K, int iters, int gs) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> x(K), y(M);
  for (float& v : x) v = dist(rng);

  const int rb = (K + 1) / 2;
  const int ng = (K + gs - 1) / gs;
  std::vector<uint8_t> q(static_cast<size_t>(M) * rb);
  std::vector<uint16_t> scales(static_cast<size_t>(M) * ng);
  std::vector<float> scales_f32(static_cast<size_t>(M) * ng, 0.02f);
  for (auto& b : q) b = static_cast<uint8_t>(rng() & 0xFF);
  for (auto& s : scales) s = f32_to_f16bits(0.02f);

  llmoc::qlwc::Int4View W;
  W.qweight = q.data();
  W.scales = scales.data();
  W.scales_f32 = scales_f32.data();
  W.zeros = nullptr;
  W.M = M;
  W.K = K;
  W.group_size = gs;
  W.scheme = llmoc::qlwc::Scheme::kAwqSym;

  for (int i = 0; i < 2; ++i) llmoc::hal::gemm_int4(x.data(), W, y.data());

  auto t0 = SteadyClock::now();
  for (int i = 0; i < iters; ++i) llmoc::hal::gemm_int4(x.data(), W, y.data());
  auto t1 = SteadyClock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);
  const double gflops = (2.0 * M * K) / (ms * 1e6);

  std::printf("[int4_gemm_bench] INT4  M=%7d K=%d  %.3f ms/call  %.1f GFLOP/s\n", M, K, ms, gflops);
}

int main(int argc, char** argv) {
  int M = 0, K = 2560, iters = 20, gs = 128;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--M") && i + 1 < argc) M = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--K") && i + 1 < argc) K = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
  }

#if defined(LLMOC_ENABLE_AVX2)
  const char* isa = "AVX2+FMA";
#else
  const char* isa = "scalar";
#endif
#if defined(_OPENMP)
  const char* omp = "OpenMP";
#else
  const char* omp = "serial";
#endif
  std::printf("[int4_gemm_bench] ISA=%s parallel=%s\n", isa, omp);

  if (M > 0) {
    bench_shape(M, K, iters, gs);
  } else {
    // 默认对照：MLP inter / lm_head vocab
    bench_shape(9216, K, iters, gs);
    bench_shape(248320, K, std::max(3, iters / 4), gs);
  }

  std::printf(
      "[int4_gemm_bench] note: GEMM-only upper bound; not e2e/pure_decode. "
      "30 tok/s SLO assumes SPR+AMX+MTP\n");
  return 0;
}
