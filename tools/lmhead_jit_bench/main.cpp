// llm-on-cpu :: tools/lmhead_jit_bench/main.cpp
// Micro-bench: INT4 lm_head GEMV (M=1, K=2560 hidden, N=151936 vocab) resident on GPU vs CPU AVX2.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "hal/cuda_backend.h"
#include "hal/int4_ops.h"
#include "common/omp_tune.h"
#include "weights/qlwc_format.h"
#include "weights/qlwc_store.h"

namespace {
// build a synthetic INT4 view: qweight packed (K/2 * M bytes), fp16 scales/zeros [M, ng]
struct Synth {
  std::vector<uint8_t> qweight;     // packed
  std::vector<uint16_t> scales;
  std::vector<uint16_t> zeros;
  int M = 0, K = 0, gs = 32, ng = 0;
  llmoc::qlwc::Int4View view() const {
    llmoc::qlwc::Int4View v{};
    v.qweight = qweight.data();
    v.scales = scales.data();
    v.zeros = zeros.data();
    v.scheme = llmoc::qlwc::Scheme::kGptqAsym;
    v.M = M; v.K = K; v.group_size = gs;
    return v;
  }
};
Synth make_synth(int M, int K, int gs) {
  Synth s; s.M = M; s.K = K; s.gs = gs; s.ng = (K + gs - 1) / gs;
  const int rb = (K + 1) >> 1;
  s.qweight.resize(static_cast<size_t>(M) * rb);
  s.scales.resize(static_cast<size_t>(M) * s.ng);
  s.zeros.resize(static_cast<size_t>(M) * s.ng);
  std::mt19937 rng(7);
  std::uniform_int_distribution<uint32_t> db(0u, 255u);
  for (auto& b : s.qweight) b = static_cast<uint8_t>(db(rng));
  std::uniform_real_distribution<float> df(-3.f, 3.f);
  for (auto& u : s.scales) {
    float f = df(rng);
    // fp16 encode
    uint32_t bits; std::memcpy(&bits, &f, 4);
    int e = (bits >> 23) & 0xff;
    uint32_t m = bits & 0x7fffffu;
    if (e < 113) { u = 0; continue; }
    int se = e - 127 + 15;
    if (se > 30) se = 30;
    uint32_t sh = (bits >> 22) & 0x8000u;
    uint16_t h = static_cast<uint16_t>(sh | (se << 10) | (m >> 13));
    u = h;
  }
  for (auto& u : s.zeros) {
    float f = df(rng) * 0.5f;
    uint32_t bits; std::memcpy(&bits, &f, 4);
    int e = (bits >> 23) & 0xff;
    uint32_t m = bits & 0x7fffffu;
    if (e < 113) { u = 0; continue; }
    int se = e - 127 + 15;
    if (se > 30) se = 30;
    uint32_t sh = (bits >> 22) & 0x8000u;
    uint16_t h = static_cast<uint16_t>(sh | (se << 10) | (m >> 13));
    u = h;
  }
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  int M = 151936;  // Qwen-4B vocab
  int K = 2560;    // Qwen-4B hidden
  int gs = 32;
  int reps = 30;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-M") && i+1 < argc) M = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-K") && i+1 < argc) K = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-gs") && i+1 < argc) gs = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-reps") && i+1 < argc) reps = std::atoi(argv[++i]);
  }

  if (!llmoc::hal::cuda::probe_available()) { std::printf("cuda unavailable\n"); return 2; }
  if (!llmoc::hal::cuda::enable(8ull << 30)) {
    std::printf("enable failed: %s\n", llmoc::hal::cuda::status());
    return 2;
  }
  if (!llmoc::hal::cuda::jit_available()) {
    std::printf("jit unavailable\n"); llmoc::hal::cuda::disable(); return 2;
  }

  auto W = make_synth(M, K, gs).view();
  std::printf("[lmhead_jit_bench] shape M=%d K=%d gs=%d int4_bytes=%.2f MiB\n",
              M, K, gs, W.qweight ? ((M * ((K+1)>>1)) + 0.0) / (1024*1024) : 0.0);

  std::vector<float> x(static_cast<size_t>(K), 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> d(-0.05f, 0.05f);
  for (auto& v : x) v = d(rng);
  std::vector<float> y_cpu(static_cast<size_t>(M), 0.0f), y_gpu(static_cast<size_t>(M), 0.0f);

  // warm up JIT once
  bool jit_ok = true;
  for (int i = 0; i < 3; ++i) {
    if (!llmoc::hal::cuda::try_gemm_int4(x.data(), W, y_gpu.data())) { jit_ok = false; break; }
  }
  using clock = std::chrono::steady_clock;
  double gpu_ms = 0.0;
  if (jit_ok) {
    auto g0 = clock::now();
    for (int i = 0; i < reps; ++i) {
      if (!llmoc::hal::cuda::try_gemm_int4(x.data(), W, y_gpu.data())) { jit_ok = false; break; }
    }
    auto g1 = clock::now();
    gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count() / reps;
    std::printf("  GPU resident   : %.3f ms / call × %d\n", gpu_ms, reps);
  } else {
    std::printf("  GPU resident   : rejected (budget too small or M cap)\n");
  }

  for (int i = 0; i < 1; ++i) llmoc::hal::gemm_int4(x.data(), W, y_cpu.data());
  auto c0 = clock::now();
  for (int i = 0; i < reps; ++i) llmoc::hal::gemm_int4(x.data(), W, y_cpu.data());
  auto c1 = clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count() / reps;
  std::printf("  CPU AVX2       : %.3f ms / call × %d\n", cpu_ms, reps);

  if (jit_ok) {
    double maxdiff = 0.0;
    for (int i = 0; i < M; ++i) {
      double ea = std::fabs(y_cpu[i]), eb = std::fabs(y_gpu[i]);
      double e = std::fabs(ea - eb) / (1.0 + ea);
      if (e > maxdiff) maxdiff = e;
    }
    std::printf("  speedup GPU/CPU = %.2fx (max rel err=%.3e)\n", cpu_ms / gpu_ms, maxdiff);
    if (maxdiff > 0.02f) std::printf("  [WARN] mismatch too high\n");
  }

  llmoc::hal::cuda::disable();
  return 0;
}
