// llm-on-cpu :: tools/attn_prefill_bench/main.cpp
// FlashPrefill-V2 attention prefill micro-bench: CPU ref vs GPU flash(dense) vs GPU flash(sparse).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#endif

#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"

namespace hal = llmoc::hal;

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

static void bench(int seq, int nh, int nkv, int hd) {
  const int reps = 8;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  std::vector<float> q(static_cast<size_t>(seq) * nh * hd),
      k(static_cast<size_t>(seq) * nkv * hd), v(static_cast<size_t>(seq) * nkv * hd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.01f * (float)((int)(i % 11) - 5);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.02f * (float)((int)(i % 7) - 3);
    v[i] = 0.015f * (float)((int)(i % 5) - 2);
  }
  std::vector<float> cpu(q.size()), gpu_naive(q.size()), gpu_d(q.size()), gpu_s(q.size());
  auto t = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto set_attn_mode = [](const char* m) {
#if defined(_WIN32)
    _putenv_s("LLMOC_ATTN_PREFILL", m);
#else
    setenv("LLMOC_ATTN_PREFILL", m, 1);
#endif
  };

  hal::attn_prefill(q.data(), k.data(), v.data(), cpu.data(), seq, nh, nkv, hd, scale);
  auto c0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
    hal::attn_prefill(q.data(), k.data(), v.data(), cpu.data(), seq, nh, nkv, hd, scale);
  auto c1 = std::chrono::steady_clock::now();
  const double c_ms = t(c0, c1) / reps;

  double gn_ms = -1.0;
  set_attn_mode("naive");
  if (hal::cuda::try_attn_prefill(q.data(), k.data(), v.data(), gpu_naive.data(), seq, nh, nkv, hd,
                                  scale)) {
    auto g0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      hal::cuda::try_attn_prefill(q.data(), k.data(), v.data(), gpu_naive.data(), seq, nh, nkv, hd,
                                  scale);
    auto g1 = std::chrono::steady_clock::now();
    gn_ms = t(g0, g1) / reps;
  }
  double g_ms = -1.0, e_d = -1.0;
  set_attn_mode("flash");
  if (hal::cuda::try_attn_prefill(q.data(), k.data(), v.data(), gpu_d.data(), seq, nh, nkv, hd,
                                  scale)) {
    auto g0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      hal::cuda::try_attn_prefill(q.data(), k.data(), v.data(), gpu_d.data(), seq, nh, nkv, hd,
                                  scale);
    auto g1 = std::chrono::steady_clock::now();
    g_ms = t(g0, g1) / reps;
    e_d = max_abs_diff(gpu_naive, gpu_d);  // flash vs 旧 GPU kernel（紧）
  }
  double gs_ms = -1.0, e_s = -1.0;
  if (hal::cuda::try_attn_prefill_sparse(q.data(), k.data(), v.data(), gpu_s.data(), seq, nh, nkv,
                                         hd, scale, 12.f, 0.f)) {
    auto s0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      hal::cuda::try_attn_prefill_sparse(q.data(), k.data(), v.data(), gpu_s.data(), seq, nh, nkv,
                                         hd, scale, 12.f, 0.f);
    auto s1 = std::chrono::steady_clock::now();
    gs_ms = t(s0, s1) / reps;
    e_s = max_abs_diff(gpu_naive, gpu_s);
  }
  std::printf(
      "seq=%5d hd=%3d | CPU=%.3fms  GPUnaive=%.3fms  flash=%.3fms (x%.1f vs CPU; err_vs_naive="
      "%.2e)  sparse=%.3fms (err_vs_naive=%.2e)\n",
      seq, hd, c_ms, gn_ms >= 0 ? gn_ms : -1., g_ms >= 0 ? g_ms : -1.,
      g_ms > 0 ? c_ms / g_ms : 0., e_d, gs_ms >= 0 ? gs_ms : -1., e_s);
}

int main() {
  if (!hal::cuda::probe_available()) {
    std::printf("cuda unavailable\n");
    return 2;
  }
  if (!hal::cuda::enable(2ull << 30)) {
    std::printf("cuda enable failed: %s\n", hal::cuda::status());
    return 2;
  }
  bench(256, 8, 4, 128);
  bench(512, 8, 4, 128);
  bench(1024, 8, 4, 128);
  bench(2048, 8, 4, 128);
  bench(4096, 8, 4, 128);
  hal::cuda::disable();
  return 0;
}