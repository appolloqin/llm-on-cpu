// llm-on-cpu :: tools/gdn_bench/main.cpp
// GDN (gated-delta recurrent) GPU kernel harness: CPU vs GPU(seq / single) 对比 + 计时。
// 独立进程便于定位 sticky illegal access，不拖垮 server。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"

namespace hal = llmoc::hal;

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
static float max_abs(const std::vector<float>& a) {
  float m = 0.f;
  for (float x : a) m = std::max(m, std::fabs(x));
  return m;
}

static void run_case(int seq, int n_heads, int dk, int dv) {
  const size_t q_size = (size_t)seq * n_heads * dk;
  const size_t v_size = (size_t)seq * n_heads * dv;
  const size_t g_size = (size_t)seq * n_heads;
  const size_t st_size = (size_t)n_heads * dk * dv;
  std::vector<float> q(q_size), k(q_size), v(v_size), g(g_size), beta(g_size);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.008f * (float)((int)(i % 13) - 6);
  for (size_t i = 0; i < k.size(); ++i) k[i] = 0.009f * (float)((int)(i % 11) - 5);
  for (size_t i = 0; i < v.size(); ++i) v[i] = 0.01f * (float)((int)(i % 7) - 3);
  for (size_t i = 0; i < g.size(); ++i) g[i] = -0.2f * (float)(i % 5);
  for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.4f + 0.1f * (float)(i % 3);

  std::vector<float> st_cpu(st_size, 0.f), st_gpu(st_size, 0.f);
  std::vector<float> out_cpu(v_size), out_gpu(v_size), out_last(v_size);

  auto t0 = std::chrono::steady_clock::now();
  hal::gated_delta_recurrent(q.data(), k.data(), v.data(), g.data(), beta.data(), st_cpu.data(),
                             out_cpu.data(), seq, n_heads, dk, dv, true);
  auto t1 = std::chrono::steady_clock::now();
  const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // GPU seq
  bool gpu_ok = false;
  double gpu_ms = -1.0;
  if (!hal::cuda::resident_gpu_enabled()) hal::cuda::try_enable_resident_gpu(0);
  if (hal::cuda::resident_gpu_enabled()) {
    auto g0 = std::chrono::steady_clock::now();
    gpu_ok = hal::cuda::try_gated_delta_gpu_seq(q.data(), k.data(), v.data(), g.data(), beta.data(),
                                                st_gpu.data(), out_gpu.data(), seq, n_heads, dk, dv);
    auto g1 = std::chrono::steady_clock::now();
    gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
  } else {
    std::printf("  resident_gpu not enabled (LLMOC_RESIDENT_GPU?) — skip GPU\n");
    return;
  }

  float e_out = -1.f, e_state = -1.f;
  if (gpu_ok) {
    e_out = max_abs_diff(out_cpu, out_gpu);
    e_state = max_abs_diff(st_cpu, st_gpu);
  }
  std::printf(
      "seq=%4d heads=%2d dk=%3d dv=%3d | CPU=%.2fms GPUseq=%.2fms (x%.1f) ok=%-d | out err=%.3e "
      "(max%.2e) state err=%.3e  gdn_fail=%llu last=%s\n",
      seq, n_heads, dk, dv, cpu_ms, gpu_ms >= 0 ? gpu_ms : -1.,
      gpu_ms > 0 ? cpu_ms / gpu_ms : 0., gpu_ok ? 1 : 0, e_out, max_abs(out_cpu), e_state,
      (unsigned long long)hal::cuda::gdn_fail_count(), hal::cuda::gdn_last_error());

  // 单步 decode 语义：最后 token 的 GPU 单步 vs CPU 在同一终态继续
  if (gpu_ok && seq >= 2) {
    const size_t off = (size_t)(seq - 1) * n_heads;
    std::vector<float> st_d(st_size, 0.f), out_d(n_heads * dv);
    std::vector<float> st_c(st_size, 0.f);
    // CPU 一路到最后一步
    hal::gated_delta_recurrent(q.data(), k.data(), v.data(), g.data(), beta.data(), st_c.data(),
                               out_cpu.data(), seq - 1, n_heads, dk, dv, true);
    // GPU 从 seq 起点跑到末 token 之前，再单步跑最后一 token
    std::vector<float> stg(st_size, 0.f);
    if (hal::cuda::try_gated_delta_gpu_seq(q.data(), k.data(), v.data(), g.data(), beta.data(),
                                           stg.data(), out_last.data(), seq - 1, n_heads, dk, dv) &&
        hal::cuda::try_gated_delta_gpu(q.data() + off * dk, k.data() + off * dk,
                                       v.data() + off * dv, g.data() + off, beta.data() + off,
                                       stg.data(), out_d.data(), n_heads, dk, dv)) {
      std::vector<float> st_ref(st_size, 0.f), out_ref(n_heads * dv);
      hal::gated_delta_recurrent(q.data(), k.data(), v.data(), g.data(), beta.data(), st_ref.data(),
                                 out_cpu.data(), seq, n_heads, dk, dv, true);
      const size_t l = (size_t)(seq - 1);
      std::vector<float> last_c(out_cpu.begin() + l * n_heads * dv,
                                out_cpu.begin() + (l + 1) * n_heads * dv);
      const float e_last = max_abs_diff(last_c, out_d);
      const float e_st = max_abs_diff(st_ref, stg);
      std::printf("  decode-last: out err=%.3e state err=%.3e\n", e_last, e_st);
    }
  }
}

int main(int argc, char** argv) {
  if (!hal::cuda::probe_available()) { std::printf("cuda unavailable\n"); return 2; }
  if (!hal::cuda::enable(2ull << 30)) { std::printf("enable failed: %s\n", hal::cuda::status()); return 2; }
  run_case(64, 32, 128, 128);
  run_case(256, 32, 128, 128);
  run_case(1024, 32, 128, 128);
  hal::cuda::disable();
  return 0;
}