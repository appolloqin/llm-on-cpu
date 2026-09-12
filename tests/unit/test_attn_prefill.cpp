// llm-on-cpu :: tests/unit/test_attn_prefill.cpp
// FlashPrefill-V2 kernel correctness:
//  - attn_prefill_flash(dense) == 独立 double 标量参考（紧容差）
//  - sparse(tau=14) 相对参考误差有界（均值校正不破坏 softmax 结果）
//  - Tensor-core FP16 长 prefill GEMM (INT4→FP16→cublasGemmEx) vs CPU gemm_int4_batch
#include <cmath>
#include <cstdlib>
#include <vector>

#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"
#include "hal/int4_ops.h"
#include "weights/qlwc_format.h"
#include "test_main.h"

#if defined(_WIN32)
#include <process.h>
namespace {
void set_env(const char* k, const char* v) { _putenv_s(k, v); }
}  // namespace
#else
#include <cstdlib>
namespace {
void set_env(const char* k, const char* v) { setenv(k, v, 1); }
}  // namespace
#endif

using namespace llmoc;

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
static float max_abs(const std::vector<float>& a) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i]));
  return m;
}

// 独立标量参考（double）：完全不依�?GPU/现有 CPU 实现
static void ref_attn(const float* q, const float* k, const float* v, float* out, int seq, int nh,
                     int nkv, int hd, float scale) {
  const int g = nh / nkv;
  std::vector<double> s(static_cast<size_t>(seq));
  for (int tq = 0; tq < seq; ++tq) {
    for (int h = 0; h < nh; ++h) {
      const int hkv = h / g;
      const float* qh = q + ((size_t)tq * nh + h) * hd;
      double mx = -1e300;
      for (int tk = 0; tk <= tq; ++tk) {
        const float* kt = k + ((size_t)tk * nkv + hkv) * hd;
        double dot = 0.0;
        for (int d = 0; d < hd; ++d) dot += (double)qh[d] * kt[d];
        const float sc = (float)(dot * (double)scale);
        s[static_cast<size_t>(tk)] = sc;
        mx = mx > (double)sc ? mx : (double)sc;
      }
      double sum = 0.0;
      for (int tk = 0; tk <= tq; ++tk) {
        s[static_cast<size_t>(tk)] = std::exp(s[static_cast<size_t>(tk)] - mx);
        sum += s[static_cast<size_t>(tk)];
      }
      float* oh = out + ((size_t)tq * nh + h) * hd;
      for (int d = 0; d < hd; ++d) { double acc = 0.0;
        for (int tk = 0; tk <= tq; ++tk) {
          const float* vt = v + ((size_t)tk * nkv + hkv) * hd;
          acc += s[static_cast<size_t>(tk)] * (double)vt[d];
        }
        oh[d] = (float)(acc / sum);
      }
    }
  }
}

static bool gpu_run(const float* q, const float* k, const float* v, float* out, int seq, int nh,
                    int nkv, int hd, float scale, float tau, float mc) {
  set_env("LLMOC_ATTN_PREFILL", "flash");
  if (tau < 1.f)
    return hal::cuda::try_attn_prefill_sparse(q, k, v, out, seq, nh, nkv, hd, scale, tau, mc);
  return hal::cuda::try_attn_prefill(q, k, v, out, seq, nh, nkv, hd, scale);
}

static void run_case(int seq, int nh, int nkv, int hd) {
  if (!hal::cuda::probe_available()) return;
  if (!hal::cuda::enabled() && !hal::cuda::enable(1ull << 26)) return;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  std::vector<float> q(static_cast<size_t>(seq) * nh * hd),
      k(static_cast<size_t>(seq) * nkv * hd), v(static_cast<size_t>(seq) * nkv * hd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.01f * (float)((int)(i % 11) - 5);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.02f * (float)((int)(i % 7) - 3);
    v[i] = 0.015f * (float)((int)(i % 5) - 2);
  }
  std::vector<float> ref(q.size()), flash(q.size()), sparse(q.size());
  ref_attn(q.data(), k.data(), v.data(), ref.data(), seq, nh, nkv, hd, scale);

  EXPECT_TRUE(gpu_run(q.data(), k.data(), v.data(), flash.data(), seq, nh, nkv, hd, scale, 1e9f,
                      0.f));
  const float a = max_abs(ref);
  const float e = max_abs_diff(ref, flash);
  if (!(e < 2e-3f * (1.f + a)))
    std::fprintf(stderr, "AttnFlash case(%d,%d,%d,%d) ref-flash err=%.5g max=%.5g\n", seq, nh, nkv,
                 hd, e, a);
  EXPECT_TRUE(e < 2e-3f * (1.f + a));

  if (gpu_run(q.data(), k.data(), v.data(), sparse.data(), seq, nh, nkv, hd, scale, 14.f, 0.f)) {
    const float es = max_abs_diff(ref, sparse);
    if (!(es < 5e-2f * (1.f + a)))
      std::fprintf(stderr, "AttnFlash case(%d,%d,%d,%d) ref-sparse err=%.5g max=%.5g\n", seq, nh, nkv,
                   hd, es, a);
    EXPECT_TRUE(es < 5e-2f * (1.f + a));
  }
}

TINY_TEST(Attn, FlashPrefillMatchesRef) {
  run_case(48, 4, 2, 32);
  run_case(64, 4, 2, 32);
  run_case(128, 8, 4, 64);
  run_case(256, 8, 4, 128);
  run_case(512, 8, 4, 128);
  // Qwen3.5-4B/9B full attention uses head_dim=256 — must tile along hd.
  run_case(64, 4, 2, 256);
  run_case(128, 8, 4, 256);
}

static uint16_t f32_to_f16_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  const int e = (bits >> 23) & 0xff;
  int se = e - 127 + 15;
  const uint32_t m = bits & 0x7fffffu;
  if (e < 113) return 0;
  if (se > 30) se = 30;
  return (uint16_t)(((bits >> 22) & 0x8000u) | ((uint16_t)se << 10) | (m >> 13));
}

static void pack_int4(const std::vector<uint8_t>& q, std::vector<uint8_t>& out, int M, int K) {
  out.assign(static_cast<size_t>(M) * ((K + 1) / 2), 0);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; k += 2) {
      const uint8_t lo = q[static_cast<size_t>(m) * K + k];
      const uint8_t hi = (k + 1 < K) ? q[static_cast<size_t>(m) * K + k + 1] : 0;
      out[static_cast<size_t>(m) * ((K + 1) / 2) + k / 2] = (uint8_t)(lo | (hi << 4));
    }
}

static void tc_case(int M, int K, int gs, bool gptq) {
  if (!hal::cuda::probe_available()) return;
  if (!hal::cuda::enabled() && !hal::cuda::enable(1ull << 26)) return;
  const int ng = (K + gs - 1) / gs;
  std::vector<uint8_t> q(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < q.size(); ++i) q[i] = (uint8_t)((i * 5 + 1) % 16);
  std::vector<uint8_t> packed;
  pack_int4(q, packed, M, K);
  std::vector<uint16_t> scales(static_cast<size_t>(M) * ng), zeros(static_cast<size_t>(M) * ng);
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = f32_to_f16_bits(0.02f * (float)((int)(i % 3) + 1));
    zeros[i] = f32_to_f16_bits(-0.1f * (float)((int)(i % 4)));
  }
  const int n = 32;
  std::vector<float> X(static_cast<size_t>(n) * K), y_cpu((size_t)n * M), y_gpu((size_t)n * M);
  for (size_t i = 0; i < X.size(); ++i) X[i] = 0.03f * (float)((int)(i % 7) - 3);

  llmoc::qlwc::Int4View W;
  W.qweight = packed.data();
  W.scales = scales.data();
  W.zeros = gptq ? zeros.data() : nullptr;
  W.M = M;
  W.K = K;
  W.group_size = gs;
  W.scheme = gptq ? llmoc::qlwc::Scheme::kGptqAsym : llmoc::qlwc::Scheme::kAwqSym;

  for (int t = 0; t < n; ++t)
    hal::gemm_int4(X.data() + static_cast<size_t>(t) * K, W,
                   y_cpu.data() + static_cast<size_t>(t) * M);
  set_env("LLMOC_TC_GEMM", "1");  // tc 默认关，用例显式开启
  const bool tc_ok = hal::cuda::tc_gemm_int4_batch_f16(X.data(), n, W, y_gpu.data());
  set_env("LLMOC_TC_GEMM", "0");
  EXPECT_TRUE(tc_ok);
  const float a = [&] { float m = 0.f; for (auto x : y_cpu) m = std::max(m, std::fabs(x)); return m; }();
  float e = 0.f;
  for (size_t i = 0; i < y_cpu.size(); ++i) e = std::max(e, std::fabs(y_cpu[i] - y_gpu[i]));
  if (!(e < 5e-3f * (1.f + a)))
    std::fprintf(stderr, "TcF16 case(%d,%d,%d,gptq=%d) err=%.5g max=%.5g\n", M, K, gs, gptq ? 1 : 0,
                 e, a);
  EXPECT_TRUE(e < 5e-3f * (1.f + a));
}

TINY_TEST(Attn, DecodeFlashMatchesCpu) {
  if (!hal::cuda::probe_available()) return;
  if (!hal::cuda::enabled() && !hal::cuda::enable(1ull << 26)) return;
  const int nh = 16, nkv = 4, hd = 128, max_seq = 512, seq_len = 400;
  const float scale = 1.f / std::sqrt((float)hd);
  std::vector<float> q((size_t)nh * hd), k((size_t)nkv * max_seq * hd), v((size_t)nkv * max_seq * hd),
      oc((size_t)nh * hd), og((size_t)nh * hd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.01f * (float)((int)(i % 13) - 6);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.02f * (float)((int)(i % 7) - 3);
    v[i] = 0.01f * (float)((int)(i % 5) - 2);
  }
  hal::attn_decode_one(q.data(), k.data(), v.data(), oc.data(), nh, nkv, hd, seq_len, max_seq, scale);
  EXPECT_TRUE(hal::cuda::try_attn_decode_gpu(q.data(), k.data(), v.data(), og.data(), seq_len,
                                             max_seq, nh, nkv, hd, scale));
  const float a = [&]() { float m = 0; for (auto x : oc) m = std::max(m, std::fabs(x)); return m; }();
  float e = 0.f;
  for (size_t i = 0; i < oc.size(); ++i) e = std::max(e, std::fabs(oc[i] - og[i]));
  if (!(e < 2e-3f * (1.f + a))) std::fprintf(stderr, "DecodeFlash err=%.5g max=%.5g\n", e, a);
  EXPECT_TRUE(e < 2e-3f * (1.f + a));
}

TINY_TEST(Attn, TcF16GemmMatchesCpu) {
  tc_case(256, 256, 128, false);
  tc_case(96, 128, 128, true);
  tc_case(512, 256, 64, true);
}
