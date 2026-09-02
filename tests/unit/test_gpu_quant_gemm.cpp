// llm-on-cpu :: tests/unit/test_gpu_quant_gemm.cpp
#include <cmath>
#include <cstdint>
#include <vector>

#include "hal/cuda_backend.h"
#include "hal/int4_ops.h"
#include "hal/quant_views.h"
#include "glm/hal/glm_awq_int4_ops.h"
#include "glm/hal/glm_nvfp4_ops.h"
#include "test_main.h"

using namespace llmoc;

static void pack_awq_row(const std::vector<uint8_t>& q, std::vector<uint8_t>& out, int M, int K) {
  out.assign(static_cast<size_t>(M) * ((K + 1) / 2), 0);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; k += 2) {
      const uint8_t lo = q[static_cast<size_t>(m) * K + k];
      const uint8_t hi = (k + 1 < K) ? q[static_cast<size_t>(m) * K + k + 1] : 0;
      out[static_cast<size_t>(m) * ((K + 1) / 2) + k / 2] =
          static_cast<uint8_t>(lo | (hi << 4));
    }
  }
}

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

TINY_TEST(GpuQuant, AwqMatchesCpuIfCuda) {
  if (!hal::cuda::probe_available() || !hal::cuda::enable(256ull << 20)) {
    EXPECT_TRUE(true);
    return;
  }
  const int M = 24, K = 64, gs = 64;
  std::vector<uint8_t> q(static_cast<size_t>(M) * K);
  for (int i = 0; i < M * K; ++i) q[i] = static_cast<uint8_t>((i * 3) % 16);
  std::vector<uint8_t> packed;
  pack_awq_row(q, packed, M, K);
  std::vector<uint16_t> scales(M);
  for (int m = 0; m < M; ++m) scales[m] = f32_to_f16bits(0.02f);
  std::vector<float> x(K), y_cpu(M), y_gpu(M);
  for (int k = 0; k < K; ++k) x[k] = 0.01f * static_cast<float>((k % 5) - 2);

  hal::AwqView W;
  W.qweight = packed.data();
  W.scales = scales.data();
  W.M = M;
  W.K = K;
  W.group_size = gs;

  glm::hal::gemm_awq_int4(x.data(), W, y_cpu.data());
  EXPECT_TRUE(hal::cuda::try_gemm_awq(x.data(), W, y_gpu.data()));
  for (int m = 0; m < M; ++m) {
    EXPECT_TRUE(std::fabs(y_gpu[m] - y_cpu[m]) < 1e-3f * (1.f + std::fabs(y_cpu[m])));
  }
  hal::cuda::disable();
}

TINY_TEST(GpuQuant, Nvfp4MatchesCpuIfCuda) {
  if (!hal::cuda::probe_available() || !hal::cuda::enable(256ull << 20)) {
    EXPECT_TRUE(true);
    return;
  }
  const int M = 16, K = 32, gs = 16;
  std::vector<uint8_t> packed(static_cast<size_t>(M) * ((K + 1) / 2));
  for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(i * 17);
  std::vector<uint8_t> scales(static_cast<size_t>(M) * ((K + gs - 1) / gs), 0x38);  // ~1.0 e4m3
  std::vector<float> x(K), y_cpu(M), y_gpu(M);
  for (int k = 0; k < K; ++k) x[k] = 0.05f * static_cast<float>((k % 3) - 1);

  hal::Nvfp4View W;
  W.qweight = packed.data();
  W.scales_fp8 = scales.data();
  W.global_scale = 1.f;
  W.M = M;
  W.K = K;
  W.group_size = gs;

  glm::hal::gemm_nvfp4(x.data(), W, y_cpu.data());
  EXPECT_TRUE(hal::cuda::try_gemm_nvfp4(x.data(), W, y_gpu.data()));
  for (int m = 0; m < M; ++m) {
    EXPECT_TRUE(std::fabs(y_gpu[m] - y_cpu[m]) < 1e-3f * (1.f + std::fabs(y_cpu[m])));
  }
  hal::cuda::disable();
}

TINY_TEST(GpuQuant, JitCompilesAndLaunches) {
  if (!hal::cuda::probe_available() || !hal::cuda::enable(256ull << 20)) {
    EXPECT_TRUE(true);
    return;
  }
  if (!hal::cuda::jit_available()) {
    EXPECT_TRUE(true);  // 无 nvrtc/nvcuda: 优雅降级
    hal::cuda::disable();
    return;
  }
  const char* src = R"CUDA(
extern "C" __global__ void add1(float* x) { x[0] += 1.0f; }
)CUDA";
  void* fn = nullptr;
  EXPECT_TRUE(hal::cuda::jit_compile(src, "add1", &fn));
  EXPECT_TRUE(fn != nullptr);
  float host = 41.f;
  void* d = hal::cuda::device_alloc(sizeof(float));
  EXPECT_TRUE(d != nullptr);
  EXPECT_TRUE(hal::cuda::h2d(d, &host, sizeof(float)));
  void* params[] = {&d};
  EXPECT_TRUE(hal::cuda::jit_launch(fn, 1, 1, 1, 1, 1, 1, 0, params));
  float out = 0.f;
  EXPECT_TRUE(hal::cuda::d2h(&out, d, sizeof(float)));
  EXPECT_TRUE(out == 42.f);
  hal::cuda::device_free(d);
  hal::cuda::disable();
}

// GPU 原生 INT4 dequant-GEMV 与 CPU hal::gemm_int4 对拍(AWQ 对称)
TINY_TEST(GpuQuant, AwqGemvJitMatchesCpu) {
  if (!hal::cuda::probe_available() || !hal::cuda::enable(256ull << 20)) {
    EXPECT_TRUE(true);
    return;
  }
  if (!hal::cuda::jit_available()) {
    EXPECT_TRUE(true);
    hal::cuda::disable();
    return;
  }
  const int M = 24, K = 256, gs = 128;
  const int ng = (K + gs - 1) / gs;
  // 构造量化数据(确定性伪随机)
  std::vector<uint8_t> q(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < q.size(); ++i) q[i] = static_cast<uint8_t>((i * 7 + 3) % 16);
  std::vector<uint8_t> packed;
  pack_awq_row(q, packed, M, K);
  std::vector<uint16_t> scales(static_cast<size_t>(M) * ng);
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = f32_to_f16bits(0.01f * float((i % 5) + 1));
  std::vector<float> x(K), y_cpu(M), y_gpu(M, -1e9f);
  for (int k = 0; k < K; ++k) x[k] = 0.02f * float((k % 11) - 5);

  qlwc::Int4View W;
  W.qweight = packed.data();
  W.scales = scales.data();
  W.zeros = nullptr;
  W.M = M; W.K = K; W.group_size = gs;
  W.scheme = qlwc::Scheme::kAwqSym;
  hal::gemm_int4(x.data(), W, y_cpu.data());

  void* dq = hal::cuda::device_alloc(packed.size());
  void* ds = hal::cuda::device_alloc(scales.size() * 2);
  void* dx = hal::cuda::device_alloc(K * 4);
  void* dy = hal::cuda::device_alloc(M * 4);
  EXPECT_TRUE(dq && ds && dx && dy);
  EXPECT_TRUE(hal::cuda::h2d(dq, packed.data(), packed.size()));
  EXPECT_TRUE(hal::cuda::h2d(ds, scales.data(), scales.size() * 2));
  EXPECT_TRUE(hal::cuda::h2d(dx, x.data(), K * 4));
  EXPECT_TRUE(hal::cuda::jit_gemv_int4(reinterpret_cast<const uint8_t*>(dq),
                                       reinterpret_cast<const uint16_t*>(ds), nullptr,
                                       reinterpret_cast<const float*>(dx),
                                       reinterpret_cast<float*>(dy), M, K, ng, gs, true));
  EXPECT_TRUE(hal::cuda::d2h(y_gpu.data(), dy, M * 4));
  for (int m = 0; m < M; ++m) {
    EXPECT_TRUE(std::fabs(y_gpu[m] - y_cpu[m]) < 1e-3f * (1.f + std::fabs(y_cpu[m])));
  }
  hal::cuda::device_free(dq);
  hal::cuda::device_free(ds);
  hal::cuda::device_free(dx);
  hal::cuda::device_free(dy);
  hal::cuda::disable();
}

// GPTQ 非对称(zeros!=null)变体
TINY_TEST(GpuQuant, GptqGemvJitMatchesCpu) {
  if (!hal::cuda::probe_available() || !hal::cuda::enable(256ull << 20)) {
    EXPECT_TRUE(true);
    return;
  }
  if (!hal::cuda::jit_available()) {
    EXPECT_TRUE(true);
    hal::cuda::disable();
    return;
  }
  const int M = 16, K = 256, gs = 128;
  const int ng = (K + gs - 1) / gs;
  std::vector<uint8_t> q(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < q.size(); ++i) q[i] = static_cast<uint8_t>((i * 5 + 1) % 16);
  std::vector<uint8_t> packed;
  pack_awq_row(q, packed, M, K);
  std::vector<uint16_t> scales(static_cast<size_t>(M) * ng);
  std::vector<uint16_t> zeros(static_cast<size_t>(M) * ng);
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = f32_to_f16bits(0.02f * float((i % 3) + 1));
    zeros[i] = f32_to_f16bits(-0.1f * float((i % 4)));
  }
  std::vector<float> x(K), y_cpu(M), y_gpu(M, -1e9f);
  for (int k = 0; k < K; ++k) x[k] = 0.03f * float((k % 7) - 3);

  qlwc::Int4View W;
  W.qweight = packed.data();
  W.scales = scales.data();
  W.zeros = zeros.data();
  W.M = M; W.K = K; W.group_size = gs;
  W.scheme = qlwc::Scheme::kGptqAsym;
  hal::gemm_int4(x.data(), W, y_cpu.data());

  void* dq = hal::cuda::device_alloc(packed.size());
  void* ds = hal::cuda::device_alloc(scales.size() * 2);
  void* dz = hal::cuda::device_alloc(zeros.size() * 2);
  void* dx = hal::cuda::device_alloc(K * 4);
  void* dy = hal::cuda::device_alloc(M * 4);
  EXPECT_TRUE(dq && ds && dz && dx && dy);
  EXPECT_TRUE(hal::cuda::h2d(dq, packed.data(), packed.size()));
  EXPECT_TRUE(hal::cuda::h2d(ds, scales.data(), scales.size() * 2));
  EXPECT_TRUE(hal::cuda::h2d(dz, zeros.data(), zeros.size() * 2));
  EXPECT_TRUE(hal::cuda::h2d(dx, x.data(), K * 4));
  EXPECT_TRUE(hal::cuda::jit_gemv_int4(reinterpret_cast<const uint8_t*>(dq),
                                       reinterpret_cast<const uint16_t*>(ds),
                                       reinterpret_cast<const uint16_t*>(dz),
                                       reinterpret_cast<const float*>(dx),
                                       reinterpret_cast<float*>(dy), M, K, ng, gs, false));
  EXPECT_TRUE(hal::cuda::d2h(y_gpu.data(), dy, M * 4));
  for (int m = 0; m < M; ++m) {
    EXPECT_TRUE(std::fabs(y_gpu[m] - y_cpu[m]) < 1e-3f * (1.f + std::fabs(y_cpu[m])));
  }
  hal::cuda::device_free(dq);
  hal::cuda::device_free(ds);
  hal::cuda::device_free(dz);
  hal::cuda::device_free(dx);
  hal::cuda::device_free(dy);
  hal::cuda::disable();
}
