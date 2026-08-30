// llm-on-cpu :: tests/unit/test_glm_mode.cpp — GLM 独立模块烟测（不改现有用例）
#include "test_main.h"

#include <cmath>
#include <string>

#include "glm/glm_config.h"
#include "glm/device/glm_device.h"
#include "glm/hal/glm_awq_int4_ops.h"
#include "glm/hal/glm_nvfp4_ops.h"
#include "glm/ops/glm_ops.h"
#include "glm/weights/glm_format.h"

#include <vector>

TINY_TEST(Glm, MagicAndModeNames) {
  EXPECT_TRUE(llmoc::glm::is_glmq_magic(llmoc::glm::kGlmqMagic));
  EXPECT_TRUE(std::string(llmoc::glm::GlmEngineConfig::mode_name(llmoc::glm::ExecMode::kPureCpu)) ==
              "pure_cpu");
  EXPECT_TRUE(std::string(llmoc::glm::GlmEngineConfig::mode_name(llmoc::glm::ExecMode::kHybridGpu)) ==
              "hybrid_gpu");
  EXPECT_TRUE(std::string(llmoc::glm::GlmEngineConfig::mode_name(llmoc::glm::ExecMode::kPureGpu)) ==
              "pure_gpu");
}

TINY_TEST(Glm, MakeDevicePureCpu) {
  bool deg = false;
  auto d = llmoc::glm::make_device(llmoc::glm::ExecMode::kPureCpu, &deg);
  EXPECT_TRUE(d != nullptr);
  EXPECT_TRUE(!deg);
  EXPECT_TRUE(std::string(d->name()) == "cpu");
}

TINY_TEST(Glm, PureGpuRequiresCuda) {
  bool deg = false;
  if (llmoc::glm::cuda_runtime_available()) {
    auto d = llmoc::glm::make_device(llmoc::glm::ExecMode::kPureGpu, &deg);
    EXPECT_TRUE(d != nullptr);
    EXPECT_TRUE(d->caps().has_cuda);
  } else {
    bool threw = false;
    try {
      (void)llmoc::glm::make_device(llmoc::glm::ExecMode::kPureGpu, &deg);
    } catch (...) {
      threw = true;
    }
    EXPECT_TRUE(threw);
  }
}

TINY_TEST(Glm, HybridGpuPreferCuda) {
  bool deg = false;
  auto d = llmoc::glm::make_device(llmoc::glm::ExecMode::kHybridGpu, &deg);
  EXPECT_TRUE(d != nullptr);
  if (llmoc::glm::cuda_runtime_available()) {
    EXPECT_TRUE(!deg);
    EXPECT_TRUE(std::string(d->name()) == "hybrid");
  } else {
    EXPECT_TRUE(deg);
  }
}

TINY_TEST(Glm, KpoolSelectExpands) {
  const int d = 2, pos = 7, pool = 4, topk_pools = 1;
  std::vector<float> q = {1.f, 0.f};
  std::vector<float> kcache(static_cast<size_t>(pos + 1) * d, 0.f);
  // Make last pool strongest
  kcache[static_cast<size_t>(6) * d] = 5.f;
  kcache[static_cast<size_t>(7) * d] = 5.f;
  std::vector<int> idx(16);
  const int n = llmoc::glm::ops::kpool_select(q.data(), kcache.data(), idx.data(), topk_pools, pos,
                                              pos + 1, d, pool);
  EXPECT_TRUE(n >= 1);
  bool has_pos = false;
  for (int i = 0; i < n; ++i)
    if (idx[i] == pos) has_pos = true;
  EXPECT_TRUE(has_pos);
}

TINY_TEST(Glm, AwqAndNvfp4GemmSmoke) {
  float x[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  uint8_t qw[4] = {0x70, 0x70, 0x70, 0x70};  // nibbles
  float scales[1] = {0.01f};
  llmoc::glm::hal::AwqView awq;
  awq.qweight = qw;
  awq.scales_f32 = scales;
  awq.M = 1;
  awq.K = 8;
  awq.group_size = 8;
  float y[1] = {0};
  llmoc::glm::hal::gemm_awq_int4(x, awq, y);
  EXPECT_TRUE(std::isfinite(y[0]));

  llmoc::glm::hal::Nvfp4View nv;
  nv.qweight = qw;
  nv.global_scale = 1.f;
  nv.M = 1;
  nv.K = 8;
  nv.group_size = 8;
  float y2[1] = {0};
  llmoc::glm::hal::gemm_nvfp4(x, nv, y2);
  EXPECT_TRUE(std::isfinite(y2[0]));
}

TINY_TEST(Glm, KdaGatedDeltaStepFinite) {
  const int nh = 2, d = 4;
  std::vector<float> q(nh * d, 0.1f), k(nh * d, 0.2f), v(nh * d, 0.3f);
  std::vector<float> g(nh, -0.5f), beta(nh, 0.5f);
  std::vector<float> state(nh * d * d, 0.f), out(nh * d, 0.f);
  llmoc::glm::ops::kda_gated_delta_step(q.data(), k.data(), v.data(), g.data(), beta.data(),
                                        state.data(), out.data(), nh, d, d);
  for (float o : out) EXPECT_TRUE(std::isfinite(o));
}

TINY_TEST(Glm, IndexerTopkIncludesPos) {
  const int d = 4, pos = 5, topk = 3;
  std::vector<float> q(d, 1.f);
  std::vector<float> kcache(static_cast<size_t>(pos + 1) * d, 0.f);
  kcache[static_cast<size_t>(pos) * d] = 10.f;  // make current strong
  std::vector<int> idx(topk);
  const int n = llmoc::glm::ops::indexer_topk(q.data(), kcache.data(), idx.data(), topk, pos,
                                              pos + 1, d);
  EXPECT_TRUE(n == topk);
  bool has = false;
  for (int i = 0; i < n; ++i)
    if (idx[i] == pos) has = true;
  EXPECT_TRUE(has);
}
