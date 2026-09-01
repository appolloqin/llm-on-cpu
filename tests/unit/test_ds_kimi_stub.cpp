// tests/unit/test_ds_kimi_stub.cpp
#include "families/deepseek_v4/ds_stub_model.h"
#include "families/kimi_k3/kimi_stub_model.h"
#include "test_main.h"

TINY_TEST(FamilyStub, DsForwardSynthetic) {
  llmoc::families::deepseek::DsStubModel m;
  m.load_synthetic({}, llmoc::contracts::ExecMode::kPureCpu);
  llmoc::model::SessionCache cache;
  m.init_cache(cache, 64);
  std::vector<float> logits;
  m.forward({1, 2, 3}, cache, logits, true);
  EXPECT_TRUE(static_cast<int>(logits.size()) == m.meta().vocab);
  float s = 0;
  for (float v : logits) s += v * v;
  EXPECT_TRUE(s > 0.f);
}

TINY_TEST(FamilyStub, KimiPureGpuDegradesToLayerStream) {
  EXPECT_TRUE(llmoc::families::kimi::pure_gpu_single_card_does_not_fit(
      llmoc::contracts::ExecMode::kPureGpu, 1));
  bool degraded = false;
  const auto m = llmoc::families::kimi::resolve_kimi_exec_mode(
      llmoc::contracts::ExecMode::kPureGpu, 1, &degraded);
  EXPECT_TRUE(degraded);
  EXPECT_TRUE(m == llmoc::contracts::ExecMode::kLayerStream);

  degraded = true;
  const auto h = llmoc::families::kimi::resolve_kimi_exec_mode(
      llmoc::contracts::ExecMode::kHybridGpu, 1, &degraded);
  EXPECT_TRUE(!degraded);
  EXPECT_TRUE(h == llmoc::contracts::ExecMode::kHybridGpu);
}
