// tests/unit/test_glm_forward.cpp
#include "test_main.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "glm/glm_config.h"
#include "glm/glm_flash_model.h"
#include "model/kv_cache.h"

TINY_TEST(Glm, ForwardSelftestGlmq) {
  const std::string path = "models/_glm_selftest.glmq";
  // Prefer pre-generated; skip soft if missing (CI without make_fake)
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "[SKIP] Glm.ForwardSelftestGlmq — run make_fake_glmq first\n");
    return;
  }
  std::fclose(f);

  llmoc::glm::GlmEngineConfig cfg;
  cfg.model_path = path;
  cfg.quant = llmoc::glm::QuantKind::kBf16;
  cfg.mode = llmoc::glm::ExecMode::kPureCpu;

  llmoc::glm::GlmFlashModel model;
  model.load_strict(cfg);
  EXPECT_TRUE(model.meta().layers == 2);
  EXPECT_TRUE(model.meta().hidden == 64);

  llmoc::model::SessionCache cache;
  model.init_cache(cache, 64);
  std::vector<float> logits;
  model.forward({1, 2, 3}, cache, logits, true);
  EXPECT_TRUE(static_cast<int>(logits.size()) == model.meta().vocab);
  bool any = false;
  for (float v : logits) {
    EXPECT_TRUE(std::isfinite(v));
    if (v != 0.f) any = true;
  }
  EXPECT_TRUE(any);

  model.forward({4}, cache, logits, false);
  EXPECT_TRUE(static_cast<int>(logits.size()) == model.meta().vocab);
}
