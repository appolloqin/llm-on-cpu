// llm-on-cpu :: tests/unit/test_moe_forward.cpp
#include "test_main.h"

#include <filesystem>
#include <string>

#include "model/moe_model.h"
#include "weights/prefetch_pipeline.h"
#include "weights/weight_manager.h"

namespace fs = std::filesystem;

TINY_TEST(Moe, LoadSelftestIfPresent) {
  const fs::path lwc = "models/_selftest.lwc";
  const fs::path cfg = "models/_selftest-hf/config.json";
  if (!fs::exists(lwc) || !fs::exists(cfg)) {
    std::printf("[skip] models/_selftest.lwc not built — run make_fake_hf + convert\n");
    return;
  }
  llmoc::wt::WeightManager wm;
  llmoc::wt::WeightManager::Config wcfg;
  wcfg.lru_budget_bytes = 64u << 20;
  wm.open(lwc.string(), wcfg);
  EXPECT_TRUE(!wm.header().groups.empty());

  llmoc::wt::ExpertPrefetcher pref;
  llmoc::wt::ExpertPrefetcher::Config pcfg;
  pcfg.slot_bytes = 4u << 20;
  pref.open(lwc.string(), pcfg);

  llmoc::model::MoeModel model;
  model.load(&wm, &pref, cfg.string());
  EXPECT_TRUE(model.meta().is_moe);
  EXPECT_EQ(model.config().n_experts, 2);

  llmoc::model::SessionCache cache;
  model.init_cache(cache, 64);
  std::vector<float> logits;
  model.forward({1, 2, 3}, cache, logits, true);
  EXPECT_TRUE(!logits.empty());
  model.forward({4}, cache, logits, false);
  EXPECT_TRUE(!logits.empty());
}
