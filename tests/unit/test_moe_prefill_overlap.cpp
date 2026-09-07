// llm-on-cpu :: tests/unit/test_moe_prefill_overlap.cpp
#include "test_main.h"

#include "moe/host_banks.h"
#include "moe/offload_cache.h"

TINY_TEST(MoePrefill, DoubleBufferStateMachine) {
  llmoc::moe::HostBanksConfig hcfg;
  hcfg.num_layers = 4;
  hcfg.num_experts = 4;
  hcfg.hidden = 32;
  hcfg.intermediate = 16;
  hcfg.group_size = 16;
  hcfg.has_zeros = false;

  llmoc::moe::QlwcExpertHostBanks banks;
  banks.configure(hcfg);
  banks.prepare_synthetic_for_test(4);

  llmoc::moe::OffloadCacheConfig ocfg;
  ocfg.num_layers = 4;
  ocfg.num_experts = 4;
  ocfg.cache_size = 8;  // 2 * E
  ocfg.prefill_overlap = true;
  ocfg.backend = llmoc::moe::MoeBackend::kOffload;

  llmoc::moe::OffloadMoeCache cache;
  cache.configure(ocfg, &banks);
  EXPECT_TRUE(cache.ready());

  cache.begin_prefill();
  EXPECT_TRUE(cache.prefill_active());

  // Prefetch L0 into buf0, L1 into buf1 (API flips buffer each call).
  cache.prefetch_prefill_layer(0);
  cache.prefetch_prefill_layer(1);
  cache.wait_prefill_layer(0);

  // Slot 0 should hold layer0 expert0
  llmoc::qlwc::Int4View g, u, d;
  EXPECT_TRUE(cache.views_for_slot(0, g, u, d));

  cache.release_prefill_layer(0);
  // After release, layer0 mapping cleared from its buffer half.
  cache.release_prefill_layer(1);

  cache.prefetch_prefill_layer(2);
  cache.wait_prefill_layer(2);
  EXPECT_TRUE(cache.views_for_slot(0, g, u, d) || cache.views_for_slot(4, g, u, d));
}

TINY_TEST(MoePrefill, RequiresTwoE) {
  llmoc::moe::HostBanksConfig hcfg;
  hcfg.num_layers = 1;
  hcfg.num_experts = 8;
  hcfg.hidden = 16;
  hcfg.intermediate = 8;
  hcfg.group_size = 8;
  hcfg.has_zeros = false;

  llmoc::moe::QlwcExpertHostBanks banks;
  banks.configure(hcfg);
  banks.prepare_synthetic_for_test(1);

  llmoc::moe::OffloadCacheConfig ocfg;
  ocfg.num_layers = 1;
  ocfg.num_experts = 8;
  ocfg.cache_size = 8;  // < 2E
  ocfg.prefill_overlap = true;
  ocfg.backend = llmoc::moe::MoeBackend::kOffload;

  llmoc::moe::OffloadMoeCache cache;
  bool threw = false;
  try {
    cache.configure(ocfg, &banks);
  } catch (...) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
