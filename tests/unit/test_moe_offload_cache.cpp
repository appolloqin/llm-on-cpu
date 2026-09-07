// llm-on-cpu :: tests/unit/test_moe_offload_cache.cpp
#include "test_main.h"

#include <vector>

#include "moe/host_banks.h"
#include "moe/offload_cache.h"
#include "moe/qlwc_expert_bytes.h"

TINY_TEST(MoeOffload, ExpertBytesGeometry) {
  // AWQ (no zeros): H=2048 I=512 gs=128
  const size_t awq = llmoc::moe::expert_storage_bytes(2048, 512, 128, false);
  const size_t gptq = llmoc::moe::expert_storage_bytes(2048, 512, 128, true);
  EXPECT_TRUE(awq > 0);
  EXPECT_TRUE(gptq > awq);
  // ~1.55 MiB/expert AWQ device footprint order-of-magnitude
  const size_t dev = llmoc::moe::expert_device_bytes(2048, 512, 128, false);
  EXPECT_TRUE(dev > (1ull << 20));
  EXPECT_TRUE(dev < (3ull << 20));
}

TINY_TEST(MoeOffload, EnsureLruAndRewrite) {
  llmoc::moe::HostBanksConfig hcfg;
  hcfg.num_layers = 2;
  hcfg.num_experts = 8;
  hcfg.hidden = 64;
  hcfg.intermediate = 32;
  hcfg.group_size = 32;
  hcfg.has_zeros = false;

  llmoc::moe::QlwcExpertHostBanks banks;
  banks.configure(hcfg);
  banks.prepare_synthetic_for_test(2);

  llmoc::moe::OffloadCacheConfig ocfg;
  ocfg.num_layers = 2;
  ocfg.num_experts = 8;
  ocfg.cache_size = 16;  // = 2E, prefill-capable
  ocfg.prefill_overlap = true;
  ocfg.backend = llmoc::moe::MoeBackend::kOffload;

  llmoc::moe::OffloadMoeCache cache;
  cache.configure(ocfg, &banks);
  EXPECT_TRUE(cache.ready());

  int ids[4] = {1, 3, 5, 7};
  cache.ensure_experts(0, ids, 4);
  // Rewritten to slots
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(ids[i] >= 0);
  EXPECT_EQ(static_cast<int>(cache.miss_slots().size()), 4);

  // Second ensure: hits — no new misses
  int ids2[4] = {1, 3, 5, 7};
  cache.ensure_experts(0, ids2, 4);
  EXPECT_EQ(static_cast<int>(cache.miss_slots().size()), 0);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(ids2[i], ids[i]);

  // Views resolve
  llmoc::qlwc::Int4View g, u, d;
  EXPECT_TRUE(cache.views_for_slot(ids[0], g, u, d));
  EXPECT_EQ(g.M, 32);
  EXPECT_EQ(g.K, 64);
}

TINY_TEST(MoeOffload, HybridFetchFrac) {
  llmoc::moe::HostBanksConfig hcfg;
  hcfg.num_layers = 1;
  hcfg.num_experts = 8;
  hcfg.hidden = 32;
  hcfg.intermediate = 16;
  hcfg.group_size = 16;
  hcfg.has_zeros = false;

  llmoc::moe::QlwcExpertHostBanks banks;
  banks.configure(hcfg);
  banks.prepare_synthetic_for_test(1);

  llmoc::moe::OffloadCacheConfig ocfg;
  ocfg.num_layers = 1;
  ocfg.num_experts = 8;
  ocfg.cache_size = 16;
  ocfg.prefill_overlap = true;
  ocfg.backend = llmoc::moe::MoeBackend::kHybrid;
  ocfg.hybrid_fetch_frac = 0.5f;

  llmoc::moe::OffloadMoeCache cache;
  cache.configure(ocfg, &banks);

  int ids[4] = {0, 1, 2, 3};  // all misses
  cache.ensure_experts_hybrid(0, ids, 4);
  EXPECT_EQ(cache.last_fetch_count(), 2);
  EXPECT_EQ(static_cast<int>(cache.miss_slots().size()), 2);
  EXPECT_EQ(static_cast<int>(cache.overflow_experts().size()), 2);
  int n_slot = 0, n_cpu = 0;
  for (int i = 0; i < 4; ++i) {
    if (ids[i] >= 0)
      ++n_slot;
    else
      ++n_cpu;
  }
  EXPECT_EQ(n_slot, 2);
  EXPECT_EQ(n_cpu, 2);
}
