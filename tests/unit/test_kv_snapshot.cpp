// llm-on-cpu :: tests/unit/test_kv_snapshot.cpp
#include "test_main.h"

#include <cmath>

#include "model/kv_cache.h"

TINY_TEST(Kv, SnapshotRestore) {
  llmoc::model::SessionCache cache;
  cache.init(/*layers=*/2, /*max_seq=*/32, /*n_kv=*/2, /*hd=*/8, /*n_v=*/4, /*dk=*/4, /*dv=*/4,
             /*conv_dim=*/16, /*conv_k=*/4);
  cache.layer(0).seq = 5;
  cache.layer(1).seq = 5;
  cache.layer(0).linear.has_state = true;
  cache.layer(0).linear.conv[0] = 1.5f;
  cache.layer(0).linear.recurrent[0] = 2.5f;
  cache.tokens = {1, 2, 3, 4, 5};

  const auto snap = cache.snapshot();

  cache.layer(0).seq = 7;
  cache.layer(0).linear.conv[0] = 9.f;
  cache.layer(0).linear.recurrent[0] = 9.f;
  cache.tokens.push_back(6);
  cache.tokens.push_back(7);

  cache.restore(snap);
  EXPECT_EQ(cache.layer(0).seq, 5);
  EXPECT_EQ(cache.layer(1).seq, 5);
  EXPECT_TRUE(cache.layer(0).linear.has_state);
  EXPECT_TRUE(std::fabs(cache.layer(0).linear.conv[0] - 1.5f) < 1e-6f);
  EXPECT_TRUE(std::fabs(cache.layer(0).linear.recurrent[0] - 2.5f) < 1e-6f);
  EXPECT_EQ(static_cast<int>(cache.tokens.size()), 5);
  EXPECT_EQ(cache.tokens.back(), 5);
}
