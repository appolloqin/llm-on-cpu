// llm-on-cpu :: tests/unit/test_gen_stop.cpp
#include "test_main.h"

#include <vector>

#include "model/gen_stop.h"

TINY_TEST(GenStop, LongPhraseCycleDetected) {
  using llmoc::model::find_token_cycle_period;
  using llmoc::model::trim_trailing_token_cycles;

  // period=20 phrase repeated 3 times (typical 「最终状态总结」loop)
  std::vector<int32_t> ids;
  ids.reserve(8 + 20 * 3);
  for (int i = 0; i < 8; ++i) ids.push_back(1000 + i);  // preamble
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < 20; ++i) ids.push_back(10 + i);

  EXPECT_EQ(find_token_cycle_period(ids), 20);

  trim_trailing_token_cycles(ids, 20);
  EXPECT_EQ(static_cast<int>(ids.size()), 8 + 20);
  EXPECT_TRUE(find_token_cycle_period(ids) < 0);
}

TINY_TEST(GenStop, ShortCycleNeedsMoreRepeats) {
  using llmoc::model::find_token_cycle_period;

  // period=4 only 3 repeats → not enough for short-cycle rule
  std::vector<int32_t> short3;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < 4; ++i) short3.push_back(50 + i);
  EXPECT_TRUE(find_token_cycle_period(short3) < 0);

  std::vector<int32_t> short8;
  for (int c = 0; c < 8; ++c)
    for (int i = 0; i < 4; ++i) short8.push_back(50 + i);
  EXPECT_EQ(find_token_cycle_period(short8), 4);
}

TINY_TEST(GenStop, NoFalsePositiveOnNormalText) {
  using llmoc::model::find_token_cycle_period;
  std::vector<int32_t> ids;
  for (int i = 0; i < 64; ++i) ids.push_back(i);
  EXPECT_TRUE(find_token_cycle_period(ids) < 0);
}
