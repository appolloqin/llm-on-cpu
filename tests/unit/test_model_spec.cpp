// llm-on-cpu :: tests/unit/test_model_spec.cpp
#include "test_main.h"

#include <string>

#include "model/graph/model_spec.h"

TINY_TEST(ModelSpec, LoadRecipe) {
  auto s = llmoc::model::graph::load_model_spec("models/recipes/qwen3_5.json");
  EXPECT_EQ(s.name, std::string("qwen3_5"));
  EXPECT_EQ(s.hidden_size, 2560);
  EXPECT_EQ(s.vocab_size, 248320);
  EXPECT_EQ(static_cast<int>(s.layers.size()), 32);
  int full = 0, lin = 0;
  for (const auto& L : s.layers) {
    if (L.type == "full_attention") ++full;
    else if (L.type == "linear_attention") ++lin;
  }
  EXPECT_EQ(full, 8);
  EXPECT_EQ(lin, 24);
  EXPECT_TRUE(s.rms_one_plus_weight);
}
