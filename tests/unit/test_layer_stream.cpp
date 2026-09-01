// tests/unit/test_layer_stream.cpp
#include "test_main.h"

#include <string>

#include "contracts/exec_mode.h"
#include "weights/layer_stream.h"
#include "weights/qlwc_store.h"

TINY_TEST(LayerStream, InferLayout) {
  std::vector<std::string> names = {"language_model.embed_tokens.weight",
                                    "language_model.layers.0.mlp.gate_proj.weight",
                                    "language_model.layers.31.mlp.up_proj.weight", "mtp.fc.weight"};
  std::string pref;
  int nl = 0;
  llmoc::wt::infer_layer_layout(names, &pref, &nl);
  EXPECT_TRUE(pref == "language_model.layers.");
  EXPECT_TRUE(nl == 32);
}

TINY_TEST(LayerStream, ParseMode) {
  EXPECT_TRUE(llmoc::contracts::parse_mode("layer_stream") ==
              llmoc::contracts::ExecMode::kLayerStream);
}

TINY_TEST(LayerStream, QlwcLazyOpenIfPresent) {
  const char* path = "models/Qwen3.5-4B.int4.qlwc";
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    // 无权重时跳过（CI 无模型）
    EXPECT_TRUE(true);
    return;
  }
  std::fclose(f);
  auto loader = llmoc::wt::make_layer_stream_loader();
  llmoc::wt::LayerStreamConfig cfg;
  cfg.window_layers = 2;
  cfg.device = "cpu";
  cfg.n_layers = 32;
  cfg.layer_prefix = "language_model.layers.";
  loader->open(path, cfg);
  EXPECT_TRUE(loader->is_qlwc());
  auto v0 = loader->pin_layer(0);
  EXPECT_TRUE(v0.layer_id == 0);
  EXPECT_TRUE(!v0.tensor_names.empty());
  loader->prefetch_layer(1);
  loader->pin_layer(1);
  loader->release_layer(0);
  auto st = loader->stats();
  EXPECT_TRUE(st.pin_loads >= 1);
  loader->close();
}
