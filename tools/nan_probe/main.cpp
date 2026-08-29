// llm-on-cpu :: tools/nan_probe/main.cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/engine_config.h"
#include "model/qwen3_5_int4_model.h"
#include "model/tokenizer_hf.h"
#include "weights/qlwc_store.h"

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_int4.yaml";
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];

  try {
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const std::string tok_dir = cfg.resolve_tokenizer_dir();
    llmoc::qlwc::QlwcStore store;
    store.open(cfg.model_path);
    llmoc::model::HfTokenizer tok;
    tok.load(tok_dir + "/tokenizer.json");
    llmoc::model::Qwen35Int4Model model;
    model.load(&store, tok_dir + "/config.json");

    const std::string prompt =
        llmoc::model::apply_qwen_chat_template({{"user", "hi"}}, true);
    auto ids = tok.encode(prompt);
    std::printf("prompt_tokens=%zu model=%s\nids:", ids.size(), cfg.model_path.c_str());
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");

    llmoc::model::SessionCache cache;
    model.init_cache(cache, 512);
    std::vector<float> logits;
    model.forward(ids, cache, logits, true);

    size_t bad = 0;
    for (float v : logits)
      if (!std::isfinite(v)) ++bad;
    std::printf("logits n=%zu bad=%zu\n", logits.size(), bad);

    std::vector<int> idx(std::min<size_t>(logits.size(), 8));
    std::vector<int> all(logits.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
    const int k = 5;
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("top5:");
    for (int i = 0; i < k; ++i) std::printf(" id=%d logit=%.4g", all[i], logits[all[i]]);
    std::printf("\n");
    return bad ? 2 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
