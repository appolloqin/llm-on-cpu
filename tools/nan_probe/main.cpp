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
  std::string raw_prompt;
  int n_decode = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--raw") && i + 1 < argc) raw_prompt = argv[++i];
    else if (!std::strcmp(argv[i], "--decode") && i + 1 < argc) n_decode = std::atoi(argv[++i]);
  }

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
        raw_prompt.empty()
            ? llmoc::model::apply_qwen_chat_template({{"user", "hi"}}, true)
            : raw_prompt;
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

    // 可选：贪心连解 n_decode 个 token, 打印 id 与文本, 判别模型续写是否退化
    if (n_decode > 0) {
      std::vector<int32_t> cur = ids;
      for (int step = 0; step < n_decode; ++step) {
        int32_t best = 0;
        float bv = -1e30f;
        for (size_t i = 0; i < logits.size(); ++i)
          if (std::isfinite(logits[i]) && logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
        std::printf("  step%d id=%d logit=%.4g\n", step, best, bv);
        cur.push_back(best);
        model.forward({best}, cache, logits, false);
        const auto txt = tok.decode({best}, false);
        std::string esc;
        for (unsigned char c : txt) {
          if (c >= 32 && c < 127) esc += static_cast<char>(c);
          else { char tmp[8]; std::snprintf(tmp, sizeof tmp, "#%d;", static_cast<int>(c)); esc += tmp; }
        }
        std::printf("         text=<%s>\n", esc.c_str());
      }
      const auto all_txt = tok.decode(cur, false);
      std::printf("full=<%s>\n", all_txt.c_str());
    }
    return bad ? 2 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
