// llm-on-cpu :: src/server/main_int4.cpp
// INT4/QLWC 专用入口 —— 不修改原 llmoc_server / BF16 路径。

#include <cstdio>
#include <cstring>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "model/generate.h"
#include "model/qwen3_5_int4_model.h"
#include "model/tokenizer_hf.h"
#include "sched/scheduler.h"
#include "server/http_api.h"
#include "weights/qlwc_store.h"

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_int4.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server_int4 --config configs/engine_int4.yaml\n");
      return 0;
    }
  }
  try {
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const std::string tok_dir = cfg.resolve_tokenizer_dir();
    LOG_INFO("[int4] model=%s tokenizer=%s port=%d", cfg.model_path.c_str(), tok_dir.c_str(),
             cfg.server_port);

    llmoc::qlwc::QlwcStore store;
    store.open(cfg.model_path);

    llmoc::model::HfTokenizer tok;
    tok.load(tok_dir + "/tokenizer.json");

    llmoc::model::Qwen35Int4Model model;
    model.load(&store, tok_dir + "/config.json");

    llmoc::model::Generator gen;
    gen.init(&model, &tok, 4096);

    llmoc::sched::Scheduler sched;
    sched.start(&gen);

    llmoc::server::HttpApi api;
    api.bind(cfg, &sched);
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[int4]: %s\n", e.what());
    return 1;
  }
}
