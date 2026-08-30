// llm-on-cpu :: src/server/main.cpp
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "model/generate.h"
#include "model/moe_model.h"
#include "model/qwen3_5_model.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/scheduler.h"
#include "server/http_api.h"
#include "weights/prefetch_pipeline.h"
#include "weights/weight_manager.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server --config configs/engine.yaml\n");
      return 0;
    }
  }
  try {
    llmoc::log::init(nullptr);
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const std::string tok_dir = cfg.resolve_tokenizer_dir();
    (void)llmoc::sched::resolve_mode(llmoc::sched::parse_mode(cfg.mode));
#if defined(_OPENMP)
    LOG_INFO("OpenMP max_threads=%d", omp_get_max_threads());
#else
    LOG_INFO("OpenMP: not compiled in");
#endif
    LOG_INFO("model=%s tokenizer=%s port=%d dram_hot=%.1fG", cfg.model_path.c_str(),
             tok_dir.c_str(), cfg.server_port, cfg.dram_hot_gb);

    llmoc::wt::WeightManager wm;
    llmoc::wt::WeightManager::Config wcfg;
    wcfg.lru_budget_bytes = static_cast<uint64_t>(cfg.dram_hot_gb * (1ull << 30));
    wcfg.io_workers = cfg.io_workers;
    wm.open(cfg.model_path, wcfg);

    llmoc::model::HfTokenizer tok;
    tok.load(tok_dir + "/tokenizer.json");

    const bool is_moe = !wm.header().groups.empty();
    std::unique_ptr<llmoc::wt::ExpertPrefetcher> pref;
    std::unique_ptr<llmoc::model::ICausalLM> model;

    if (is_moe) {
      pref = std::make_unique<llmoc::wt::ExpertPrefetcher>();
      llmoc::wt::ExpertPrefetcher::Config pcfg;
      pcfg.io_workers = cfg.io_workers;
      pcfg.slot_bytes = 128u << 20;
      pref->open(cfg.model_path, pcfg);
      auto moe = std::make_unique<llmoc::model::MoeModel>();
      moe->load(&wm, pref.get(), tok_dir + "/config.json");
      model = std::move(moe);
      LOG_INFO("backend=moe (LWC groups=%zu)", wm.header().groups.size());
    } else {
      auto q = std::make_unique<llmoc::model::Qwen35Model>();
      q->load(&wm, tok_dir + "/config.json");
      model = std::move(q);
      LOG_INFO("backend=qwen3_5 (dense/hybrid)");
    }

    llmoc::model::Generator gen;
    gen.init(model.get(), &tok, 4096);

    llmoc::sched::Scheduler sched;
    sched.start(&gen);

    llmoc::server::HttpApi api;
    api.bind(cfg, &sched);
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 1;
  }
}
