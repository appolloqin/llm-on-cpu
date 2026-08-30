// llm-on-cpu :: src/server/main.cpp
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/engine_config.h"
#include "common/log.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/moe_model.h"
#include "model/qwen3_5_model.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/placement_planner.h"
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
      std::printf("  modes: pure_cpu | hybrid_gpu | pure_gpu | auto\n");
      return 0;
    }
  }
  try {
    llmoc::log::init(nullptr);
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const std::string tok_dir = cfg.resolve_tokenizer_dir();

    const bool cuda_ok = llmoc::hal::cuda::probe_available();
    bool degraded = false;
    std::string mode_err;
    const auto req = llmoc::sched::parse_mode(cfg.mode);
    const auto mode = llmoc::sched::resolve_mode(req, cuda_ok, &degraded, &mode_err);
    if (req == llmoc::sched::ExecMode::kPureGpu && !cuda_ok) {
      throw std::runtime_error(mode_err.empty() ? "pure_gpu requires CUDA" : mode_err);
    }
    if (degraded) {
      LOG_WARN("mode=hybrid_gpu requested but CUDA unavailable — degraded to pure_cpu (%s)",
               llmoc::hal::cuda::status());
    }

    if (mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) {
      double vram_gb = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      const size_t budget = static_cast<size_t>(vram_gb * (1ull << 30));
      if (!llmoc::hal::cuda::enable(budget)) {
        throw std::runtime_error(std::string("CUDA enable failed: ") + llmoc::hal::cuda::status());
      }
      llmoc::hal::cuda::log_status();
      llmoc::sched::PlacementPlanner::Config pcfg;
      pcfg.vram_bytes = budget;
      pcfg.dram_bytes = static_cast<uint64_t>(cfg.dram_hot_gb * (1ull << 30));
      // Expert hints filled after wm.open when MoE; empty ok for dense.
      auto plan = llmoc::sched::PlacementPlanner::solve(mode, pcfg, {});
      LOG_INFO("placement: %s", plan.summary.c_str());
    }

#if defined(_OPENMP)
    LOG_INFO("OpenMP max_threads=%d", omp_get_max_threads());
    if (omp_get_max_threads() > 64) {
      LOG_WARN("OpenMP threads=%d is high for bandwidth-bound decode; try "
               "set OMP_NUM_THREADS=<physical cores> (often ~half of logical)",
               omp_get_max_threads());
    }
#else
    LOG_INFO("OpenMP: not compiled in");
#endif
    LOG_INFO("mode=%s model=%s tokenizer=%s port=%d dram_hot=%.1fG gpu_vram=%.1fG",
             llmoc::sched::mode_name(mode), cfg.model_path.c_str(), tok_dir.c_str(),
             cfg.server_port, cfg.dram_hot_gb, cfg.gpu_vram_gb);

    llmoc::wt::WeightManager wm;
    llmoc::wt::WeightManager::Config wcfg;
    wcfg.lru_budget_bytes = static_cast<uint64_t>(cfg.dram_hot_gb * (1ull << 30));
    wcfg.io_workers = cfg.io_workers;
    wm.open(cfg.model_path, wcfg);

    // MoE: re-solve placement with expert sizes from LWC groups (informational + log).
    if ((mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) &&
        !wm.header().groups.empty()) {
      std::vector<llmoc::sched::ExpertHint> hints;
      for (const auto& g : wm.header().groups) {
        llmoc::sched::ExpertHint h;
        h.layer = static_cast<int>(g.layer);
        h.expert = static_cast<int>(g.expert_id);
        h.freq = 1.0;
        h.bytes = 0;
        for (const auto& tn : g.tensor_names) {
          try {
            h.bytes += wm.get(tn).size();
          } catch (...) {
          }
        }
        hints.push_back(h);
      }
      llmoc::sched::PlacementPlanner::Config pcfg;
      pcfg.vram_bytes = llmoc::hal::cuda::vram_budget();
      pcfg.dram_bytes = wcfg.lru_budget_bytes;
      auto plan = llmoc::sched::PlacementPlanner::solve(mode, pcfg, hints);
      LOG_INFO("placement(moe): %s", plan.summary.c_str());
    }

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
