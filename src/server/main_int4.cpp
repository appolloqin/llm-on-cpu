// llm-on-cpu :: src/server/main_int4.cpp
// INT4/QLWC 专用入口 —— 不修改原 llmoc_server / BF16 路径。
// M5: hybrid/pure_gpu 可选启用 CUDA（BF16 透传 Linear）；INT4 专家核仍走 CPU。

#include <cstdio>
#include <cstring>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/qwen3_5_int4_model.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/scheduler.h"
#include "server/http_api.h"
#include "weights/qlwc_store.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_int4.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server_int4 --config configs/engine_int4.yaml\n");
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
      LOG_WARN("mode=hybrid_gpu requested but CUDA unavailable — degraded to pure_cpu");
    }
    if (mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) {
      double vram_gb = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      if (!llmoc::hal::cuda::enable(static_cast<size_t>(vram_gb * (1ull << 30)))) {
        throw std::runtime_error(std::string("CUDA enable failed: ") + llmoc::hal::cuda::status());
      }
      llmoc::hal::cuda::log_status();
    }

#if defined(_OPENMP)
    LOG_INFO("OpenMP max_threads=%d", omp_get_max_threads());
    if (omp_get_max_threads() > 64) {
      LOG_WARN("OpenMP threads=%d is high for bandwidth-bound decode; try "
               "OMP_NUM_THREADS=<physical cores>",
               omp_get_max_threads());
    }
#endif
    LOG_INFO("[int4] mode=%s model=%s tokenizer=%s port=%d", llmoc::sched::mode_name(mode),
             cfg.model_path.c_str(), tok_dir.c_str(), cfg.server_port);

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
