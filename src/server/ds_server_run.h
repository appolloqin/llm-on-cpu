// llm-on-cpu :: server/ds_server_run.h — DeepSeek MoE stub (expert quant locked by binary)
#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "families/deepseek_v4/ds_stub_model.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/scheduler.h"
#include "server/http_api.h"

namespace llmoc::server::ds_run {

inline int run(int argc, char** argv, families::deepseek::ExpertQuant force_q, const char* exe_name,
               const char* default_cfg) {
  std::string cfg_path = default_cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: %s --config %s\n", exe_name, default_cfg);
      std::printf("  DeepSeek MoE stub · expert quant locked to %s\n",
                  families::deepseek::expert_quant_name(force_q));
      std::printf("  siblings: llmoc_server_ds_bf16 | _int4 | _nvfp4\n");
      return 0;
    }
  }
  try {
    log::init(nullptr);
    auto cfg = EngineConfig::load(cfg_path);
    const auto mode = sched::parse_mode(cfg.mode);
    if (mode == sched::ExecMode::kHybridGpu || mode == sched::ExecMode::kPureGpu) {
      const double vram = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      if (!hal::cuda::enable(static_cast<size_t>(vram * (1ull << 30)))) {
        if (mode == sched::ExecMode::kPureGpu)
          throw std::runtime_error(std::string("CUDA required: ") + hal::cuda::status());
        LOG_WARN("ds: CUDA unavailable — CPU GEMM (%s)", hal::cuda::status());
      }
    }
    families::deepseek::DsStubModel model;
    model.load_file(cfg.model_path, mode, force_q);
    if (hal::cuda::enabled()) model.warm_gpu_weights();

    model::HfTokenizer tok;
    try {
      tok.load(cfg.resolve_tokenizer_dir() + "/tokenizer.json");
    } catch (const std::exception& e) {
      LOG_WARN("ds: tokenizer optional (%s)", e.what());
    }
    model::Generator gen;
    gen.init(&model, &tok, 512);
    sched::Scheduler sched;
    sched.start(&gen);
    server::HttpApi api;
    api.bind(cfg, &sched);
    LOG_INFO("[ds-%s] listening", families::deepseek::expert_quant_name(force_q));
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[ds]: %s\n", e.what());
    return 1;
  }
}

}  // namespace llmoc::server::ds_run
