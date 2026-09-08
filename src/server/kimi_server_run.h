// llm-on-cpu :: server/kimi_server_run.h — Kimi MoE stub (expert quant locked by binary)
#pragma once

#include <cstdio>
#include <cstring>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "families/kimi_k3/kimi_stub_model.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/scheduler.h"
#include "server/http_api.h"

namespace llmoc::server::kimi_run {

inline int run(int argc, char** argv, families::deepseek::ExpertQuant force_q, const char* exe_name,
               const char* default_cfg) {
  std::string cfg_path = default_cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: %s --config %s\n", exe_name, default_cfg);
      std::printf("  Kimi MoE stub · expert quant locked to %s\n",
                  families::deepseek::expert_quant_name(force_q));
      std::printf("  siblings: llmoc_server_kimi_bf16 | _int4 | _nvfp4\n");
      std::printf("  note: pure_gpu single-card → auto layer_stream\n");
      return 0;
    }
  }
  try {
    log::init(nullptr);
    auto cfg = EngineConfig::load(cfg_path);
    const auto req = sched::parse_mode(cfg.mode);
    const int world = 1;
    bool degraded = false;
    const auto mode = families::kimi::resolve_kimi_exec_mode(req, world, &degraded);
    if (degraded) {
      LOG_WARN(
          "kimi-k3: pure_gpu single-card ActiveSetDoesNotFit → layer_stream "
          "(run-first; see docs/DESIGN_LAYER_STREAM.md)");
      cfg.mode = "layer_stream";
    }

    if (mode == sched::ExecMode::kHybridGpu || mode == sched::ExecMode::kPureGpu) {
      const double vram = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      if (!hal::cuda::enable(static_cast<size_t>(vram * (1ull << 30)))) {
        LOG_WARN("kimi: CUDA unavailable — CPU GEMM (%s)", hal::cuda::status());
      }
    } else if (mode == sched::ExecMode::kLayerStream &&
               cfg.layer_stream_device.rfind("cuda", 0) == 0) {
      const double vram = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 4.0;
      if (hal::cuda::probe_available() &&
          hal::cuda::enable(static_cast<size_t>(vram * (1ull << 30)))) {
        hal::cuda::log_status();
      }
    }

    families::kimi::KimiStubModel model;
    model.load_file(cfg.model_path, mode, force_q, "kimi_k3_stub");
    if (hal::cuda::enabled()) model.warm_gpu_weights();

    model::HfTokenizer tok;
    try {
      tok.load(cfg.resolve_tokenizer_dir() + "/tokenizer.json");
    } catch (const std::exception& e) {
      LOG_WARN("kimi: tokenizer optional (%s)", e.what());
    }
    model::Generator gen;
    gen.init(&model, &tok, 512);
    sched::Scheduler sched;
    sched.start(&gen);
    server::HttpApi api;
    api.bind(cfg, &sched);
    LOG_INFO("[kimi-%s] listening mode=%s (requested=%s)",
             families::deepseek::expert_quant_name(force_q), sched::mode_name(mode),
             sched::mode_name(req));
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[kimi]: %s\n", e.what());
    return 1;
  }
}

}  // namespace llmoc::server::kimi_run
