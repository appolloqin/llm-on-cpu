// llm-on-cpu :: src/server/main_kimi.cpp — Kimi-K3 stub server
#include <cstdio>
#include <cstring>
#include <stdexcept>
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

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_kimi_hybrid.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server_kimi --config configs/engine_kimi_hybrid.yaml\n");
      std::printf("  Kimi-STUB-v0; pure_gpu single-card → auto layer_stream\n");
      return 0;
    }
  }
  try {
    llmoc::log::init(nullptr);
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const auto req = llmoc::sched::parse_mode(cfg.mode);
    const int world = 1;  // stub: no multi-card yet
    bool degraded = false;
    const auto mode = llmoc::families::kimi::resolve_kimi_exec_mode(req, world, &degraded);
    if (degraded) {
      LOG_WARN(
          "kimi-k3: pure_gpu single-card ActiveSetDoesNotFit → layer_stream "
          "(run-first; see docs/DESIGN_LAYER_STREAM.md)");
      cfg.mode = "layer_stream";
    }

    if (mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) {
      const double vram = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      if (!llmoc::hal::cuda::enable(static_cast<size_t>(vram * (1ull << 30)))) {
        LOG_WARN("kimi: CUDA unavailable — CPU GEMM (%s)", llmoc::hal::cuda::status());
      }
    } else if (mode == llmoc::sched::ExecMode::kLayerStream &&
               cfg.layer_stream_device.rfind("cuda", 0) == 0) {
      const double vram = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 4.0;
      if (llmoc::hal::cuda::probe_available() &&
          llmoc::hal::cuda::enable(static_cast<size_t>(vram * (1ull << 30)))) {
        llmoc::hal::cuda::log_status();
      }
    }

    llmoc::families::kimi::KimiStubModel model;
    model.load_file(cfg.model_path, mode);
    if (llmoc::hal::cuda::enabled()) model.warm_gpu_weights();

    llmoc::model::HfTokenizer tok;
    try {
      tok.load(cfg.resolve_tokenizer_dir() + "/tokenizer.json");
    } catch (const std::exception& e) {
      LOG_WARN("kimi: tokenizer optional (%s)", e.what());
    }
    llmoc::model::Generator gen;
    gen.init(&model, &tok, 512);
    llmoc::sched::Scheduler sched;
    sched.start(&gen);
    llmoc::server::HttpApi api;
    api.bind(cfg, &sched);
    LOG_INFO("[kimi-stub] listening mode=%s (requested=%s)", llmoc::sched::mode_name(mode),
             llmoc::sched::mode_name(req));
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[kimi]: %s\n", e.what());
    return 1;
  }
}
