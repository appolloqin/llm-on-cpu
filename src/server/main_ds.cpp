// llm-on-cpu :: src/server/main_ds.cpp — DeepSeek-V4 stub server
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

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_ds_nvfp4.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server_ds --config configs/engine_ds_nvfp4.yaml\n");
      std::printf("  DS-STUB-v0 (latent MoE + NVFP4 experts)\n");
      return 0;
    }
  }
  try {
    llmoc::log::init(nullptr);
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const auto mode = llmoc::sched::parse_mode(cfg.mode);
    if (mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) {
      const double vram = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      if (!llmoc::hal::cuda::enable(static_cast<size_t>(vram * (1ull << 30)))) {
        if (mode == llmoc::sched::ExecMode::kPureGpu)
          throw std::runtime_error(std::string("CUDA required: ") + llmoc::hal::cuda::status());
        LOG_WARN("ds: CUDA unavailable — CPU GEMM (%s)", llmoc::hal::cuda::status());
      }
    }
    llmoc::families::deepseek::DsStubModel model;
    model.load_file(cfg.model_path, mode);
    if (llmoc::hal::cuda::enabled()) model.warm_gpu_weights();

    llmoc::model::HfTokenizer tok;
    try {
      tok.load(cfg.resolve_tokenizer_dir() + "/tokenizer.json");
    } catch (const std::exception& e) {
      LOG_WARN("ds: tokenizer optional (%s)", e.what());
    }
    llmoc::model::Generator gen;
    gen.init(&model, &tok, 512);
    llmoc::sched::Scheduler sched;
    sched.start(&gen);
    llmoc::server::HttpApi api;
    api.bind(cfg, &sched);
    LOG_INFO("[ds-stub] listening mode=%s", llmoc::sched::mode_name(mode));
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[ds]: %s\n", e.what());
    return 1;
  }
}
