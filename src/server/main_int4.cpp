// llm-on-cpu :: src/server/main_int4.cpp
// INT4/QLWC 专用入口 —— 不修改原 llmoc_server / BF16 路径。

#include <cstdio>
#include <cstring>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "common/omp_tune.h"
#include "exec/factory.h"
#include "exec/nccl_probe.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/qwen3_5_int4_model.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/scheduler.h"
#include "server/http_api.h"
#include "weights/qlwc_store.h"

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

    const auto req = llmoc::sched::parse_mode(cfg.mode);
    bool degraded = false;
    std::string mode_err;
    llmoc::sched::ExecMode mode;

    if (req == llmoc::sched::ExecMode::kPureCpu) {
      // Never probe CUDA/NCCL on pure_cpu — INT4 hot path stays CPU-only.
      mode = llmoc::sched::ExecMode::kPureCpu;
    } else {
      const bool cuda_ok = llmoc::hal::cuda::probe_available();
      mode = llmoc::sched::resolve_mode(req, cuda_ok, &degraded, &mode_err);
      if (req == llmoc::sched::ExecMode::kPureGpu && !cuda_ok) {
        throw std::runtime_error(mode_err.empty() ? "pure_gpu requires CUDA" : mode_err);
      }
      if (degraded) {
        LOG_WARN("mode=hybrid_gpu requested but CUDA unavailable — degraded to pure_cpu");
      }
      if (mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) {
        llmoc::contracts::DeviceMesh mesh;
        std::string mesh_err;
        if (!llmoc::sched::resolve_mesh_for_mode(mode, cfg.mesh_spec(),
                                                 llmoc::hal::cuda::device_count(),
                                                 llmoc::exec::nccl_available(), false, &mesh,
                                                 &mesh_err)) {
          throw std::runtime_error(mesh_err.empty() ? "device mesh resolve failed" : mesh_err);
        }
        llmoc::exec::MakeExecOptions eopt;
        eopt.mesh = mesh;
        std::string exec_err;
        auto backend = llmoc::exec::make_exec(mode, eopt, &exec_err);
        if (!backend) throw std::runtime_error(exec_err.empty() ? "make_exec failed" : exec_err);
        LOG_INFO("exec %s experts_on_gpu=%d attn_on_gpu=%d", mesh.summary().c_str(),
                 backend->caps().experts_on_gpu ? 1 : 0, backend->caps().attn_on_gpu ? 1 : 0);

        double vram_gb = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
        if (!llmoc::hal::cuda::enable(static_cast<size_t>(vram_gb * (1ull << 30)))) {
          throw std::runtime_error(std::string("CUDA enable failed: ") +
                                   llmoc::hal::cuda::status());
        }
        llmoc::hal::cuda::log_status();
      }
    }

    llmoc::tune_openmp_for_decode();
    LOG_INFO("[int4] mode=%s model=%s tokenizer=%s port=%d", llmoc::sched::mode_name(mode),
             cfg.model_path.c_str(), tok_dir.c_str(), cfg.server_port);

    llmoc::qlwc::QlwcStore store;
    store.open(cfg.model_path);

    llmoc::model::HfTokenizer tok;
    tok.load(tok_dir + "/tokenizer.json");

    llmoc::model::Qwen35Int4Model model;
    model.load(&store, tok_dir + "/config.json");

    // 预热 OpenMP 线程组 + GEMM 微内核，避免首包 decode 虚慢（与 BENCH warm 同理）
    {
      llmoc::model::SessionCache wc;
      model.init_cache(wc, 256);
      std::vector<float> logits;
      const auto warm_ids = tok.encode("hi");
      if (!warm_ids.empty()) {
        model.forward(warm_ids, wc, logits, true);
        for (int i = 0; i < 4; ++i) model.forward({warm_ids.back()}, wc, logits, false);
      }
      LOG_INFO("int4 warmup: %d prefill + 4 decode forwards", static_cast<int>(warm_ids.size()));
    }

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
