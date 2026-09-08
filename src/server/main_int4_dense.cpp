// llm-on-cpu :: src/server/main_int4_dense.cpp
// INT4/QLWC dense-only. MoE QLWC → llmoc_server_int4_moe.

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/engine_config.h"
#include "common/log.h"
#include "common/omp_tune.h"
#include "exec/factory.h"
#include "exec/nccl_probe.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/qwen3_5_int4_model.h"
#include "model/qwen3_8_int4_model.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/scheduler.h"
#include "server/http_api.h"
#include "server/int4_detect.h"
#include "weights/layer_stream.h"
#include "weights/qlwc_store.h"

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_int4.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server_int4 --config configs/engine_int4.yaml\n");
      std::printf("  INT4 dense only. MoE QLWC → llmoc_server_int4_moe\n");
      std::printf("  modes: pure_cpu | hybrid_gpu | pure_gpu | auto | layer_stream\n");
      return 0;
    }
  }
  try {
    using namespace llmoc::server::int4_detect;
    llmoc::log::init(nullptr);
    auto cfg = llmoc::EngineConfig::load(cfg_path);
    const std::string tok_dir = cfg.resolve_tokenizer_dir();

    const auto req = llmoc::sched::parse_mode(cfg.mode);
    bool degraded = false;
    std::string mode_err;
    llmoc::sched::ExecMode mode;

    const uint64_t weight_bytes = file_size_u64(cfg.model_path);
    const uint64_t dram_budget =
        static_cast<uint64_t>(cfg.dram_hot_gb * static_cast<double>(1ull << 30));

    if (req == llmoc::sched::ExecMode::kPureCpu) {
      mode = llmoc::sched::ExecMode::kPureCpu;
    } else if (req == llmoc::sched::ExecMode::kLayerStream) {
      mode = llmoc::sched::ExecMode::kLayerStream;
    } else if (req == llmoc::sched::ExecMode::kAuto) {
      const bool cuda_ok = llmoc::hal::cuda::probe_available();
      mode = llmoc::sched::resolve_mode(req, cuda_ok, &degraded, &mode_err);
      if (cfg.auto_layer_stream && mode == llmoc::sched::ExecMode::kPureCpu && weight_bytes > 0 &&
          dram_budget > 0 && weight_bytes > dram_budget) {
        mode = llmoc::sched::ExecMode::kLayerStream;
        LOG_WARN(
            "auto → layer_stream: model %.2f GiB > dram_hot_gb=%.2f (run-first; see "
            "DESIGN_LAYER_STREAM.md)",
            weight_bytes / (1024.0 * 1024.0 * 1024.0), cfg.dram_hot_gb);
      }
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
        size_t want = static_cast<size_t>(vram_gb * (1ull << 30));
        size_t free_b = 0, total_b = 0;
        if (llmoc::hal::cuda::device_mem_info(&free_b, &total_b) && total_b > 0) {
          const size_t margin = 512ull << 20;
          const size_t cap = total_b > margin ? total_b - margin : total_b;
          if (want > cap) {
            LOG_INFO("vram budget clamp: config=%.2fGiB → device_total-margin=%.2fGiB (free=%.2fGiB)",
                     vram_gb, cap / double(1ull << 30), free_b / double(1ull << 30));
            want = cap;
          }
        }
        if (!llmoc::hal::cuda::enable(want)) {
          throw std::runtime_error(std::string("CUDA enable failed: ") +
                                   llmoc::hal::cuda::status());
        }
        llmoc::hal::cuda::log_status();
      }
    }

    llmoc::tune_openmp_for_decode();
    LOG_INFO("[int4-dense] mode=%s model=%s tokenizer=%s port=%d max_seq=%d",
             llmoc::sched::mode_name(mode), cfg.model_path.c_str(), tok_dir.c_str(),
             cfg.server_port, cfg.max_seq);

    const bool use_stream = (mode == llmoc::sched::ExecMode::kLayerStream);
    std::unique_ptr<llmoc::wt::ILayerStreamLoader> streamer;
    llmoc::qlwc::QlwcStore store;
    llmoc::qlwc::QlwcStore* store_ptr = &store;

    if (use_stream) {
      if (cfg.layer_stream_device.rfind("cuda", 0) == 0) {
        const double vram_gb = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 4.0;
        if (llmoc::hal::cuda::probe_available() &&
            llmoc::hal::cuda::enable(static_cast<size_t>(vram_gb * (1ull << 30)))) {
          llmoc::hal::cuda::log_status();
        } else {
          LOG_WARN("layer_stream device=%s but CUDA unavailable — host window only",
                   cfg.layer_stream_device.c_str());
        }
      }
      streamer = llmoc::wt::make_layer_stream_loader();
      llmoc::wt::LayerStreamConfig lsc;
      lsc.window_layers = cfg.layer_stream_window > 0 ? cfg.layer_stream_window : 2;
      lsc.device = cfg.layer_stream_device;
      lsc.max_window_bytes = cfg.layer_stream_max_window_mb
                                 ? cfg.layer_stream_max_window_mb * (1ull << 20)
                                 : 0;
      lsc.io_workers = cfg.io_workers;
      streamer->open(cfg.model_path, lsc);
      store_ptr = streamer->qlwc();
      if (!store_ptr)
        throw std::runtime_error("layer_stream: expected QLWC store from loader");
      LOG_INFO("layer_stream: window=%d device=%s loaded_bootstrap", lsc.window_layers,
               lsc.device.c_str());
    } else {
      llmoc::qlwc::OpenOptions oopt;
      oopt.lazy = weight_bytes > (4ull << 30);
      LOG_INFO("qlwc open: %s (%.2f GiB, lazy=%d)", cfg.model_path.c_str(),
               weight_bytes / (1024.0 * 1024.0 * 1024.0), oopt.lazy ? 1 : 0);
      store.open(cfg.model_path, oopt);
      LOG_INFO("qlwc open done: tensors=%zu loaded=%.2f GiB lazy=%d",
               store.header().tensors.size(), store.loaded_bytes() / (1024.0 * 1024.0 * 1024.0),
               store.lazy() ? 1 : 0);
    }

    LOG_INFO("loading tokenizer + model config…");
    llmoc::model::HfTokenizer tok;
    tok.load(tok_dir + "/tokenizer.json");

    const std::string cfg_json = tok_dir + "/config.json";
    const bool want_moe = hf_config_looks_moe(cfg_json) || store_looks_moe(store_ptr);
    if (want_moe) {
      throw std::runtime_error(
          "MoE INT4/QLWC detected. Use llmoc_server_int4_moe instead of llmoc_server_int4.");
    }
    const bool want_38 = hf_config_looks_qwen38(cfg_json, cfg.model_path, /*is_moe=*/false);
    std::unique_ptr<llmoc::model::Qwen35Int4Model> model_holder;
    if (want_38) {
      model_holder = std::make_unique<llmoc::model::Qwen38Int4Model>();
      LOG_INFO("INT4 model class=Qwen38Int4Model (3.8 dense)");
    } else {
      model_holder = std::make_unique<llmoc::model::Qwen35Int4Model>();
      LOG_INFO("INT4 model class=Qwen35Int4Model (3.5 dense)");
    }
    auto& model = *model_holder;
    model.load(store_ptr, cfg_json);
    if (use_stream && streamer) {
      model.enable_layer_stream(streamer.get());
    }

    if (llmoc::hal::cuda::enabled() && !use_stream) {
      LOG_INFO("warm_gpu_int4: pinning attn/MLP/lm_head…");
      int warm_ok = 0, warm_fail = 0;
      model.warm_gpu_int4_weights(&warm_ok, &warm_fail);
      const size_t need = model.resident_workspace_bytes();
      const double need_g = need / double(1ull << 30);
      const double used_g = llmoc::hal::cuda::vram_used() / double(1ull << 30);
      const double bud_g = llmoc::hal::cuda::vram_budget() / double(1ull << 30);
      const bool warm_healthy = warm_ok > 0 && warm_ok >= warm_fail;
      if (warm_healthy && llmoc::hal::cuda::try_enable_resident_gpu(need)) {
        model.enable_resident_gpu(true);
        LOG_INFO("resident_gpu=ON need=%.2fGiB used=%.2fGiB budget=%.2fGiB warm_ok=%d fail=%d",
                 need_g, llmoc::hal::cuda::vram_used() / double(1ull << 30), bud_g, warm_ok,
                 warm_fail);
      } else {
        model.enable_resident_gpu(false);
        LOG_INFO("resident_gpu=OFF need=%.2fGiB used=%.2fGiB budget=%.2fGiB jit=%d warm_ok=%d "
                 "fail=%d (%s)",
                 need_g, used_g, bud_g, llmoc::hal::cuda::jit_available() ? 1 : 0, warm_ok,
                 warm_fail, llmoc::hal::cuda::status());
      }
    }

    {
      LOG_INFO("int4 warmup starting…");
      llmoc::model::SessionCache wc;
      model.init_cache(wc, 256);
      std::vector<float> logits;
      const auto warm_ids = tok.encode("hi");

      if (!warm_ids.empty()) {
        model.forward(warm_ids, wc, logits, true);
        LOG_INFO("int4 warmup: prefill done, decode x4…");
        for (int i = 0; i < 4; ++i) model.forward({warm_ids.back()}, wc, logits, false);
        model.release_session_device_state(wc);
      }
      LOG_INFO("int4 warmup: %d prefill + 4 decode forwards", static_cast<int>(warm_ids.size()));
      if (llmoc::hal::cuda::resident_gpu_enabled()) {
        llmoc::hal::cuda::log_resident_stats();
      }
      if (use_stream && streamer) {
        const auto st = streamer->stats();
        LOG_INFO("layer_stream stats: pin_loads=%llu pin_hits=%llu releases=%llu window_bytes=%llu",
                 static_cast<unsigned long long>(st.pin_loads),
                 static_cast<unsigned long long>(st.pin_hits),
                 static_cast<unsigned long long>(st.releases),
                 static_cast<unsigned long long>(st.window_bytes));
      }
    }

    llmoc::model::Generator gen;
    gen.init(&model, &tok, cfg.max_seq > 0 ? cfg.max_seq : 16384);

    llmoc::sched::Scheduler sched;
    sched.start(&gen);

    llmoc::server::HttpApi api;
    api.bind(cfg, &sched);
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[int4-dense]: %s\n", e.what());
    return 1;
  }
}
