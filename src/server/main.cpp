// llm-on-cpu :: src/server/main.cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
#include "model/moe_model.h"
#include "model/qwen3_5_model.h"
#include "model/tokenizer_hf.h"
#include "sched/mode_controller.h"
#include "sched/placement_planner.h"
#include "sched/scheduler.h"
#include "server/http_api.h"
#include "weights/prefetch_pipeline.h"
#include "weights/weight_manager.h"

namespace {
uint64_t file_size_u64(const std::string& p) {
  std::error_code ec;
  const auto sz = std::filesystem::file_size(p, ec);
  return ec ? 0 : static_cast<uint64_t>(sz);
}
}  // namespace

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server --config configs/engine.yaml\n");
      std::printf("  modes: pure_cpu | hybrid_gpu | pure_gpu | auto | layer_stream\n");
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
    llmoc::sched::ExecMode mode = llmoc::sched::ExecMode::kPureCpu;
    llmoc::contracts::DeviceMesh mesh;
    std::unique_ptr<llmoc::contracts::IExecBackend> exec_backend;

    const uint64_t weight_bytes = file_size_u64(cfg.model_path);
    const uint64_t dram_budget =
        static_cast<uint64_t>(cfg.dram_hot_gb * static_cast<double>(1ull << 30));

    if (req == llmoc::sched::ExecMode::kPureCpu || req == llmoc::sched::ExecMode::kLayerStream) {
      mode = req == llmoc::sched::ExecMode::kLayerStream ? llmoc::sched::ExecMode::kLayerStream
                                                         : llmoc::sched::ExecMode::kPureCpu;
      mesh.ids = {0};
      mesh.world_size = mesh.ep_size = mesh.tp_size = 1;
      llmoc::exec::MakeExecOptions eopt;
      eopt.mesh = mesh;
      exec_backend = llmoc::exec::make_exec(mode, eopt);
      LOG_INFO("exec caps: %s (CUDA/NCCL not probed)", llmoc::sched::mode_name(mode));
    } else if (req == llmoc::sched::ExecMode::kAuto) {
      const bool cuda_ok = llmoc::hal::cuda::probe_available();
      mode = llmoc::sched::resolve_mode(req, cuda_ok, &degraded, &mode_err);
      if (cfg.auto_layer_stream && mode == llmoc::sched::ExecMode::kPureCpu && weight_bytes > 0 &&
          dram_budget > 0 && weight_bytes > dram_budget) {
        mode = llmoc::sched::ExecMode::kLayerStream;
        LOG_WARN("auto → layer_stream: model %.2f GiB > dram_hot_gb=%.2f",
                 weight_bytes / (1024.0 * 1024.0 * 1024.0), cfg.dram_hot_gb);
      }
      mesh.ids = {0};
      mesh.world_size = mesh.ep_size = mesh.tp_size = 1;
      if (mode == llmoc::sched::ExecMode::kHybridGpu || mode == llmoc::sched::ExecMode::kPureGpu) {
        // fall through to GPU branch below by reusing code — simplify: treat as else
      }
      llmoc::exec::MakeExecOptions eopt;
      eopt.mesh = mesh;
      exec_backend = llmoc::exec::make_exec(
          mode == llmoc::sched::ExecMode::kLayerStream ? llmoc::sched::ExecMode::kLayerStream
                                                       : mode,
          eopt);
      if (mode == llmoc::sched::ExecMode::kPureCpu ||
          mode == llmoc::sched::ExecMode::kLayerStream) {
        LOG_INFO("exec caps: %s", llmoc::sched::mode_name(mode));
      }
    }

    if (req != llmoc::sched::ExecMode::kPureCpu && req != llmoc::sched::ExecMode::kLayerStream &&
        !(req == llmoc::sched::ExecMode::kAuto &&
          (mode == llmoc::sched::ExecMode::kPureCpu ||
           mode == llmoc::sched::ExecMode::kLayerStream))) {
      const bool cuda_ok = llmoc::hal::cuda::probe_available();
      if (req != llmoc::sched::ExecMode::kAuto) {
        mode = llmoc::sched::resolve_mode(req, cuda_ok, &degraded, &mode_err);
      }
      if (req == llmoc::sched::ExecMode::kPureGpu && !cuda_ok) {
        throw std::runtime_error(mode_err.empty() ? "pure_gpu requires CUDA" : mode_err);
      }
      if (degraded) {
        LOG_WARN("mode=hybrid_gpu requested but CUDA unavailable — degraded to pure_cpu (%s)",
                 llmoc::hal::cuda::status());
      }

      std::string mesh_err;
      if (!llmoc::sched::resolve_mesh_for_mode(mode, cfg.mesh_spec(),
                                               cuda_ok ? llmoc::hal::cuda::device_count() : 0,
                                               llmoc::exec::nccl_available(), cfg.has_moe_hint,
                                               &mesh, &mesh_err)) {
        throw std::runtime_error(mesh_err.empty() ? "device mesh resolve failed" : mesh_err);
      }

      std::string exec_err;
      llmoc::exec::MakeExecOptions eopt;
      eopt.mesh = mesh;
      eopt.vram_budget_per_rank =
          static_cast<size_t>((cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0) * (1ull << 30));
      exec_backend = llmoc::exec::make_exec(mode, eopt, &exec_err);
      if (!exec_backend) {
        throw std::runtime_error(exec_err.empty() ? "make_exec failed" : exec_err);
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
        pcfg.expert_slot_extra = cfg.expert_slot_extra;
        pcfg.strict_vram = cfg.strict_vram;
        pcfg.margin_bytes = static_cast<uint64_t>(cfg.margin_mb) << 20;
        llmoc::contracts::ActiveSetProfile ap;
        ap.has_moe = cfg.has_moe_hint;
        auto plan = llmoc::sched::PlacementPlanner::solve_active(mode, pcfg, mesh, ap);
        if (!plan.ok) throw std::runtime_error(plan.error);
        exec_backend->configure(plan);
        LOG_INFO("placement: %s", plan.summary.c_str());
        LOG_INFO("exec caps: experts_on_gpu=%d attn_on_gpu=%d %s",
                 exec_backend->caps().experts_on_gpu ? 1 : 0,
                 exec_backend->caps().attn_on_gpu ? 1 : 0, mesh.summary().c_str());
      } else {
        LOG_INFO("exec caps: pure_cpu %s", mesh.summary().c_str());
      }
    }

    llmoc::tune_openmp_for_decode();
    LOG_INFO("mode=%s model=%s tokenizer=%s port=%d dram_hot=%.1fG gpu_vram=%.1fG",
             llmoc::sched::mode_name(mode), cfg.model_path.c_str(), tok_dir.c_str(),
             cfg.server_port, cfg.dram_hot_gb, cfg.gpu_vram_gb);

    llmoc::wt::WeightManager wm;
    llmoc::wt::WeightManager::Config wcfg;
    wcfg.lru_budget_bytes = static_cast<uint64_t>(cfg.dram_hot_gb * (1ull << 30));
    wcfg.io_workers = cfg.io_workers;
    if (mode == llmoc::sched::ExecMode::kLayerStream) {
      wcfg.stream_dense_layers = true;
      // 窗口约 2 层：用 max_window_mb 或 dram 的 1/8
      if (cfg.layer_stream_max_window_mb)
        wcfg.lru_budget_bytes = cfg.layer_stream_max_window_mb * (1ull << 20);
      else
        wcfg.lru_budget_bytes = std::max<uint64_t>(wcfg.lru_budget_bytes / 8, 256ull << 20);
      LOG_INFO("BF16 layer_stream: stream_dense_layers=1 lru_budget=%.1f MiB",
               wcfg.lru_budget_bytes / (1024.0 * 1024.0));
    }
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
      if (llmoc::hal::cuda::enabled()) q->warm_gpu_bf16_weights();
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
