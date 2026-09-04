// llm-on-cpu :: tools/bench_decode_tps/main.cpp
// P0: 暖机后测 completion tok/s（含短 prefill）；与 scheduler last_tps 同口径

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "common/omp_tune.h"
#include "hal/cuda_backend.h"
#include "model/generate.h"
#include "model/qwen3_5_int4_model.h"
#include "model/qwen3_5_model.h"
#include "model/tokenizer_hf.h"
#include "weights/qlwc_store.h"
#include "weights/weight_manager.h"

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_int4.yaml";
  int neu = 256;
  int warm = 1;
  int prompt_tokens_target = 0;  // >0: pad user text until chat prompt ≈ this many tokens
  std::string user_msg = "Write a detailed explanation of binary search with examples.";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--new") && i + 1 < argc) neu = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warm") && i + 1 < argc) warm = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) user_msg = argv[++i];
    else if (!std::strcmp(argv[i], "--prompt-tokens") && i + 1 < argc)
      prompt_tokens_target = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--short")) user_msg = "hi";
  }

  auto cfg = llmoc::EngineConfig::load(cfg_path);
  llmoc::tune_openmp_for_decode();
  const std::string tok_dir = cfg.resolve_tokenizer_dir();
  llmoc::model::HfTokenizer tok;
  tok.load(tok_dir + "/tokenizer.json");

  if (prompt_tokens_target > 0) {
    // Grow a repetitive body until apply_chat_template-ish length reaches target.
    // Matches server prompt shape used by Generator (im_start/user/.../assistant).
    auto approx_prompt_n = [&](const std::string& body) {
      const std::string full = "<|im_start|>user\n" + body +
                               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
      return static_cast<int>(tok.encode(full).size());
    };
    std::string body = user_msg + "\n";
    const std::string pad = "The quick brown fox jumps over the lazy dog. ";
    int guard = 0;
    while (approx_prompt_n(body) < prompt_tokens_target && guard++ < 200000) body += pad;
    user_msg = std::move(body);
    std::printf("[bench_decode_tps] padded prompt_tokens≈%d (target=%d)\n",
                approx_prompt_n(user_msg), prompt_tokens_target);
  }

  std::unique_ptr<llmoc::model::ICausalLM> model;
  llmoc::wt::WeightManager wm;
  llmoc::qlwc::QlwcStore qstore;
  const bool int4 = cfg.model_path.size() > 5 &&
                    cfg.model_path.substr(cfg.model_path.size() - 5) == ".qlwc";

  if (int4) {
    qstore.open(cfg.model_path);
    auto m = std::make_unique<llmoc::model::Qwen35Int4Model>();
    m->load(&qstore, tok_dir + "/config.json");
    // hybrid / pure_gpu: 启用 CUDA + 上传层权重 (resident) 让 layer GEMV 走 JIT
    if (cfg.mode == "hybrid_gpu" || cfg.mode == "pure_gpu") {
      const double vram_gb = cfg.gpu_vram_gb > 0 ? cfg.gpu_vram_gb : 8.0;
      if (llmoc::hal::cuda::probe_available() &&
          llmoc::hal::cuda::enable(static_cast<size_t>(vram_gb * (1ull << 30)))) {
        m->warm_gpu_int4_weights();
      }
    }
    model = std::move(m);
  } else {
    llmoc::wt::WeightManager::Config wcfg;
    wcfg.lru_budget_bytes = static_cast<uint64_t>(cfg.dram_hot_gb * (1ull << 30));
    wcfg.io_workers = cfg.io_workers;
    wm.open(cfg.model_path, wcfg);
    auto m = std::make_unique<llmoc::model::Qwen35Model>();
    m->load(&wm, tok_dir + "/config.json");
    model = std::move(m);
  }

  llmoc::model::Generator gen;
  gen.init(model.get(), &tok, cfg.max_seq > 0 ? cfg.max_seq : 16384);

  llmoc::model::GenerateRequest req;
  req.messages = {{"user", user_msg}};
  req.max_new_tokens = neu;
  req.temperature = 0.f;
  req.enable_thinking = false;
  req.mtp = cfg.mtp;
  req.spec_k = cfg.spec_k;

  for (int i = 0; i < warm; ++i) {
    auto w = gen.generate(req);
    LOG_INFO("warm[%d] completion=%d mtp_steps=%d accepted=%d", i, w.completion_tokens,
             w.mtp_verify_steps, w.mtp_draft_accepted);
  }

  const auto t0 = std::chrono::steady_clock::now();
  auto result = gen.generate(req);
  const auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();
  // 与 sched::last_tps 一致：completion_tokens / generate 墙钟
  const double e2e_tps =
      sec > 0 ? static_cast<double>(result.completion_tokens) / sec : 0.0;
  std::printf("[bench_decode_tps] model=%s has_mtp=%d\n", cfg.model_path.c_str(),
              model->has_mtp() ? 1 : 0);
  std::printf("[bench_decode_tps] prompt=%d completion=%d wall=%.3fs e2e_tps=%.2f "
              "(=last_tps口径)\n",
              result.prompt_tokens, result.completion_tokens, sec, e2e_tps);
  std::printf("[bench_decode_tps] mtp_verify_steps=%d mtp_draft_accepted=%d\n",
              result.mtp_verify_steps, result.mtp_draft_accepted);
  std::printf("[bench_decode_tps] text=%.60s\n", result.text.c_str());

  // 纯 decode：prefill 后只计 forward；argmax 在计时外
  {
    llmoc::model::SessionCache cache;
    model->init_cache(cache, 4096);
    auto ids = tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    std::vector<float> logits;
    model->forward(ids, cache, logits, true);
    const int N = 24;
    double fwd_ms = 0.0, argmax_ms = 0.0;
    for (int i = 0; i < N; ++i) {
      const auto a0 = std::chrono::steady_clock::now();
      int32_t next = 0;
      float best = -1e30f;
      for (int v = 0; v < static_cast<int>(logits.size()); ++v) {
        if (logits[v] > best) {
          best = logits[v];
          next = v;
        }
      }
      const auto a1 = std::chrono::steady_clock::now();
      argmax_ms += std::chrono::duration<double, std::milli>(a1 - a0).count();
      const auto f0 = std::chrono::steady_clock::now();
      model->forward({next}, cache, logits, false);
      const auto f1 = std::chrono::steady_clock::now();
      fwd_ms += std::chrono::duration<double, std::milli>(f1 - f0).count();
    }
    std::printf(
        "[bench_decode_tps] pure_decode n=%d forward_ms=%.1f argmax_ms=%.1f decode_tps=%.2f "
        "(ms/tok=%.1f)\n",
        N, fwd_ms, argmax_ms, 1000.0 * N / fwd_ms, fwd_ms / N);
  }
  return 0;
}
