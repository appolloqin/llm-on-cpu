// llm-on-cpu :: src/server/main_glm.cpp
// 独立 GLM 服务入口 —— 不修改 main.cpp / main_int4.cpp

#include <cstdio>
#include <cstring>
#include <string>

#include "common/engine_config.h"
#include "common/log.h"
#include "glm/glm_config.h"
#include "glm/glm_flash_model.h"
#include "model/generate.h"
#include "model/tokenizer_hf.h"
#include "sched/scheduler.h"
#include "server/http_api.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

llmoc::EngineConfig to_http_cfg(const llmoc::glm::GlmEngineConfig& g) {
  llmoc::EngineConfig c;
  c.model_path = g.model_path;
  c.tokenizer_dir = g.tokenizer_dir;
  c.mode = llmoc::glm::GlmEngineConfig::mode_name(g.mode);
  c.dram_hot_gb = g.dram_hot_gb;
  c.kv_pool_gb = g.kv_pool_gb;
  c.io_workers = g.io_workers;
  c.server_port = g.server_port;
  c.api_key_env = g.api_key_env;
  c.max_new_tokens = g.max_new_tokens;
  c.model_dtype = llmoc::glm::GlmEngineConfig::quant_name(g.quant);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  std::string cfg_path = "configs/engine_glm_nvfp4.yaml";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: llmoc_server_glm --config configs/engine_glm_nvfp4.yaml\n");
      std::printf("  modes: pure_cpu | hybrid_gpu | pure_gpu\n");
      std::printf("  quant: nvfp4 (default) | awq_int4  (see configs/engine_glm_*.yaml)\n");
      return 0;
    }
  }
  try {
    llmoc::log::init(nullptr);
    auto gcfg = llmoc::glm::GlmEngineConfig::load(cfg_path);
    const std::string tok_dir = gcfg.resolve_tokenizer_dir();
#if defined(_OPENMP)
    LOG_INFO("OpenMP max_threads=%d", omp_get_max_threads());
#endif
    LOG_INFO("[glm] config=%s tokenizer=%s port=%d", cfg_path.c_str(), tok_dir.c_str(),
             gcfg.server_port);

    llmoc::glm::GlmFlashModel model;
    model.load(gcfg);
    if (!model.weights_ready()) {
      std::fprintf(stderr,
                   "FATAL: GLM weights not loaded (path=%s).\n"
                   "  %s\n"
                   "  Fix: run download_glm.cmd (or ./download_glm.sh), confirm "
                   "models/GLM-5.3-Flash.nvfp4.glmq (or .awq.glmq) exists,\n"
                   "  and model.path in %s matches. See docs/MODEL_GLM53_FLASH.md\n",
                   gcfg.model_path.c_str(),
                   model.load_error().empty() ? "(no detail)" : model.load_error().c_str(),
                   cfg_path.c_str());
      return 1;
    }

    llmoc::model::HfTokenizer tok;
    try {
      tok.load(tok_dir + "/tokenizer.json");
    } catch (const std::exception& e) {
      LOG_WARN("glm: tokenizer not loaded (%s) — place HF tokenizer at %s", e.what(),
               tok_dir.c_str());
    }

    llmoc::model::Generator gen;
    gen.init(&model, &tok, 4096);

    llmoc::sched::Scheduler sched;
    sched.start(&gen);

    llmoc::server::HttpApi api;
    api.bind(to_http_cfg(gcfg), &sched);
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 1;
  }
}
