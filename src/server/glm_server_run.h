// llm-on-cpu :: server/glm_server_run.h — shared GLM MoE entry (quant forced by binary)
#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
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

namespace llmoc::server::glm_run {

inline EngineConfig to_http_cfg(const glm::GlmEngineConfig& g) {
  EngineConfig c;
  c.model_path = g.model_path;
  c.tokenizer_dir = g.tokenizer_dir;
  c.mode = glm::GlmEngineConfig::mode_name(g.mode);
  c.dram_hot_gb = g.dram_hot_gb;
  c.kv_pool_gb = g.kv_pool_gb;
  c.io_workers = g.io_workers;
  c.server_port = g.server_port;
  c.api_key_env = g.api_key_env;
  c.max_new_tokens = g.max_new_tokens;
  c.model_dtype = glm::GlmEngineConfig::quant_name(g.quant);
  return c;
}

inline int run(int argc, char** argv, glm::QuantKind force_quant, const char* exe_name,
               const char* default_cfg) {
  std::string cfg_path = default_cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: %s --config %s\n", exe_name, default_cfg);
      std::printf("  GLM MoE only · quant locked to %s\n",
                  glm::GlmEngineConfig::quant_name(force_quant));
      std::printf("  siblings: llmoc_server_glm_bf16 | _int4 | _nvfp4\n");
      std::printf("  modes: pure_cpu | hybrid_gpu | pure_gpu\n");
      return 0;
    }
  }
  try {
    log::init(nullptr);
    auto gcfg = glm::GlmEngineConfig::load(cfg_path);
    gcfg.quant = force_quant;
    const std::string tok_dir = gcfg.resolve_tokenizer_dir();
#if defined(_OPENMP)
    LOG_INFO("OpenMP max_threads=%d", omp_get_max_threads());
#endif
    LOG_INFO("[glm-%s] config=%s tokenizer=%s port=%d",
             glm::GlmEngineConfig::quant_name(force_quant), cfg_path.c_str(), tok_dir.c_str(),
             gcfg.server_port);

    glm::GlmFlashModel model;
    model.load(gcfg);
    if (!model.weights_ready()) {
      std::fprintf(stderr,
                   "FATAL: GLM weights not loaded (path=%s).\n"
                   "  %s\n"
                   "  Fix: matching .glmq for %s; see docs/MODEL_GLM53_FLASH.md\n",
                   gcfg.model_path.c_str(),
                   model.load_error().empty() ? "(no detail)" : model.load_error().c_str(),
                   glm::GlmEngineConfig::quant_name(force_quant));
      return 1;
    }
    if (model.quant() != force_quant) {
      throw std::runtime_error(std::string("glm quant mismatch after load: got ") +
                               glm::GlmEngineConfig::quant_name(model.quant()));
    }

    model::HfTokenizer tok;
    try {
      tok.load(tok_dir + "/tokenizer.json");
    } catch (const std::exception& e) {
      LOG_WARN("glm: tokenizer not loaded (%s) — place HF tokenizer at %s", e.what(),
               tok_dir.c_str());
    }

    model::Generator gen;
    gen.init(&model, &tok, gcfg.max_seq > 0 ? gcfg.max_seq : 16384);
    sched::Scheduler sched;
    sched.start(&gen);
    server::HttpApi api;
    api.bind(to_http_cfg(gcfg), &sched);
    api.listen();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL[glm]: %s\n", e.what());
    return 1;
  }
}

}  // namespace llmoc::server::glm_run
