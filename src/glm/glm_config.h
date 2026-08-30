#pragma once
// llm-on-cpu :: glm/glm_config.h — GLM 专用配置（不修改 common/engine_config）

#include <string>

namespace llmoc::glm {

enum class QuantKind { kAwqInt4, kNvfp4, kBf16 };
enum class ExecMode { kPureCpu, kHybridGpu, kPureGpu };

struct GlmEngineConfig {
  std::string model_path = "models/GLM-5.3-Flash.nvfp4.glmq";
  std::string tokenizer_dir;
  std::string arch = "glm53_flash";
  QuantKind quant = QuantKind::kNvfp4;
  ExecMode mode = ExecMode::kHybridGpu;
  double dram_hot_gb = 40.0;
  double gpu_vram_gb = 20.0;
  double kv_pool_gb = 2.0;
  double prefetch_buf_gb = 4.0;
  unsigned io_workers = 4;
  int server_port = 15085;
  std::string api_key_env = "LLMOC_API_KEY";
  int max_new_tokens = 2048;

  static GlmEngineConfig load(const std::string& path);
  std::string resolve_tokenizer_dir() const;
  static const char* quant_name(QuantKind q);
  static const char* mode_name(ExecMode m);
};

}  // namespace llmoc::glm
