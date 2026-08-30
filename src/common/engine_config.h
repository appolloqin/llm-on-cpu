#pragma once
// llm-on-cpu :: common/engine_config.h
// 极简 YAML 子集解析器 —— 只覆盖 configs/engine.yaml 已知键。

#include <cstdint>
#include <string>

namespace llmoc {

struct EngineConfig {
  std::string model_path = "models/Qwen3.5-4B.lwc";
  std::string model_dtype = "bf16";
  std::string mtp = "false";  // auto|true|false
  std::string tokenizer_dir;  // 默认由 model_path 推 *-hf/
  std::string mode = "pure_cpu";
  double dram_hot_gb = 16.0;
  double gpu_vram_gb = 0.0;  // M5: hybrid/pure_gpu VRAM budget; 0 = unset
  double kv_pool_gb = 2.0;
  unsigned io_workers = 2;
  int spec_k = 3;
  int server_port = 15085;
  std::string api_key_env = "LLMOC_API_KEY";
  int max_new_tokens = 256;

  static EngineConfig load(const std::string& path);
  std::string resolve_tokenizer_dir() const;
};

}  // namespace llmoc
