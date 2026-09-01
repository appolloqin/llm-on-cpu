#pragma once
// llm-on-cpu :: common/engine_config.h

#include <cstdint>
#include <string>
#include <vector>

#include "contracts/device_mesh.h"

namespace llmoc {

struct EngineConfig {
  std::string model_path = "models/Qwen3.5-4B.lwc";
  std::string model_dtype = "bf16";
  std::string mtp = "false";
  std::string tokenizer_dir;
  std::string mode = "pure_cpu";
  // layer_stream 子配置
  int layer_stream_window = 2;
  std::string layer_stream_device = "cpu";
  uint64_t layer_stream_max_window_mb = 0;  // 0 = 不限制
  bool auto_layer_stream = true;           // mode=auto 且权重超 DRAM 预算 → layer_stream

  double dram_hot_gb = 16.0;
  double gpu_vram_gb = 0.0;
  double kv_pool_gb = 2.0;
  unsigned io_workers = 2;
  int spec_k = 3;
  int server_port = 15085;
  std::string api_key_env = "LLMOC_API_KEY";
  int max_new_tokens = 256;

  // G8 device mesh
  std::vector<int> device_ids;  // empty => [0]
  bool devices_auto = false;    // ids: auto
  std::string device_strategy = "auto";
  int ep_size = 0;
  int tp_size = 0;
  std::string ep_shard = "mod";
  bool require_nccl = true;
  int expert_slot_extra = 1;
  bool strict_vram = true;
  int margin_mb = 512;
  bool has_moe_hint = true;

  contracts::DeviceMeshSpec mesh_spec() const;

  static EngineConfig load(const std::string& path);
  std::string resolve_tokenizer_dir() const;
};

}  // namespace llmoc
