#pragma once
// llm-on-cpu :: model/graph/model_spec.h
// P1: 配方驱动的模型结构（LayerSpec），避免每架构新增巨型 *_model.cpp。

#include <string>
#include <vector>

namespace llmoc::model::graph {

struct AttnSpec {
  int num_heads = 16;
  int num_kv_heads = 4;
  int head_dim = 256;
  bool q_gate = true;
};

struct LinearAttnSpec {
  int num_key_heads = 16;
  int num_value_heads = 32;
  int key_head_dim = 128;
  int value_head_dim = 128;
  int conv_kernel = 4;
};

struct MlpSpec {
  int intermediate_size = 9216;
  std::string act = "silu";
};

struct LayerSpec {
  std::string type;  // full_attention | linear_attention
  AttnSpec attn;
  LinearAttnSpec linear;
  MlpSpec mlp;
};

struct RopeSpec {
  float theta = 10000000.f;
  float partial_rotary_factor = 0.25f;
};

struct ModelSpec {
  std::string name = "qwen3_5";
  int hidden_size = 2560;
  int vocab_size = 248320;
  float rms_norm_eps = 1e-6f;
  bool rms_one_plus_weight = true;
  RopeSpec rope;
  bool tie_word_embeddings = true;
  std::string weight_prefix = "language_model.";
  std::vector<LayerSpec> layers;
};

// 从 recipe JSON 文件加载；失败抛 std::runtime_error。
ModelSpec load_model_spec(const std::string& json_path);

// 从 HF config.json（含 text_config）生成 ModelSpec（不写盘）。
ModelSpec model_spec_from_hf_config(const std::string& hf_config_json_path,
                                    const std::string& name = "qwen3_5");

}  // namespace llmoc::model::graph
