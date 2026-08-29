// llm-on-cpu :: model/graph/model_spec.cpp
#include "model/graph/model_spec.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace llmoc::model::graph {
namespace {

LayerSpec layer_from_type(const std::string& type, const nlohmann::json& tc) {
  LayerSpec L;
  L.type = type;
  L.attn.num_heads = tc.value("num_attention_heads", 16);
  L.attn.num_kv_heads = tc.value("num_key_value_heads", 4);
  L.attn.head_dim = tc.value("head_dim", 256);
  L.attn.q_gate = true;
  L.linear.num_key_heads = tc.value("linear_num_key_heads", 16);
  L.linear.num_value_heads = tc.value("linear_num_value_heads", 32);
  L.linear.key_head_dim = tc.value("linear_key_head_dim", 128);
  L.linear.value_head_dim = tc.value("linear_value_head_dim", 128);
  L.linear.conv_kernel = tc.value("linear_conv_kernel_dim", 4);
  L.mlp.intermediate_size = tc.value("intermediate_size", 9216);
  L.mlp.act = "silu";
  return L;
}

ModelSpec from_text_config(const nlohmann::json& tc, const std::string& name) {
  ModelSpec s;
  s.name = name;
  s.hidden_size = tc.value("hidden_size", 2560);
  s.vocab_size = tc.value("vocab_size", 248320);
  s.rms_norm_eps = static_cast<float>(tc.value("rms_norm_eps", 1e-6));
  s.rms_one_plus_weight = true;
  s.tie_word_embeddings = tc.value("tie_word_embeddings", true);
  s.weight_prefix = "language_model.";
  if (tc.contains("rope_parameters")) {
    s.rope.theta = tc["rope_parameters"].value("rope_theta", 10000000.f);
    s.rope.partial_rotary_factor = tc["rope_parameters"].value("partial_rotary_factor", 0.25f);
  }
  const int n_layers = tc.value("num_hidden_layers", 32);
  s.layers.clear();
  if (tc.contains("layer_types")) {
    for (const auto& t : tc["layer_types"])
      s.layers.push_back(layer_from_type(t.get<std::string>(), tc));
  } else {
    for (int i = 0; i < n_layers; ++i) {
      const char* ty = (i + 1) % 4 == 0 ? "full_attention" : "linear_attention";
      s.layers.push_back(layer_from_type(ty, tc));
    }
  }
  return s;
}

}  // namespace

ModelSpec load_model_spec(const std::string& json_path) {
  std::ifstream in(json_path);
  if (!in) throw std::runtime_error("cannot open recipe: " + json_path);
  nlohmann::json root;
  in >> root;
  ModelSpec s;
  s.name = root.value("name", "qwen3_5");
  s.hidden_size = root.value("hidden_size", 2560);
  s.vocab_size = root.value("vocab_size", 248320);
  s.rms_norm_eps = static_cast<float>(root.value("rms_norm_eps", 1e-6));
  s.rms_one_plus_weight = root.value("rms_one_plus_weight", true);
  s.tie_word_embeddings = root.value("tie_word_embeddings", true);
  s.weight_prefix = root.value("weight_prefix", "language_model.");
  if (root.contains("rope")) {
    s.rope.theta = root["rope"].value("theta", 10000000.f);
    s.rope.partial_rotary_factor = root["rope"].value("partial_rotary_factor", 0.25f);
  }
  s.layers.clear();
  for (const auto& lj : root.at("layers")) {
    LayerSpec L;
    L.type = lj.at("type").get<std::string>();
    if (lj.contains("attn")) {
      L.attn.num_heads = lj["attn"].value("num_heads", 16);
      L.attn.num_kv_heads = lj["attn"].value("num_kv_heads", 4);
      L.attn.head_dim = lj["attn"].value("head_dim", 256);
      L.attn.q_gate = lj["attn"].value("q_gate", true);
    }
    if (lj.contains("linear")) {
      L.linear.num_key_heads = lj["linear"].value("num_key_heads", 16);
      L.linear.num_value_heads = lj["linear"].value("num_value_heads", 32);
      L.linear.key_head_dim = lj["linear"].value("key_head_dim", 128);
      L.linear.value_head_dim = lj["linear"].value("value_head_dim", 128);
      L.linear.conv_kernel = lj["linear"].value("conv_kernel", 4);
    }
    if (lj.contains("mlp")) {
      L.mlp.intermediate_size = lj["mlp"].value("intermediate_size", 9216);
      L.mlp.act = lj["mlp"].value("act", "silu");
    }
    s.layers.push_back(L);
  }
  return s;
}

ModelSpec model_spec_from_hf_config(const std::string& hf_config_json_path,
                                    const std::string& name) {
  std::ifstream in(hf_config_json_path);
  if (!in) throw std::runtime_error("cannot open " + hf_config_json_path);
  nlohmann::json root;
  in >> root;
  const auto& tc = root.contains("text_config") ? root["text_config"] : root;
  return from_text_config(tc, name);
}

}  // namespace llmoc::model::graph
