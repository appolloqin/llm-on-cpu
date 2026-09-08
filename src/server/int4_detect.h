// llm-on-cpu :: server/int4_detect.h
// Shared HF/QLWC family detection for INT4 dense vs MoE entry points.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "weights/qlwc_store.h"

namespace llmoc::server::int4_detect {

inline uint64_t file_size_u64(const std::string& p) {
  std::error_code ec;
  const auto sz = std::filesystem::file_size(p, ec);
  return ec ? 0 : static_cast<uint64_t>(sz);
}

inline bool hf_config_looks_moe(const std::string& config_json_path) {
  std::ifstream in(config_json_path);
  if (!in) return false;
  nlohmann::json root;
  try {
    in >> root;
  } catch (...) {
    return false;
  }
  const auto& tc = root.contains("text_config") ? root["text_config"] : root;
  const int n = tc.value("num_experts", tc.value("n_routed_experts", 0));
  return n > 0;
}

inline bool store_looks_moe(const qlwc::QlwcStore* store) {
  if (!store) return false;
  return store->has("language_model.layers.0.mlp.experts.0.gate_proj.weight") ||
         store->has("language_model.layers.0.mlp.gate.weight") ||
         store->has("layers.0.mlp.experts.0.gate_proj.weight");
}

inline std::string lower_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

inline bool path_looks_qwen36(const std::string& s) {
  const std::string l = lower_ascii(s);
  return l.find("qwen3.6") != std::string::npos || l.find("qwen36") != std::string::npos ||
         l.find("3.6-35") != std::string::npos || l.find("a3b") != std::string::npos;
}

inline bool path_looks_qwen38(const std::string& s) {
  const std::string l = lower_ascii(s);
  return l.find("qwen3.8") != std::string::npos || l.find("qwen38") != std::string::npos ||
         l.find("3.8-27") != std::string::npos;
}

// Family 3.8 (dense or MoE). Path wins; dense geometry fallback must not steal 3.6 MoE.
inline bool hf_config_looks_qwen38(const std::string& config_json_path,
                                   const std::string& model_path_hint, bool is_moe) {
  if (path_looks_qwen38(config_json_path) || path_looks_qwen38(model_path_hint)) return true;
  if (path_looks_qwen36(config_json_path) || path_looks_qwen36(model_path_hint)) return false;
  if (is_moe) return false;  // unnamed MoE → 3.6 MoE entry

  std::ifstream in(config_json_path);
  if (!in) return false;
  nlohmann::json root;
  try {
    in >> root;
  } catch (...) {
    return false;
  }
  const auto& tc = root.contains("text_config") ? root["text_config"] : root;
  const int H = tc.value("hidden_size", 0);
  const int L = tc.value("num_hidden_layers", 0);
  // 3.8-27B dense: hidden=5120 layers=64; keep clear of 3.5-4B (2560/32).
  return H >= 4096 || L >= 48;
}

}  // namespace llmoc::server::int4_detect
