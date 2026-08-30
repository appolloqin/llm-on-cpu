// llm-on-cpu :: glm/glm_config.cpp
#include "glm/glm_config.h"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace llmoc::glm {
namespace {

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')))
    s = s.substr(1, s.size() - 2);
  const auto hash = s.find('#');
  if (hash != std::string::npos) {
    s = s.substr(0, hash);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  }
  return s;
}

}  // namespace

const char* GlmEngineConfig::quant_name(QuantKind q) {
  switch (q) {
    case QuantKind::kAwqInt4: return "awq_int4";
    case QuantKind::kNvfp4: return "nvfp4";
    default: return "bf16";
  }
}

const char* GlmEngineConfig::mode_name(ExecMode m) {
  switch (m) {
    case ExecMode::kHybridGpu: return "hybrid_gpu";
    case ExecMode::kPureGpu: return "pure_gpu";
    default: return "pure_cpu";
  }
}

GlmEngineConfig GlmEngineConfig::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("glm: cannot open config: " + path);
  GlmEngineConfig cfg;
  std::string section, line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line.back() == ':' && line.find(' ') == std::string::npos) {
      section = line.substr(0, line.size() - 1);
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon));
    const std::string val = trim(line.substr(colon + 1));
    const std::string full = section.empty() ? key : section + "." + key;
    if (full == "model.path") cfg.model_path = val;
    else if (full == "model.dtype" || full == "quant") {
      if (val == "nvfp4") cfg.quant = QuantKind::kNvfp4;
      else if (val == "bf16") cfg.quant = QuantKind::kBf16;
      else cfg.quant = QuantKind::kAwqInt4;
    } else if (full == "model.arch") cfg.arch = val;
    else if (full == "model.tokenizer_dir") cfg.tokenizer_dir = val;
    else if (full == "mode") {
      if (val == "hybrid_gpu") cfg.mode = ExecMode::kHybridGpu;
      else if (val == "pure_gpu") cfg.mode = ExecMode::kPureGpu;
      else cfg.mode = ExecMode::kPureCpu;
    } else if (full == "tiers.dram_hot_gb") cfg.dram_hot_gb = std::stod(val);
    else if (full == "tiers.gpu_vram_gb") cfg.gpu_vram_gb = std::stod(val);
    else if (full == "tiers.kv_pool_gb") cfg.kv_pool_gb = std::stod(val);
    else if (full == "tiers.prefetch_buf_gb") cfg.prefetch_buf_gb = std::stod(val);
    else if (full == "io.workers") cfg.io_workers = static_cast<unsigned>(std::stoul(val));
    else if (full == "decode.max_new_tokens") cfg.max_new_tokens = std::stoi(val);
    else if (full == "server.port") cfg.server_port = std::stoi(val);
    else if (full == "server.api_key_env") cfg.api_key_env = val;
  }
  return cfg;
}

std::string GlmEngineConfig::resolve_tokenizer_dir() const {
  if (!tokenizer_dir.empty()) return tokenizer_dir;
  std::string p = model_path;
  const auto slash = p.find_last_of("/\\");
  const std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
  const auto dot = base.find('.');
  const std::string stem = dot == std::string::npos ? base : base.substr(0, dot);
  if (slash == std::string::npos) return stem + "-hf";
  return p.substr(0, slash + 1) + stem + "-hf";
}

}  // namespace llmoc::glm
