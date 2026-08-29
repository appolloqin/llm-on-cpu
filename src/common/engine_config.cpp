// llm-on-cpu :: common/engine_config.cpp
#include "common/engine_config.h"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace llmoc {
namespace {

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')))
    s = s.substr(1, s.size() - 2);
  // 去掉行内注释
  const auto hash = s.find('#');
  if (hash != std::string::npos) {
    s = s.substr(0, hash);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  }
  return s;
}

}  // namespace

EngineConfig EngineConfig::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  EngineConfig cfg;
  std::string section;
  std::string line;
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
    else if (full == "model.dtype") cfg.model_dtype = val;
    else if (full == "model.mtp") cfg.mtp = val;
    else if (full == "model.tokenizer_dir") cfg.tokenizer_dir = val;
    else if (full == "mode") cfg.mode = val;
    else if (full == "tiers.dram_hot_gb") cfg.dram_hot_gb = std::stod(val);
    else if (full == "tiers.kv_pool_gb") cfg.kv_pool_gb = std::stod(val);
    else if (full == "io.workers") cfg.io_workers = static_cast<unsigned>(std::stoul(val));
    else if (full == "decode.spec_k") cfg.spec_k = std::stoi(val);
    else if (full == "decode.max_new_tokens") cfg.max_new_tokens = std::stoi(val);
    else if (full == "server.port") cfg.server_port = std::stoi(val);
    else if (full == "server.api_key_env") cfg.api_key_env = val;
  }
  return cfg;
}

std::string EngineConfig::resolve_tokenizer_dir() const {
  if (!tokenizer_dir.empty()) return tokenizer_dir;
  // models/Foo.lwc -> models/Foo-hf
  std::string p = model_path;
  if (p.size() > 4 && p.substr(p.size() - 4) == ".lwc") {
    return p.substr(0, p.size() - 4) + "-hf";
  }
  return p + "-hf";
}

}  // namespace llmoc
