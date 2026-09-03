// llm-on-cpu :: common/engine_config.cpp
#include "common/engine_config.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace llmoc {
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

std::vector<int> parse_int_list(const std::string& val) {
  std::vector<int> out;
  std::string cur;
  for (char c : val) {
    if (c == '[' || c == ']' || c == ' ') continue;
    if (c == ',') {
      if (!cur.empty()) {
        out.push_back(std::stoi(cur));
        cur.clear();
      }
      continue;
    }
    cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(std::stoi(cur));
  return out;
}

}  // namespace

contracts::DeviceMeshSpec EngineConfig::mesh_spec() const {
  contracts::DeviceMeshSpec s;
  if (devices_auto) {
    s.ids = {-1};  // sentinel → all visible
  } else if (!device_ids.empty()) {
    s.ids = device_ids;
  } else {
    s.ids = {0};
  }
  s.strategy = contracts::parse_mesh_strategy(device_strategy);
  s.ep_size = ep_size;
  s.tp_size = tp_size;
  s.ep_shard = contracts::parse_ep_shard(ep_shard);
  s.require_nccl = require_nccl;
  s.has_moe = has_moe_hint;
  return s;
}

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
    // 根级 key(无 section): mode, server.port 等。section 只在碰到 "x:" 头时被覆写,
    // 不会因为 "model:" 结束而清空, 所以根级 key 会被错误归到 model.* 下。直接按 key 识别。
    if (key == "mode") cfg.mode = val;
    else if (key == "server.port") cfg.server_port = std::stoi(val);
    if (full == "model.path") cfg.model_path = val;
    else if (full == "model.dtype") cfg.model_dtype = val;
    else if (full == "model.mtp") cfg.mtp = val;
    else if (full == "model.tokenizer_dir") cfg.tokenizer_dir = val;
    else if (full == "mode") cfg.mode = val;
    else if (full == "layer_stream.window_layers") cfg.layer_stream_window = std::stoi(val);
    else if (full == "layer_stream.device") cfg.layer_stream_device = val;
    else if (full == "layer_stream.max_window_mb")
      cfg.layer_stream_max_window_mb = static_cast<uint64_t>(std::stoll(val));
    else if (full == "layer_stream.auto")
      cfg.auto_layer_stream = (val == "true" || val == "1" || val == "yes");
    else if (full == "tiers.dram_hot_gb") cfg.dram_hot_gb = std::stod(val);
    else if (full == "tiers.gpu_vram_gb") cfg.gpu_vram_gb = std::stod(val);
    else if (full == "tiers.kv_pool_gb") cfg.kv_pool_gb = std::stod(val);
    else if (full == "tiers.pure_gpu.expert_slot_extra") cfg.expert_slot_extra = std::stoi(val);
    else if (full == "tiers.pure_gpu.strict_vram")
      cfg.strict_vram = (val == "true" || val == "1" || val == "yes");
    else if (full == "tiers.pure_gpu.margin_mb") cfg.margin_mb = std::stoi(val);
    else if (full == "devices.ids") {
      if (val == "auto") {
        cfg.devices_auto = true;
        cfg.device_ids.clear();
      } else {
        cfg.devices_auto = false;
        cfg.device_ids = parse_int_list(val);
      }
    } else if (full == "devices.strategy") cfg.device_strategy = val;
    else if (full == "devices.ep_size") cfg.ep_size = std::stoi(val);
    else if (full == "devices.tp_size") cfg.tp_size = std::stoi(val);
    else if (full == "devices.ep_shard") cfg.ep_shard = val;
    else if (full == "devices.nccl")
      cfg.require_nccl = (val == "true" || val == "1" || val == "yes");
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
  std::string p = model_path;
  if (p.size() > 4 && p.substr(p.size() - 4) == ".lwc") {
    return p.substr(0, p.size() - 4) + "-hf";
  }
  return p + "-hf";
}

}  // namespace llmoc
