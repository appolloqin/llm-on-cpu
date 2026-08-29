// llm-on-cpu :: model/tokenizer_hf.cpp
#include "model/tokenizer_hf.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace llmoc::model {
namespace {

// GPT-2 / HF ByteLevel: byte -> unicode codepoint mapping
std::string bytes_to_unicode_table() {
  // Build once as static map below
  return {};
}

const std::unordered_map<unsigned char, std::string>& byte_encoder() {
  static std::unordered_map<unsigned char, std::string> enc;
  static bool init = false;
  if (init) return enc;
  std::vector<int> bs;
  for (int i = 33; i <= 126; ++i) bs.push_back(i);
  for (int i = 161; i <= 172; ++i) bs.push_back(i);
  for (int i = 174; i <= 255; ++i) bs.push_back(i);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) != bs.end()) continue;
    bs.push_back(b);
    cs.push_back(256 + n);
    ++n;
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    std::string s;
    const int cp = cs[i];
    if (cp < 0x80) {
      s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    enc[static_cast<unsigned char>(bs[i])] = s;
  }
  init = true;
  return enc;
}

const std::unordered_map<std::string, unsigned char>& byte_decoder() {
  static std::unordered_map<std::string, unsigned char> dec;
  static bool init = false;
  if (init) return dec;
  for (const auto& [b, s] : byte_encoder()) dec[s] = b;
  init = true;
  return dec;
}

}  // namespace

void HfTokenizer::load(const std::string& tokenizer_json_path) {
  std::ifstream in(tokenizer_json_path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open tokenizer: " + tokenizer_json_path);
  nlohmann::json j;
  try {
    in >> j;
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("bad JSON in " + tokenizer_json_path + ": " + e.what() +
                             " (file must be UTF-8; re-download if corrupted)");
  }

  token_to_id_.clear();
  id_to_token_.clear();
  merges_.clear();
  merge_rank_.clear();
  special_ids_.clear();

  const auto& vocab = j.at("model").at("vocab");
  int max_id = 0;
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const int id = it.value().get<int>();
    token_to_id_[it.key()] = id;
    max_id = std::max(max_id, id);
  }
  id_to_token_.assign(static_cast<size_t>(max_id) + 1, "");
  for (const auto& [tok, id] : token_to_id_) {
    if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) id_to_token_[id] = tok;
  }

  if (j["model"].contains("merges")) {
    int rank = 0;
    for (const auto& m : j["model"]["merges"]) {
      std::string a, b;
      if (m.is_array()) {
        a = m[0].get<std::string>();
        b = m[1].get<std::string>();
      } else {
        const std::string s = m.get<std::string>();
        const auto sp = s.find(' ');
        if (sp == std::string::npos) continue;
        a = s.substr(0, sp);
        b = s.substr(sp + 1);
      }
      merges_.push_back({a, b});
      merge_rank_[a + "\x1f" + b] = rank++;
    }
  }

  if (j.contains("added_tokens")) {
    for (const auto& t : j["added_tokens"]) {
      const int id = t.at("id").get<int>();
      const std::string content = t.at("content").get<std::string>();
      token_to_id_[content] = id;
      if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1);
      id_to_token_[id] = content;
      if (t.value("special", false)) special_ids_.insert(id);
      if (content == "<|endoftext|>") {
        eos_id_ = id;
        bos_id_ = id;
        pad_id_ = id;
      } else if (content == "<|im_start|>")
        im_start_id_ = id;
      else if (content == "<|im_end|>")
        im_end_id_ = id;
    }
  }
  if (j.contains("unk_token") && token_to_id_.count(j["unk_token"]))
    unk_id_ = token_to_id_[j["unk_token"]];
}

std::vector<std::string> HfTokenizer::byte_encode(const std::string& text) const {
  const auto& enc = byte_encoder();
  std::vector<std::string> out;
  out.reserve(text.size());
  for (unsigned char c : text) out.push_back(enc.at(c));
  return out;
}

std::vector<std::string> HfTokenizer::bpe(const std::vector<std::string>& chars) const {
  if (chars.empty()) return {};
  std::vector<std::string> word = chars;
  while (word.size() > 1) {
    int best_rank = INT_MAX;
    int best_i = -1;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
      const std::string key = word[i] + "\x1f" + word[i + 1];
      auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = static_cast<int>(i);
      }
    }
    if (best_i < 0) break;
    std::vector<std::string> next;
    next.reserve(word.size());
    for (size_t i = 0; i < word.size();) {
      if (static_cast<int>(i) == best_i) {
        next.push_back(word[i] + word[i + 1]);
        i += 2;
      } else {
        next.push_back(word[i]);
        ++i;
      }
    }
    word.swap(next);
  }
  return word;
}

std::vector<int32_t> HfTokenizer::encode(const std::string& text, bool /*add_special*/) const {
  // 特殊 / 控制 token 整段匹配（含 Qwen3.5 的 <think>，其 special=false）
  std::vector<int32_t> ids;
  size_t i = 0;
  while (i < text.size()) {
    bool matched = false;
    for (int len = 64; len >= 1 && !matched; --len) {
      if (i + static_cast<size_t>(len) > text.size()) continue;
      const std::string piece = text.substr(i, len);
      auto it = token_to_id_.find(piece);
      if (it == token_to_id_.end()) continue;
      const bool atomic = special_ids_.count(it->second) != 0 ||
                          (piece.size() >= 3 && piece.front() == '<' && piece.back() == '>');
      if (!atomic) continue;
      ids.push_back(it->second);
      i += static_cast<size_t>(len);
      matched = true;
    }
    if (matched) continue;

    // 普通文本: 直到下一个 '<' 或结束
    size_t j = i;
    while (j < text.size() && text[j] != '<') ++j;
    if (j == i) {
      // 未识别的 '<'：按单字节推进，避免死循环
      j = i + 1;
    }
    const std::string chunk = text.substr(i, j - i);
    auto chars = byte_encode(chunk);
    auto pieces = bpe(chars);
    for (const auto& p : pieces) {
      auto it = token_to_id_.find(p);
      ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
    }
    i = j;
  }
  return ids;
}

std::string HfTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  std::string merged;
  for (int32_t id : ids) {
    if (skip_special && special_ids_.count(id)) continue;
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
    merged += id_to_token_[id];
  }
  // byte decode
  const auto& dec = byte_decoder();
  std::string out;
  size_t i = 0;
  while (i < merged.size()) {
    bool ok = false;
    for (int len = 3; len >= 1; --len) {
      if (i + static_cast<size_t>(len) > merged.size()) continue;
      const std::string piece = merged.substr(i, len);
      auto it = dec.find(piece);
      if (it != dec.end()) {
        out.push_back(static_cast<char>(it->second));
        i += static_cast<size_t>(len);
        ok = true;
        break;
      }
    }
    if (!ok) {
      out.push_back(merged[i]);
      ++i;
    }
  }
  return out;
}

std::string apply_qwen_chat_template(const std::vector<ChatMessage>& messages,
                                     bool add_generation_prompt, bool enable_thinking) {
  std::ostringstream oss;
  for (const auto& m : messages) {
    oss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
  }
  if (add_generation_prompt) {
    oss << "<|im_start|>assistant\n";
    // 与 models/.../chat_template.jinja 一致
    if (enable_thinking)
      oss << "<think>\n";
    else
      oss << "<think>\n\n</think>\n\n";
  }
  return oss.str();
}

std::string strip_qwen_think(const std::string& text) {
  std::string out = text;
  const std::string open = "<think>";
  const std::string close = "</think>";
  bool stripped_pair = false;
  for (;;) {
    const auto a = out.find(open);
    if (a == std::string::npos) break;
    stripped_pair = true;
    const auto b = out.find(close, a + open.size());
    if (b == std::string::npos) {
      out.erase(a);
      break;
    }
    out.erase(a, b + close.size() - a);
  }
  // 空 think 预填后模型仍可能只吐出「…</think>答案」：丢掉首个孤立闭合标签之前的内容
  if (!stripped_pair) {
    const auto c = out.find(close);
    if (c != std::string::npos) out.erase(0, c + close.size());
  }
  size_t i = 0;
  while (i < out.size() && (out[i] == '\n' || out[i] == '\r' || out[i] == ' ')) ++i;
  return out.substr(i);
}

}  // namespace llmoc::model
