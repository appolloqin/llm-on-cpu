#pragma once
// llm-on-cpu :: model/tokenizer_hf.h
// HuggingFace tokenizer.json (BPE + ByteLevel) 最小实现。

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llmoc::model {

class HfTokenizer {
 public:
  void load(const std::string& tokenizer_json_path);

  std::vector<int32_t> encode(const std::string& text, bool add_special = false) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;

  int32_t bos_id() const { return bos_id_; }
  int32_t eos_id() const { return eos_id_; }
  int32_t pad_id() const { return pad_id_; }
  int32_t im_start_id() const { return im_start_id_; }
  int32_t im_end_id() const { return im_end_id_; }
  int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

 private:
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::vector<std::pair<std::string, std::string>> merges_;
  std::unordered_map<std::string, int> merge_rank_;
  std::unordered_set<int32_t> special_ids_;

  int32_t bos_id_ = -1, eos_id_ = -1, pad_id_ = -1;
  int32_t im_start_id_ = -1, im_end_id_ = -1;
  int32_t unk_id_ = 0;

  std::vector<std::string> byte_encode(const std::string& text) const;
  std::vector<std::string> bpe(const std::vector<std::string>& chars) const;
};

struct ChatImage {
  std::vector<uint8_t> bytes;  // PNG/JPEG 等原始字节
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::vector<ChatImage> images;  // 多模态：插在文本前（Picture N + vision tokens）
};

// Qwen chat 模板裁剪实现(文本 messages)。
// enable_thinking=false 时按 HF 惯例预写空 <think></think>，跳过思考链。
std::string apply_qwen_chat_template(const std::vector<ChatMessage>& messages,
                                     bool add_generation_prompt,
                                     bool enable_thinking = false);

// 去掉 <think>...</think> 块，便于展示 / 多轮历史。
std::string strip_qwen_think(const std::string& text);

}  // namespace llmoc::model
