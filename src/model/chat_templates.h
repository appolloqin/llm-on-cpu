// llm-on-cpu :: model/chat_templates.h
// Per-family chat templates — Qwen3.5 / 3.6 / 3.8 / GLM / DeepSeek / Kimi.
// thinking_off_style can override family defaults via engine yaml.
#pragma once

#include <string>
#include <vector>

#include "model/tokenizer_hf.h"

namespace llmoc::model {

enum class ChatFamily {
  kQwen35,
  kQwen36,
  kQwen38,
  kGlm,
  kDeepSeek,
  kKimi,
  kUnknown,
};

// When enable_thinking=false:
//   family        — per-model-family default (3.5=empty_prefill, 3.6/3.8=no_tags, others=no_tags)
//   empty_prefill — always inject empty <think></think>
//   no_tags       — never inject think tags
enum class ThinkingOffStyle { kFamily, kEmptyPrefill, kNoTags };

const char* chat_family_name(ChatFamily f);
const char* thinking_off_style_name(ThinkingOffStyle s);
ThinkingOffStyle parse_thinking_off_style(const std::string& s);

ChatFamily chat_family_from_kind(const std::string& kind);

struct ChatTemplateOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = false;
  ThinkingOffStyle thinking_off_style = ThinkingOffStyle::kFamily;
};

std::string apply_chat_template(ChatFamily family, const std::vector<ChatMessage>& messages,
                                const ChatTemplateOptions& opt);

std::string postprocess_completion(ChatFamily family, const std::string& text,
                                   bool enable_thinking);

void append_assistant_generation_prefix(ChatFamily family, const ChatTemplateOptions& opt,
                                        std::string& out_text);

}  // namespace llmoc::model
