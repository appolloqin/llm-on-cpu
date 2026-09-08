// llm-on-cpu :: model/chat_templates.cpp
#include "model/chat_templates.h"

#include <sstream>

namespace llmoc::model {
namespace {

std::string lower_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

std::string im_body(const std::vector<ChatMessage>& messages) {
  std::ostringstream oss;
  for (const auto& m : messages) {
    oss << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
  }
  return oss.str();
}

ThinkingOffStyle resolve_off_style(ChatFamily family, ThinkingOffStyle cfg) {
  if (cfg != ThinkingOffStyle::kFamily) return cfg;
  switch (family) {
    case ChatFamily::kQwen35:
      return ThinkingOffStyle::kEmptyPrefill;
    case ChatFamily::kQwen36:
    case ChatFamily::kQwen38:
    case ChatFamily::kGlm:
    case ChatFamily::kDeepSeek:
    case ChatFamily::kKimi:
    default:
      return ThinkingOffStyle::kNoTags;
  }
}

bool family_supports_qwen_think(ChatFamily family) {
  return family == ChatFamily::kQwen35 || family == ChatFamily::kQwen36 ||
         family == ChatFamily::kQwen38 || family == ChatFamily::kUnknown;
}

void append_think_suffix(ChatFamily family, const ChatTemplateOptions& opt, std::ostringstream& oss) {
  if (!family_supports_qwen_think(family)) return;
  if (opt.enable_thinking) {
    oss << "<think>\n";
    return;
  }
  const ThinkingOffStyle st = resolve_off_style(family, opt.thinking_off_style);
  if (st == ThinkingOffStyle::kEmptyPrefill) oss << "<think>\n\n</think>\n\n";
}

std::string tmpl_im(const std::vector<ChatMessage>& messages, ChatFamily family,
                    const ChatTemplateOptions& opt) {
  std::ostringstream oss;
  oss << im_body(messages);
  if (!opt.add_generation_prompt) return oss.str();
  oss << "<|im_start|>assistant\n";
  append_think_suffix(family, opt, oss);
  return oss.str();
}

}  // namespace

const char* chat_family_name(ChatFamily f) {
  switch (f) {
    case ChatFamily::kQwen35: return "qwen35";
    case ChatFamily::kQwen36: return "qwen36";
    case ChatFamily::kQwen38: return "qwen38";
    case ChatFamily::kGlm: return "glm";
    case ChatFamily::kDeepSeek: return "deepseek";
    case ChatFamily::kKimi: return "kimi";
    default: return "unknown";
  }
}

const char* thinking_off_style_name(ThinkingOffStyle s) {
  switch (s) {
    case ThinkingOffStyle::kEmptyPrefill: return "empty_prefill";
    case ThinkingOffStyle::kNoTags: return "no_tags";
    default: return "family";
  }
}

ThinkingOffStyle parse_thinking_off_style(const std::string& s) {
  const std::string v = lower_ascii(s);
  if (v == "empty_prefill" || v == "empty" || v == "prefill")
    return ThinkingOffStyle::kEmptyPrefill;
  if (v == "no_tags" || v == "none" || v == "off" || v == "disabled")
    return ThinkingOffStyle::kNoTags;
  return ThinkingOffStyle::kFamily;
}

ChatFamily chat_family_from_kind(const std::string& kind) {
  const std::string k = lower_ascii(kind);
  if (k.find("kimi") != std::string::npos) return ChatFamily::kKimi;
  if (k.find("deepseek") != std::string::npos || k.find("ds_stub") != std::string::npos ||
      k.rfind("deepseek_v4_stub", 0) == 0)
    return ChatFamily::kDeepSeek;
  if (k.find("glm") != std::string::npos) return ChatFamily::kGlm;
  if (k.find("qwen3_8") != std::string::npos || k.find("qwen38") != std::string::npos ||
      k.find("3.8") != std::string::npos)
    return ChatFamily::kQwen38;
  if (k.find("qwen3_6") != std::string::npos || k.find("qwen36") != std::string::npos ||
      k.find("3.6") != std::string::npos)
    return ChatFamily::kQwen36;
  if (k.find("qwen3_5") != std::string::npos || k.find("qwen35") != std::string::npos ||
      k == "qwen3_5" || k == "moe")
    return ChatFamily::kQwen35;
  return ChatFamily::kUnknown;
}

std::string apply_chat_template(ChatFamily family, const std::vector<ChatMessage>& messages,
                                const ChatTemplateOptions& opt) {
  return tmpl_im(messages, family, opt);
}

std::string postprocess_completion(ChatFamily family, const std::string& text,
                                   bool enable_thinking) {
  if (!family_supports_qwen_think(family)) return text;
  if (!enable_thinking) return strip_qwen_think(text);
  return text;
}

void append_assistant_generation_prefix(ChatFamily family, const ChatTemplateOptions& opt,
                                        std::string& out_text) {
  out_text += "<|im_start|>assistant\n";
  std::ostringstream oss;
  append_think_suffix(family, opt, oss);
  out_text += oss.str();
}

}  // namespace llmoc::model
