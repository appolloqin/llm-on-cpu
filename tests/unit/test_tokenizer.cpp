// llm-on-cpu :: tests/unit/test_tokenizer.cpp
#include "test_main.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "common/engine_config.h"
#include "model/chat_templates.h"
#include "model/tokenizer_hf.h"

namespace fs = std::filesystem;

TINY_TEST(Tok, ChatTemplateFamilies) {
  using llmoc::model::ChatMessage;
  using llmoc::model::ChatFamily;
  using llmoc::model::ChatTemplateOptions;
  using llmoc::model::ThinkingOffStyle;
  using llmoc::model::apply_chat_template;
  using llmoc::model::chat_family_from_kind;

  EXPECT_TRUE(chat_family_from_kind("qwen3_5_int4") == ChatFamily::kQwen35);
  EXPECT_TRUE(chat_family_from_kind("qwen3_6_moe_int4") == ChatFamily::kQwen36);
  EXPECT_TRUE(chat_family_from_kind("qwen3_8_dense_int4") == ChatFamily::kQwen38);
  EXPECT_TRUE(chat_family_from_kind("glm53_flash") == ChatFamily::kGlm);
  EXPECT_TRUE(chat_family_from_kind("deepseek_v4_stub_nvfp4") == ChatFamily::kDeepSeek);
  EXPECT_TRUE(chat_family_from_kind("kimi_k3_stub_bf16") == ChatFamily::kKimi);

  const std::vector<ChatMessage> msgs = {{"user", "hi"}};
  ChatTemplateOptions off;
  off.enable_thinking = false;
  ChatTemplateOptions on;
  on.enable_thinking = true;

  const std::string q35_off = apply_chat_template(ChatFamily::kQwen35, msgs, off);
  EXPECT_TRUE(q35_off.find("<think>\n\n</think>\n\n") != std::string::npos);

  const std::string q36_off = apply_chat_template(ChatFamily::kQwen36, msgs, off);
  EXPECT_TRUE(q36_off.find("<think>") == std::string::npos);

  const std::string q38_off = apply_chat_template(ChatFamily::kQwen38, msgs, off);
  EXPECT_TRUE(q38_off.find("<think>") == std::string::npos);

  ChatTemplateOptions force_empty = off;
  force_empty.thinking_off_style = ThinkingOffStyle::kEmptyPrefill;
  const std::string q36_force = apply_chat_template(ChatFamily::kQwen36, msgs, force_empty);
  EXPECT_TRUE(q36_force.find("<think>\n\n</think>\n\n") != std::string::npos);

  const std::string q36_on = apply_chat_template(ChatFamily::kQwen36, msgs, on);
  EXPECT_TRUE(q36_on.find("<think>\n") != std::string::npos);
  EXPECT_TRUE(q36_on.find("</think>") == std::string::npos);

  const std::string glm_on = apply_chat_template(ChatFamily::kGlm, msgs, on);
  EXPECT_TRUE(glm_on.find("<think>") == std::string::npos);
  EXPECT_TRUE(glm_on.find("<|im_start|>assistant\n") != std::string::npos);

  // Legacy alias stays on Qwen3.5 policy.
  const std::string legacy = llmoc::model::apply_qwen_chat_template(msgs, true, false);
  EXPECT_TRUE(legacy.find("<think>\n\n</think>\n\n") != std::string::npos);

  const std::string stripped =
      llmoc::model::strip_qwen_think("<think>\n\n</think>\n\n你好");
  EXPECT_EQ(stripped, std::string("你好"));
}

TINY_TEST(Tok, EngineConfig) {
  fs::create_directories("models");
  const fs::path p = "models/_test_engine.yaml";
  {
    std::ofstream f(p);
    f << "model:\n  path: models/Qwen3.5-4B.lwc\n  dtype: bf16\n"
         "thinking:\n  enable: true\n  off_style: empty_prefill\n"
         "server:\n  port: 18080\n";
  }
  auto cfg = llmoc::EngineConfig::load(p.string());
  EXPECT_EQ(cfg.model_path, std::string("models/Qwen3.5-4B.lwc"));
  EXPECT_EQ(cfg.server_port, 18080);
  EXPECT_EQ(cfg.resolve_tokenizer_dir(), std::string("models/Qwen3.5-4B-hf"));
  EXPECT_TRUE(cfg.thinking_enable);
  EXPECT_EQ(cfg.thinking_off_style, std::string("empty_prefill"));
}

TINY_TEST(Tok, EncodeSpecial) {
  const char* paths[] = {"models/Qwen3.5-4B-hf/tokenizer.json",
                         "models/Qwen3.5-4B.int4-hf/tokenizer.json"};
  std::string tok_path;
  for (const char* p : paths) {
    if (fs::exists(p)) {
      tok_path = p;
      break;
    }
  }
  if (tok_path.empty()) {
    std::fprintf(stderr, "[SKIP] Tok.EncodeSpecial — no tokenizer.json\n");
    EXPECT_TRUE(true);
    return;
  }
  llmoc::model::HfTokenizer tok;
  tok.load(tok_path);
  // <think> 非 special，但必须整段编码且不能卡死
  const auto think = tok.encode("<think>\n\n</think>\n\n");
  EXPECT_TRUE(think.size() >= 2);
  EXPECT_TRUE(think.size() < 32);
}
