// llm-on-cpu :: tests/unit/test_tokenizer.cpp
#include "test_main.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "common/engine_config.h"
#include "model/tokenizer_hf.h"

namespace fs = std::filesystem;

TINY_TEST(Tok, ChatTemplate) {
  using llmoc::model::ChatMessage;
  const std::string off = llmoc::model::apply_qwen_chat_template(
      {{"system", "you are helpful"}, {"user", "hi"}}, true, false);
  EXPECT_TRUE(off.find("<|im_start|>system") != std::string::npos);
  EXPECT_TRUE(off.find("<|im_start|>assistant\n") != std::string::npos);
  EXPECT_TRUE(off.find("<think>\n\n</think>\n\n") != std::string::npos);
  const std::string on = llmoc::model::apply_qwen_chat_template({{"user", "hi"}}, true, true);
  EXPECT_TRUE(on.find("<think>\n") != std::string::npos);
  EXPECT_TRUE(on.find("</think>") == std::string::npos);
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
         "server:\n  port: 18080\n";
  }
  auto cfg = llmoc::EngineConfig::load(p.string());
  EXPECT_EQ(cfg.model_path, std::string("models/Qwen3.5-4B.lwc"));
  EXPECT_EQ(cfg.server_port, 18080);
  EXPECT_EQ(cfg.resolve_tokenizer_dir(), std::string("models/Qwen3.5-4B-hf"));
}

TINY_TEST(Tok, EncodeIfPresent) {
  const fs::path tj = "models/Qwen3.5-4B-hf/tokenizer.json";
  if (!fs::exists(tj)) return;
  llmoc::model::HfTokenizer tok;
  tok.load(tj.string());
  const auto ids = tok.encode("你好");
  EXPECT_TRUE(!ids.empty());
  // <think> 非 special，但必须整段编码且不能卡死
  const auto think = tok.encode("<think>\n\n</think>\n\n");
  EXPECT_TRUE(think.size() >= 2);
  EXPECT_TRUE(think.size() < 32);
}
