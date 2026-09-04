#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "model/causal_lm.h"
#include "model/kv_cache.h"
#include "model/tokenizer_hf.h"

namespace llmoc::model {

struct TopLogprob {
  std::string token;
  float logprob = 0.f;
  std::vector<int> bytes;
};

struct TokenLogprob {
  std::string token;
  float logprob = 0.f;
  std::vector<int> bytes;
  std::vector<TopLogprob> top_logprobs;
};

struct GenerateRequest {
  std::vector<ChatMessage> messages;
  int max_new_tokens = 256;
  bool stream = false;
  float temperature = 0.f;  // 0 = greedy
  bool enable_thinking = false;
  // P0: auto|true|false —— true/auto 且 model->has_mtp() 才投机；否则逐步 decode
  std::string mtp = "auto";
  int spec_k = 3;
  // OpenAI chat: logprobs + top_logprobs (0..20). logprobs=false 时不计算。
  bool logprobs = false;
  int top_logprobs = 0;
};

struct GenerateResult {
  std::string text;
  std::vector<int32_t> token_ids;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  // 诊断：投机步数 / 采纳草稿数（无投机则 0）
  int mtp_verify_steps = 0;
  int mtp_draft_accepted = 0;
  // logprobs=true 时填充；与 token_ids 一一对应
  std::vector<TokenLogprob> logprobs;
  // exp(-mean(token logprob))；无 token 时为 0
  double perplexity = 0.0;
};

// text = 本步解码文本；lp 非空表示本步有 logprob（与 text 同属一个 token）
using TokenSink = std::function<void(const std::string& text, const TokenLogprob* lp)>;

class Generator {
 public:
  void init(ICausalLM* model, HfTokenizer* tok, int max_seq = 16384);

  GenerateResult generate(const GenerateRequest& req, const TokenSink& on_token = {});

  RadixKvPool& radix() { return radix_; }

 private:
  ICausalLM* model_ = nullptr;
  HfTokenizer* tok_ = nullptr;
  int max_seq_ = 16384;
  RadixKvPool radix_;
};

}  // namespace llmoc::model
