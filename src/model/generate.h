#pragma once
#include <functional>
#include <string>
#include <vector>

#include "model/causal_lm.h"
#include "model/kv_cache.h"
#include "model/tokenizer_hf.h"

namespace llmoc::model {

struct GenerateRequest {
  std::vector<ChatMessage> messages;
  int max_new_tokens = 256;
  bool stream = false;
  float temperature = 0.f;  // 0 = greedy
  bool enable_thinking = false;
  // P0: auto|true|false —— true/auto 且 model->has_mtp() 才投机；否则逐步 decode
  std::string mtp = "auto";
  int spec_k = 3;
};

struct GenerateResult {
  std::string text;
  std::vector<int32_t> token_ids;
  int prompt_tokens = 0;
  int completion_tokens = 0;
  // 诊断：投机步数 / 采纳草稿数（无投机则 0）
  int mtp_verify_steps = 0;
  int mtp_draft_accepted = 0;
};

class Generator {
 public:
  void init(ICausalLM* model, HfTokenizer* tok, int max_seq = 4096);

  GenerateResult generate(const GenerateRequest& req,
                          const std::function<void(const std::string& delta)>& on_token = {});

  RadixKvPool& radix() { return radix_; }

 private:
  ICausalLM* model_ = nullptr;
  HfTokenizer* tok_ = nullptr;
  int max_seq_ = 4096;
  RadixKvPool radix_;
};

}  // namespace llmoc::model
