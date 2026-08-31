#pragma once
// llm-on-cpu :: model/causal_lm.h
// 统一因果 LM 接口: Qwen3.5 hybrid / MoE 共用 Generator。

#include <cstdint>
#include <string>
#include <vector>

#include "model/kv_cache.h"

namespace llmoc::model {

struct ChatMessage;  // tokenizer_hf.h

struct CausalLmMeta {
  int hidden = 0;
  int layers = 0;
  int vocab = 0;
  int n_kv = 0;
  int head_dim = 0;
  int linear_num_v = 0;
  int linear_dk = 0;
  int linear_dv = 0;
  int conv_dim = 0;
  int conv_k = 4;
  bool is_moe = false;
  std::string kind;  // "qwen3_5" | "moe"
};

class ICausalLM {
 public:
  virtual ~ICausalLM() = default;
  virtual const CausalLmMeta& meta() const = 0;
  virtual void init_cache(SessionCache& cache, int max_seq) const = 0;
  virtual void forward(const std::vector<int32_t>& tokens, SessionCache& cache,
                       std::vector<float>& logits, bool is_prefill) = 0;

  // 一次前向写出每个位置的 logits（[n * vocab]），供 MTP 批量 verify。
  // 默认逐 token 调用 forward（正确但无加速）；稠密/Qwen 路径应覆盖。
  virtual void forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                                  std::vector<float>& logits_all, bool is_prefill) {
    logits_all.clear();
    if (tokens.empty()) return;
    std::vector<float> one;
    for (size_t i = 0; i < tokens.size(); ++i) {
      forward({tokens[i]}, cache, one, is_prefill && i == 0);
      logits_all.insert(logits_all.end(), one.begin(), one.end());
    }
  }

  // Greedy decode 快路径：n=1、temperature≈0 时可跳过全词表 logits 写回。
  virtual bool forward_decode_greedy(const std::vector<int32_t>& tokens, SessionCache& cache,
                                     int32_t& out_token) {
    (void)tokens;
    (void)cache;
    (void)out_token;
    return false;
  }

  // 批量 verify 截断后：把 last_hidden/last_logits 切到 forward_all 的第 pos 个位置
  virtual void commit_prefix_state(int /*pos*/) {}

  // P0 MTP：无头时保持 false；禁止用随机草稿充数
  virtual bool has_mtp() const { return false; }
  // 基于已提交 history 草拟至多 draft_k 个 token；失败返回 false
  virtual bool draft_propose(const std::vector<int32_t>& /*history*/, int /*draft_k*/,
                             std::vector<int32_t>& /*out*/, int32_t /*pin_first*/ = -1) {
    return false;
  }

  // 视觉：prefill 前注入 image_pad 位置的 embedding（[n_tok * hidden]）
  virtual bool has_vision() const { return false; }
  virtual void set_vision_embeds(std::vector<float> /*embeds*/, int /*n_tok*/) {}
  virtual void clear_vision_embeds() {}
  virtual int32_t image_pad_token_id() const { return -1; }
  virtual int32_t vision_start_token_id() const { return -1; }
  virtual int32_t vision_end_token_id() const { return -1; }
  // 编码 messages 中的图片 → embeds + 每图 merged token 数；无视觉返回 false
  virtual bool encode_message_images(const std::vector<struct ChatMessage>& /*messages*/,
                                     std::vector<float>& /*embeds_out*/,
                                     std::vector<int>& /*pad_counts_out*/) {
    return false;
  }
};

}  // namespace llmoc::model
