#pragma once
// llm-on-cpu :: model/mtp_head.h
// Qwen3.5 内置 MTP 草稿头（对齐 vLLM Qwen3_5MultiTokenPredictor 数据流）。

#include <cstdint>
#include <string>
#include <vector>

#include "hal/cpu_ops.h"
#include "model/kv_cache.h"

namespace llmoc::model {

// 权重访问：name → BF16/F16 指针或 INT4 gemm；由宿主模型注入。
struct MtpWeightAccess {
  int hidden = 2560;
  int n_heads = 16;
  int n_kv = 4;
  int head_dim = 256;
  int intermediate = 9216;
  int vocab = 248320;
  float rms_eps = 1e-6f;
  float rope_theta = 10000000.f;
  float partial_rotary = 0.25f;
  bool rms_one_plus = true;
  hal::WDtype pass_dt = hal::WDtype::kBF16;

  // 返回 false 表示缺权重
  bool (*has)(void* ctx, const std::string& name) = nullptr;
  void (*gemm)(void* ctx, const float* x, const std::string& wname, float* y, int M, int K) =
      nullptr;
  const uint16_t* (*pass)(void* ctx, const std::string& name) = nullptr;  // RMSNorm 等
  void (*embed)(void* ctx, int32_t token, float* out) = nullptr;
  // 可选：上一轮主模型 logits，用于 MTP 只在 top_m 候选上打分（避开全词表 GEMM）
  const float* hint_logits = nullptr;
  int hint_top_m = 256;
  // 可选：单行 embedding 点积（INT4/BF16）；若为空则退回全量 gemm
  float (*embed_dot)(void* ctx, const float* h, int32_t token) = nullptr;
  void* ctx = nullptr;
};

bool mtp_weights_present(const MtpWeightAccess& wa);

// 基于主模型最终 hidden（lm_head 前）草拟 draft_k 个 token；失败返回 false。
bool mtp_draft_propose(const MtpWeightAccess& wa, const std::vector<float>& last_hidden,
                       const std::vector<int32_t>& history, int draft_k,
                       std::vector<int32_t>& out);

}  // namespace llmoc::model
