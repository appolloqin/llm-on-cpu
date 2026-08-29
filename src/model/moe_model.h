#pragma once
// llm-on-cpu :: model/moe_model.h
// 标准 MoE 因果 LM(Qwen3-MoE / DeepSeek 系命名): GQA + router top-k + 专家 SwiGLU。
// Decode 路径经 ExpertPrefetcher; Prefill 走 WeightManager::get。

#include <memory>
#include <string>
#include <vector>

#include "hal/cpu_ops.h"
#include "model/causal_lm.h"
#include "weights/prefetch_pipeline.h"
#include "weights/weight_manager.h"

namespace llmoc::model {

struct MoeConfig {
  int hidden = 2048;
  int layers = 48;
  int n_heads = 16;
  int n_kv = 4;
  int head_dim = 128;
  int intermediate = 0;       // dense mlp
  int moe_intermediate = 768;
  int n_experts = 128;
  int topk = 8;
  int first_k_dense = 0;
  int vocab = 151936;
  float rms_eps = 1e-6f;
  float rope_theta = 1000000.f;
  float partial_rotary = 1.f;
  bool tie_embeddings = true;
  bool has_q_norm = false;
  bool fused_kv = false;  // selftest: kv_proj
};

class MoeModel final : public ICausalLM {
 public:
  void load(wt::WeightManager* wm, wt::ExpertPrefetcher* pref, const std::string& hf_config_json);

  const CausalLmMeta& meta() const override { return meta_; }
  void init_cache(SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, SessionCache& cache, std::vector<float>& logits,
               bool is_prefill) override;

  const MoeConfig& config() const { return cfg_; }

 private:
  wt::WeightManager* wm_ = nullptr;
  wt::ExpertPrefetcher* pref_ = nullptr;
  MoeConfig cfg_;
  CausalLmMeta meta_;
  std::string prefix_;  // "" or "language_model."
  hal::WDtype wd_ = hal::WDtype::kBF16;

  bool has(const std::string& name) const;
  const uint16_t* w(const std::string& name);
  void embed(int32_t token, float* out);
  void attn_layer(int layer, float* x, SessionCache& cache, int pos_start, int n_tok,
                  bool is_prefill);
  void dense_mlp(int layer, float* x, int n_tok);
  void moe_mlp_token(int layer, float* x /*[H]*/, bool use_prefetch, int* out_ids, float* out_w,
                     int* n_sel);
  void router_topk(const float* x, const uint16_t* Wgate, int* ids, float* weights);
};

}  // namespace llmoc::model
