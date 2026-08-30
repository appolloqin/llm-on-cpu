#pragma once
// llm-on-cpu :: model/qwen3_5_model.h
#include <string>
#include <vector>

#include "hal/cpu_ops.h"
#include "model/causal_lm.h"
#include "model/mtp_head.h"
#include "weights/weight_manager.h"

namespace llmoc::model {

struct Qwen35Config {
  int hidden = 2560;
  int layers = 32;
  int n_heads = 16;
  int n_kv = 4;
  int head_dim = 256;
  int intermediate = 9216;
  int vocab = 248320;
  float rms_eps = 1e-6f;
  float rope_theta = 10000000.f;
  float partial_rotary = 0.25f;
  int linear_num_k = 16;
  int linear_num_v = 32;
  int linear_dk = 128;
  int linear_dv = 128;
  int conv_k = 4;
  bool tie_embeddings = true;
  std::vector<std::string> layer_types;
};

class Qwen35Model final : public ICausalLM {
 public:
  void load(wt::WeightManager* wm, const std::string& hf_config_json_path);
  const Qwen35Config& config() const { return cfg_; }

  const CausalLmMeta& meta() const override { return meta_; }
  void init_cache(SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, SessionCache& cache, std::vector<float>& logits,
               bool is_prefill) override;
  void forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                          std::vector<float>& logits_all, bool is_prefill) override;
  void commit_prefix_state(int pos) override;
  bool has_mtp() const override;
  bool draft_propose(const std::vector<int32_t>& history, int draft_k,
                     std::vector<int32_t>& out) override;

 private:
  wt::WeightManager* wm_ = nullptr;
  Qwen35Config cfg_;
  CausalLmMeta meta_;
  std::string prefix_ = "language_model.";
  std::string lm_head_name_;  // empty => tied to embed
  hal::WDtype wd_ = hal::WDtype::kBF16;

  const uint16_t* w(const std::string& name);
  void embed(int32_t token, float* out);
  void layer_forward(int layer, float* x, SessionCache& cache, int pos_start, int n_tok,
                     bool is_prefill);
  MtpWeightAccess mtp_access();

  static bool mtp_has_cb(void* ctx, const std::string& name);
  static void mtp_gemm_cb(void* ctx, const float* x, const std::string& wname, float* y, int M,
                          int K);
  static const uint16_t* mtp_pass_cb(void* ctx, const std::string& name);
  static void mtp_embed_cb(void* ctx, int32_t token, float* out);
  static float mtp_embed_dot_cb(void* ctx, const float* h, int32_t token);

  std::vector<float> last_hidden_;
  std::vector<float> last_logits_;
  std::vector<float> prefix_hiddens_;
  std::vector<float> prefix_logits_;
};

}  // namespace llmoc::model
