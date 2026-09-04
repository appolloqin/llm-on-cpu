#pragma once
#include <memory>
#include <string>
#include <vector>

#include "contracts/exec_backend.h"
#include "glm/device/glm_device.h"
#include "glm/glm_config.h"
#include "glm/weights/glm_expert_prefetch.h"
#include "glm/weights/glm_weight_store.h"
#include "model/causal_lm.h"

namespace llmoc::glm {

class GlmFlashModel final : public model::ICausalLM {
 public:
  void load(const GlmEngineConfig& cfg);
  void load_strict(const GlmEngineConfig& cfg);

  bool weights_ready() const { return weights_ready_; }
  const std::string& load_error() const { return load_error_; }

  const model::CausalLmMeta& meta() const override { return meta_; }
  void init_cache(model::SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, model::SessionCache& cache,
               std::vector<float>& logits, bool is_prefill) override;

  ExecMode exec_mode() const { return mode_; }
  QuantKind quant() const { return quant_; }
  contracts::IExecBackend* exec() const { return exec_.get(); }

 private:
  void apply_geometry_from_header();
  void load_meta_sidecar(const std::string& glmq_path);
  void warm_gpu_attn_weights();
  void bind_expert_hosts();
  void setup_exec_backend(const GlmEngineConfig& cfg);
  void layer_forward(int layer, float* x, model::SessionCache& cache, int pos);
  void attn_linear_kda(int layer, const float* normed, float* oproj, model::SessionCache& cache);
  void attn_sparse_mla_or_gqa(int layer, const float* normed, float* oproj,
                              model::SessionCache& cache, int pos);
  void moe_ffn(int layer, float* x);
  void dense_ffn(int layer, float* x);
  void gemm_linear(const float* x, const std::string& wname, float* y, int M, int K);
  void gemm_bf16_named(const float* x, const std::string& wname, float* y, int M, int K);

  GlmEngineConfig cfg_;
  ExecMode mode_ = ExecMode::kPureCpu;
  QuantKind quant_ = QuantKind::kBf16;
  model::CausalLmMeta meta_;
  std::unique_ptr<IGlmDevice> device_;
  std::unique_ptr<contracts::IExecBackend> exec_;
  GlmWeightStore store_;
  GlmExpertPrefetch prefetch_;
  bool weights_ready_ = false;
  std::string load_error_;
  bool use_gpu_gemm_ = false;
  float rms_eps_ = 1e-6f;
  float rope_theta_ = 10000.f;
  int H_ = 0, L_ = 0, V_ = 0, nh_ = 0, nkv_ = 0, hd_ = 0, E_ = 0, topk_ = 0, I_ = 0;
  int qk_dim_ = 0, v_dim_ = 0;
  int first_k_dense_ = 0;
  int hc_mult_ = 1;
  int index_topk_ = 0;
  int index_dim_ = 0;
  int index_kpool_ = 4;
  bool index_kpool_compress_ = true;
  bool mhc_ = false;
  std::vector<std::string> layer_types_;
};

}  // namespace llmoc::glm
