#pragma once
// llm-on-cpu :: families/deepseek_v4/ds_stub_model.h
// DS-STUB-v0：latent KV + MoE top-k；可挂 BF16/NVFP4 GPU GEMM（非论文 CSA/HCA）

#include <cstdint>
#include <string>
#include <vector>

#include "hal/quant_views.h"
#include "model/causal_lm.h"
#include "sched/mode_controller.h"

namespace llmoc::families::deepseek {

struct DsStubGeometry {
  int hidden = 64;
  int layers = 2;
  int vocab = 128;
  int n_experts = 4;
  int top_k = 2;
  int d_latent = 16;
  int intermediate = 128;
};

class DsStubModel final : public model::ICausalLM {
 public:
  void load_file(const std::string& path, contracts::ExecMode mode);
  void load_synthetic(DsStubGeometry g, contracts::ExecMode mode);
  void warm_gpu_weights();

  struct ExpW {
    std::vector<uint8_t> q;
    std::vector<uint8_t> scales;
    hal::Nvfp4View view;
  };

  const model::CausalLmMeta& meta() const override { return meta_; }
  void init_cache(model::SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, model::SessionCache& cache,
               std::vector<float>& logits, bool is_prefill) override;

 private:
  void finish_load(contracts::ExecMode mode);
  void gemm_bf16(const float* x, const uint16_t* W, float* y, int M, int K);
  void gemm_nv(const float* x, const hal::Nvfp4View& W, float* y);

  DsStubGeometry g_{};
  model::CausalLmMeta meta_;
  contracts::ExecMode mode_ = contracts::ExecMode::kPureCpu;
  bool use_gpu_ = false;

  std::vector<uint16_t> embed_;      // [V,H]
  std::vector<uint16_t> lm_head_;    // [V,H]
  std::vector<uint16_t> final_norm_;  // [H]
  // per layer
  std::vector<std::vector<uint16_t>> ln1_, ln2_;
  std::vector<std::vector<uint16_t>> w_qc_, w_out_;  // latent attn
  std::vector<std::vector<uint16_t>> w_gate_;        // [E,H]
  // experts NVFP4 packed
  std::vector<std::vector<ExpW>> exp_gate_, exp_up_, exp_down_;  // [L][E]
};

// Write tiny DSS1 file for smoke tests.
void write_fake_dskq(const std::string& path, DsStubGeometry g = {});

}  // namespace llmoc::families::deepseek
