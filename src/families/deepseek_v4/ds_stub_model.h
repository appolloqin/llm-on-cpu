#pragma once
// llm-on-cpu :: families/deepseek_v4/ds_stub_model.h
// DS-STUB-v0：latent KV + MoE top-k；专家量化 bf16 | awq_int4 | nvfp4（按二进制锁定）

#include <cstdint>
#include <string>
#include <vector>

#include "hal/quant_views.h"
#include "model/causal_lm.h"
#include "sched/mode_controller.h"

namespace llmoc::families::deepseek {

enum class ExpertQuant { kBf16, kAwqInt4, kNvfp4 };

inline const char* expert_quant_name(ExpertQuant q) {
  switch (q) {
    case ExpertQuant::kBf16: return "bf16";
    case ExpertQuant::kAwqInt4: return "awq_int4";
    default: return "nvfp4";
  }
}

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
  void load_file(const std::string& path, contracts::ExecMode mode, ExpertQuant expert_q);
  void load_synthetic(DsStubGeometry g, contracts::ExecMode mode, ExpertQuant expert_q);
  void warm_gpu_weights();
  ExpertQuant expert_quant() const { return expert_q_; }

  struct ExpW {
    std::vector<uint16_t> bf16;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scales_u16;
    std::vector<float> scales_f32;
    std::vector<uint8_t> scales_fp8;
    hal::AwqView awq;
    hal::Nvfp4View nv;
    int M = 0;
    int K = 0;
  };

  const model::CausalLmMeta& meta() const override { return meta_; }
  void init_cache(model::SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, model::SessionCache& cache,
               std::vector<float>& logits, bool is_prefill) override;

 private:
  void finish_load(contracts::ExecMode mode);
  void gemm_bf16(const float* x, const uint16_t* W, float* y, int M, int K);
  void gemm_expert(const float* x, const ExpW& W, float* y);

  DsStubGeometry g_{};
  ExpertQuant expert_q_ = ExpertQuant::kNvfp4;
  model::CausalLmMeta meta_;
  contracts::ExecMode mode_ = contracts::ExecMode::kPureCpu;
  bool use_gpu_ = false;

  std::vector<uint16_t> embed_;
  std::vector<uint16_t> lm_head_;
  std::vector<uint16_t> final_norm_;
  std::vector<std::vector<uint16_t>> ln1_, ln2_;
  std::vector<std::vector<uint16_t>> w_qc_, w_out_;
  std::vector<std::vector<uint16_t>> w_gate_;
  std::vector<std::vector<ExpW>> exp_gate_, exp_up_, exp_down_;
};

void write_fake_dskq(const std::string& path, DsStubGeometry g = {});

}  // namespace llmoc::families::deepseek
