#pragma once
// llm-on-cpu :: model/qwen3_6_moe_int4_model.h
// Qwen3.6-*-A3B MoE INT4 path. Isolated from dense Qwen35Int4Model so MoE/AutoAWQ
// fixes cannot retune shared awq_sym zp used by Qwen3.5-4B.
// Qwen3.8 MoE subclasses this (qwen3_8_moe_int4_model.h) — keep moe_ffn here.

#include <memory>

#include "model/qwen3_5_int4_model.h"
#include "moe/host_banks.h"
#include "moe/offload_cache.h"

namespace llmoc::model {

struct MoeOffloadRuntimeConfig {
  std::string backend = "hybrid";  // offload | hybrid | cpu
  int cache_slots = 0;             // 0 = auto
  bool host_pin = true;
  bool prefill_overlap = true;
  float hybrid_fetch_frac = 0.5f;
  double dram_hot_gb = 0.0;  // 0 = unlimited pin budget
};

class Qwen36MoeInt4Model : public Qwen35Int4Model {
 public:
  void load(qlwc::QlwcStore* store, const std::string& hf_config_json_path) override;

  // After load (+ CUDA enable): fill host banks + build slot cache (pin deferred).
  void init_moe_offload(const MoeOffloadRuntimeConfig& cfg);
  // Call AFTER warm_gpu_int4 so cudaHostRegister does not starve attn cudaMalloc/JIT.
  void pin_moe_host_banks();
  bool moe_offload_ready() const { return moe_cache_ && moe_cache_->ready(); }
  moe::OffloadMoeCache* moe_cache() { return moe_cache_.get(); }
  moe::QlwcExpertHostBanks* moe_banks() { return moe_banks_.get(); }

  // MoE-first VRAM reservation bytes (device footprint × slots), 0 if not inited.
  size_t moe_slot_reserve_bytes() const { return moe_slot_reserve_bytes_; }
  int moe_cache_slots() const { return moe_cache_ ? moe_cache_->cache_size() : 0; }

 protected:
  void moe_ffn_token(int layer, const float* normed, float* down_acc) override;
  void on_prefill_begin() override;
  void on_prefill_layer(int layer, int n_tok) override;
  void on_prefill_layer_done(int layer) override;

  virtual const char* int4_kind() const { return "qwen3_6_moe_int4"; }
  virtual const char* int4_class_name() const { return "Qwen36MoeInt4Model"; }

  void moe_ffn_token_offload(int layer, const float* normed, float* down_acc,
                             const std::vector<int>& order, const std::vector<float>& logits,
                             double wsum, int K);
  void moe_cpu_experts(int layer, const float* normed, float* down_acc,
                       const std::vector<int>& experts, const std::vector<float>& logits,
                       double wsum);

  std::unique_ptr<moe::QlwcExpertHostBanks> moe_banks_;
  std::unique_ptr<moe::OffloadMoeCache> moe_cache_;
  MoeOffloadRuntimeConfig moe_rt_{};
  size_t moe_slot_reserve_bytes_ = 0;
};

}  // namespace llmoc::model
