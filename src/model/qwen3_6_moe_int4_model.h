#pragma once
// llm-on-cpu :: model/qwen3_6_moe_int4_model.h
// Qwen3.6-*-A3B MoE INT4 path. Isolated from dense Qwen35Int4Model so MoE/AutoAWQ
// fixes cannot retune shared awq_sym zp used by Qwen3.5-4B.
// Qwen3.8 MoE subclasses this (qwen3_8_moe_int4_model.h) — keep moe_ffn here.

#include "model/qwen3_5_int4_model.h"

namespace llmoc::model {

class Qwen36MoeInt4Model : public Qwen35Int4Model {
 public:
  void load(qlwc::QlwcStore* store, const std::string& hf_config_json_path) override;

 protected:
  void moe_ffn_token(int layer, const float* normed, float* down_acc) override;
  // Subclasses (e.g. Qwen38 MoE) override kind without duplicating FFN/load guards.
  virtual const char* int4_kind() const { return "qwen3_6_moe_int4"; }
  virtual const char* int4_class_name() const { return "Qwen36MoeInt4Model"; }
};

}  // namespace llmoc::model
