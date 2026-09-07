#pragma once
// llm-on-cpu :: model/qwen3_8_moe_int4_model.h
// Qwen3.8 MoE INT4 — separate entry from Qwen3.8 dense and from Qwen3.6-A3B MoE.
// Reuses Qwen36MoeInt4Model::moe_ffn_token; do not retune awq_sym zp here.

#include "model/qwen3_6_moe_int4_model.h"

namespace llmoc::model {

class Qwen38MoeInt4Model final : public Qwen36MoeInt4Model {
 protected:
  const char* int4_kind() const override;
  const char* int4_class_name() const override;
};

}  // namespace llmoc::model
