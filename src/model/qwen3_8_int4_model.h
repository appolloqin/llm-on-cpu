#pragma once
// llm-on-cpu :: model/qwen3_8_int4_model.h
// Qwen3.8 dense INT4 — isolated from Qwen3.5-4B and from Qwen3.8 MoE
// (see qwen3_8_moe_int4_model.h). Shares GDN+GQA forward via Qwen35Int4Model.

#include "model/qwen3_5_int4_model.h"

namespace llmoc::model {

class Qwen38Int4Model final : public Qwen35Int4Model {
 public:
  void load(qlwc::QlwcStore* store, const std::string& hf_config_json_path) override;
};

}  // namespace llmoc::model
