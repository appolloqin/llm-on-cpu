// llm-on-cpu :: model/qwen3_8_int4_model.cpp
#include "model/qwen3_8_int4_model.h"

#include <stdexcept>
#include <string>

#include "common/log.h"
#include "weights/qlwc_format.h"

namespace llmoc::model {

void Qwen38Int4Model::load(qlwc::QlwcStore* store, const std::string& hf_config_json_path) {
  allow_moe_ = false;
  Qwen35Int4Model::load(store, hf_config_json_path);
  if (cfg_.is_moe) {
    throw std::runtime_error(
        "Qwen38Int4Model: MoE weights detected; use Qwen38MoeInt4Model instead");
  }
  meta_.kind = "qwen3_8_dense_int4";
  LOG_INFO("Qwen38Int4Model ready: layers=%d hidden=%d heads=%d (dense; awq_sym_zp=%d)",
           cfg_.layers, cfg_.hidden, cfg_.n_heads, qlwc::kLocalAwqSymZero);
}

}  // namespace llmoc::model
