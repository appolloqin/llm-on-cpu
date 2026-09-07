// llm-on-cpu :: model/qwen3_8_moe_int4_model.cpp
#include "model/qwen3_8_moe_int4_model.h"

namespace llmoc::model {

const char* Qwen38MoeInt4Model::int4_kind() const { return "qwen3_8_moe_int4"; }

const char* Qwen38MoeInt4Model::int4_class_name() const { return "Qwen38MoeInt4Model"; }

}  // namespace llmoc::model
