// llm-on-cpu :: model/graph/graph_executor.cpp
#include "model/graph/graph_executor.h"

#include <stdexcept>

#include "model/qwen3_5_model.h"

namespace llmoc::model::graph {

void GraphExecutor::load(ModelSpec spec, wt::WeightManager* wm,
                         const std::string& hf_config_json_path) {
  if (!wm) throw std::runtime_error("GraphExecutor: null WeightManager");
  spec_ = std::move(spec);
  meta_.hidden = spec_.hidden_size;
  meta_.layers = static_cast<int>(spec_.layers.size());
  meta_.vocab = spec_.vocab_size;
  meta_.is_moe = false;
  meta_.kind = "graph:" + spec_.name;

  if (!spec_.layers.empty()) {
    const auto& L0 = spec_.layers[0];
    meta_.n_kv = L0.attn.num_kv_heads;
    meta_.head_dim = L0.attn.head_dim;
    meta_.linear_num_v = L0.linear.num_value_heads;
    meta_.linear_dk = L0.linear.key_head_dim;
    meta_.linear_dv = L0.linear.value_head_dim;
    meta_.conv_k = L0.linear.conv_kernel;
    meta_.conv_dim = L0.linear.num_key_heads * L0.linear.key_head_dim * 2 +
                     L0.linear.num_value_heads * L0.linear.value_head_dim;
  }

  // P1 过渡：用现有 Qwen35 前向保证数值正确；Graph 循环替换在后续 PR。
  legacy_ = std::make_unique<Qwen35Model>();
  legacy_->load(wm, hf_config_json_path);
  meta_ = legacy_->meta();
  meta_.kind = "graph:" + spec_.name + "+legacy";
}

const CausalLmMeta& GraphExecutor::meta() const { return meta_; }

void GraphExecutor::init_cache(SessionCache& cache, int max_seq) const {
  if (!legacy_) throw std::runtime_error("GraphExecutor not loaded");
  legacy_->init_cache(cache, max_seq);
}

void GraphExecutor::forward(const std::vector<int32_t>& tokens, SessionCache& cache,
                            std::vector<float>& logits, bool is_prefill) {
  if (!legacy_) throw std::runtime_error("GraphExecutor not loaded");
  legacy_->forward(tokens, cache, logits, is_prefill);
}

void GraphExecutor::forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                                       std::vector<float>& logits_all, bool is_prefill) {
  if (!legacy_) throw std::runtime_error("GraphExecutor not loaded");
  legacy_->forward_all_logits(tokens, cache, logits_all, is_prefill);
}

void GraphExecutor::commit_prefix_state(int pos) {
  if (legacy_) legacy_->commit_prefix_state(pos);
}

bool GraphExecutor::has_mtp() const {
  return legacy_ && legacy_->has_mtp();
}

bool GraphExecutor::draft_propose(const std::vector<int32_t>& history, int draft_k,
                                  std::vector<int32_t>& out, int32_t pin_first) {
  if (!legacy_) {
    out.clear();
    return false;
  }
  return legacy_->draft_propose(history, draft_k, out, pin_first);
}

}  // namespace llmoc::model::graph
