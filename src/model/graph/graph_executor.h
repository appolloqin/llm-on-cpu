#pragma once
// llm-on-cpu :: model/graph/graph_executor.h
// P1 骨架：配方 + 权重 → ICausalLM。首版 forward 委托 Qwen35Model（过渡），
// 后续按 LayerSpec 调度 ops，去掉专用前向文件。

#include <memory>
#include <string>

#include "model/causal_lm.h"
#include "model/graph/model_spec.h"
#include "weights/weight_manager.h"

namespace llmoc::model {
class Qwen35Model;
}

namespace llmoc::model::graph {

class GraphExecutor final : public ICausalLM {
 public:
  // 加载 recipe；若 backend=qwen3_5_legacy 则内部挂 Qwen35Model 保证可跑。
  void load(ModelSpec spec, wt::WeightManager* wm, const std::string& hf_config_json_path);

  const CausalLmMeta& meta() const override;
  void init_cache(SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, SessionCache& cache, std::vector<float>& logits,
               bool is_prefill) override;
  void forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                          std::vector<float>& logits_all, bool is_prefill) override;
  void commit_prefix_state(int pos) override;
  bool has_mtp() const override;
  bool draft_propose(const std::vector<int32_t>& history, int draft_k,
                     std::vector<int32_t>& out, int32_t pin_first = -1) override;

  const ModelSpec& spec() const { return spec_; }

 private:
  ModelSpec spec_;
  CausalLmMeta meta_;
  std::unique_ptr<Qwen35Model> legacy_;  // 过渡委托
};

}  // namespace llmoc::model::graph
