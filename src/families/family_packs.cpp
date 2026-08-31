// llm-on-cpu :: families/family_packs.cpp
#include "families/family_packs.h"

namespace llmoc::families {

contracts::ActiveSetProfile Qwen38Pack::active_profile() const {
  contracts::ActiveSetProfile a;
  a.family = "qwen38";
  a.has_moe = false;
  a.n_layers = 64;
  a.n_experts = 0;
  a.top_k = 0;
  // Dense INT4 ~ half of BF16; planner uses these as estimates until weighed.
  a.per_layer_attn_bytes = 200ull << 20;
  a.shared_bytes = 500ull << 20;
  a.expert_bytes = 0;
  a.kv_working_bytes = 256ull << 20;
  a.workspace_bytes = 64ull << 20;
  return a;
}

void Glm53Pack::set_geometry(int layers, int n_experts, int top_k, uint64_t expert_bytes,
                             uint64_t attn_bytes) {
  ap_.family = "glm53_flash";
  ap_.has_moe = true;
  ap_.n_layers = layers;
  ap_.n_experts = n_experts;
  ap_.top_k = top_k;
  ap_.expert_bytes = expert_bytes;
  ap_.per_layer_attn_bytes = attn_bytes;
  ap_.shared_bytes = attn_bytes;
  ap_.kv_working_bytes = 512ull << 20;
  ap_.workspace_bytes = 128ull << 20;
}

contracts::ActiveSetProfile Glm53Pack::active_profile() const {
  if (ap_.n_layers > 0) return ap_;
  contracts::ActiveSetProfile a;
  a.family = "glm53_flash";
  a.has_moe = true;
  a.n_layers = 45;
  a.n_experts = 128;
  a.top_k = 8;
  // NVFP4 expert rough ~ tens of MiB; conservative 64MiB
  a.expert_bytes = 64ull << 20;
  a.per_layer_attn_bytes = 128ull << 20;
  a.shared_bytes = 256ull << 20;
  a.kv_working_bytes = 512ull << 20;
  a.workspace_bytes = 128ull << 20;
  return a;
}

contracts::ActiveSetProfile DeepSeekV4Pack::active_profile() const {
  contracts::ActiveSetProfile a;
  a.family = "deepseek_v4";
  a.has_moe = true;
  a.n_layers = 60;
  a.n_experts = 256;
  a.top_k = 8;
  a.expert_bytes = 48ull << 20;
  a.per_layer_attn_bytes = 96ull << 20;
  a.shared_bytes = 200ull << 20;
  a.kv_working_bytes = 256ull << 20;  // compressed KV
  a.workspace_bytes = 128ull << 20;
  return a;
}

contracts::ActiveSetProfile KimiK3Pack::active_profile() const {
  contracts::ActiveSetProfile a;
  a.family = "kimi_k3";
  a.has_moe = true;
  a.n_layers = 80;
  a.n_experts = 896;
  a.top_k = 16;
  // ~104B active → single-card 24G must fail active-set planner
  a.expert_bytes = 2ull << 30;  // ~2 GiB / expert (quantized still huge at top-k)
  a.per_layer_attn_bytes = 1ull << 30;
  a.shared_bytes = 2ull << 30;
  a.kv_working_bytes = 2ull << 30;
  a.workspace_bytes = 1ull << 30;
  return a;
}

}  // namespace llmoc::families
