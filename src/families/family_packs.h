#pragma once
// llm-on-cpu :: families/* — ActiveSet profiles + pack registration

#include "contracts/family_pack.h"

namespace llmoc::families {

class Qwen38Pack final : public contracts::IFamilyPack {
 public:
  const char* name() const override { return "qwen38"; }
  contracts::ActiveSetProfile active_profile() const override;
};

class Glm53Pack final : public contracts::IFamilyPack {
 public:
  const char* name() const override { return "glm53_flash"; }
  contracts::ActiveSetProfile active_profile() const override;
  // Fill from loaded GLMQ geometry when available.
  void set_geometry(int layers, int n_experts, int top_k, uint64_t expert_bytes,
                    uint64_t attn_bytes);
 private:
  contracts::ActiveSetProfile ap_{};
};

class DeepSeekV4Pack final : public contracts::IFamilyPack {
 public:
  const char* name() const override { return "deepseek_v4"; }
  contracts::ActiveSetProfile active_profile() const override;
};

class KimiK3Pack final : public contracts::IFamilyPack {
 public:
  const char* name() const override { return "kimi_k3"; }
  contracts::ActiveSetProfile active_profile() const override;
};

}  // namespace llmoc::families
