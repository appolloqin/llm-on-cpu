#pragma once
// llm-on-cpu :: contracts/family_pack.h

#include <cstdint>
#include <memory>
#include <string>

#include "contracts/exec_backend.h"
#include "contracts/kv_store.h"

namespace llmoc::contracts {

struct ActiveSetProfile {
  std::string family;
  bool has_moe = false;
  int n_layers = 0;
  int n_experts = 0;
  int top_k = 0;
  uint64_t per_layer_attn_bytes = 0;
  uint64_t shared_bytes = 0;
  uint64_t expert_bytes = 0;
  uint64_t kv_working_bytes = 0;
  uint64_t workspace_bytes = 0;
};

class IFamilyPack {
 public:
  virtual ~IFamilyPack() = default;
  virtual const char* name() const = 0;
  virtual ActiveSetProfile active_profile() const = 0;
};

}  // namespace llmoc::contracts
