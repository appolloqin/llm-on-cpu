#pragma once
// llm-on-cpu :: contracts/expert_runtime.h

#include "contracts/tensor_view.h"
#include "contracts/weight_block.h"

namespace llmoc::contracts {

class IExpertRuntime {
 public:
  virtual ~IExpertRuntime() = default;
  virtual void prefetch(int layer, const ExpertId* ids, int n) = 0;
  virtual void pin(int layer, const ExpertId* ids, int n, BlockHandle* out) = 0;
  virtual void gemm_swiglu(const BlockHandle& h, const TensorView& x, TensorView& y) = 0;
  virtual void release(const BlockHandle* hs, int n) = 0;
  virtual bool owns(ExpertId id) const = 0;
};

}  // namespace llmoc::contracts
