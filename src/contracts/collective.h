#pragma once
// llm-on-cpu :: contracts/collective.h

#include "contracts/tensor_view.h"

namespace llmoc::contracts {

class ICollective {
 public:
  virtual ~ICollective() = default;
  virtual int world_size() const = 0;
  virtual void allreduce_sum(TensorView inout) = 0;
  virtual void alltoall(TensorView send, TensorView recv) = 0;
  virtual void broadcast(TensorView buf, int root) = 0;
};

}  // namespace llmoc::contracts
