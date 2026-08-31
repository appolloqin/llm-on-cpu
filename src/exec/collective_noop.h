#pragma once
// llm-on-cpu :: exec/collective_noop.h

#include "contracts/collective.h"

namespace llmoc::exec {

class CollectiveNoop final : public contracts::ICollective {
 public:
  explicit CollectiveNoop(int world = 1) : world_(world < 1 ? 1 : world) {}
  int world_size() const override { return world_; }
  void allreduce_sum(contracts::TensorView) override {}
  void alltoall(contracts::TensorView, contracts::TensorView) override {}
  void broadcast(contracts::TensorView, int) override {}

 private:
  int world_;
};

}  // namespace llmoc::exec
