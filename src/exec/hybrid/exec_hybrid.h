#pragma once
#include "contracts/exec_backend.h"
#include "contracts/expert_runtime.h"
#include "contracts/linear_op.h"
#include "sched/placement_planner.h"

#include <memory>

namespace llmoc::exec::hybrid {

class GemmHybrid final : public contracts::IGemm {
 public:
  bool gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) override;
};

class ExpertRuntimeCpuBridge final : public contracts::IExpertRuntime {
 public:
  void prefetch(int, const contracts::ExpertId*, int) override {}
  void pin(int layer, const contracts::ExpertId* ids, int n, contracts::BlockHandle* out) override;
  void gemm_swiglu(const contracts::BlockHandle&, const contracts::TensorView&,
                   contracts::TensorView&) override {}
  void release(const contracts::BlockHandle*, int) override {}
  bool owns(contracts::ExpertId) const override { return true; }  // experts on CPU
};

class KvPolicyDevice final : public contracts::IKvDevicePolicy {
 public:
  bool kv_on_device() const override { return true; }
};

class ExecHybrid final : public contracts::IExecBackend {
 public:
  explicit ExecHybrid(contracts::DeviceMesh mesh);
  contracts::ExecMode mode() const override { return contracts::ExecMode::kHybridGpu; }
  contracts::ExecCaps caps() const override;
  const contracts::DeviceMesh& mesh() const override { return mesh_; }
  void configure(const sched::PlacementPlan& plan) override;
  contracts::IGemm* gemm() override { return &gemm_; }
  contracts::IExpertRuntime* experts() override { return &experts_; }
  contracts::IKvDevicePolicy* kv_policy() override { return &kv_; }
  contracts::ICollective* coll() override { return coll_.get(); }

 private:
  contracts::DeviceMesh mesh_;
  GemmHybrid gemm_;
  ExpertRuntimeCpuBridge experts_;
  KvPolicyDevice kv_;
  std::unique_ptr<contracts::ICollective> coll_;
  sched::PlacementPlan plan_{};
  size_t vram_budget_ = 0;
  size_t dram_budget_ = 0;
};

}  // namespace llmoc::exec::hybrid
