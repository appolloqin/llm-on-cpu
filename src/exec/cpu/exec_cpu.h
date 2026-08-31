#pragma once
#include "contracts/exec_backend.h"
#include "contracts/expert_runtime.h"
#include "contracts/linear_op.h"
#include "exec/collective_noop.h"
#include "sched/placement_planner.h"

namespace llmoc::exec::cpu {

class GemmCpu final : public contracts::IGemm {
 public:
  bool gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) override;
};

class ExpertRuntimeCpu final : public contracts::IExpertRuntime {
 public:
  void prefetch(int, const contracts::ExpertId*, int) override {}
  void pin(int layer, const contracts::ExpertId* ids, int n, contracts::BlockHandle* out) override;
  void gemm_swiglu(const contracts::BlockHandle&, const contracts::TensorView&,
                   contracts::TensorView&) override {}
  void release(const contracts::BlockHandle*, int) override {}
  bool owns(contracts::ExpertId) const override { return true; }
};

class KvPolicyHost final : public contracts::IKvDevicePolicy {
 public:
  bool kv_on_device() const override { return false; }
};

class ExecCpu final : public contracts::IExecBackend {
 public:
  ExecCpu();
  contracts::ExecMode mode() const override { return contracts::ExecMode::kPureCpu; }
  contracts::ExecCaps caps() const override;
  const contracts::DeviceMesh& mesh() const override { return mesh_; }
  void configure(const sched::PlacementPlan& plan) override;
  contracts::IGemm* gemm() override { return &gemm_; }
  contracts::IExpertRuntime* experts() override { return &experts_; }
  contracts::IKvDevicePolicy* kv_policy() override { return &kv_; }
  contracts::ICollective* coll() override { return &coll_; }

 private:
  contracts::DeviceMesh mesh_;
  GemmCpu gemm_;
  ExpertRuntimeCpu experts_;
  KvPolicyHost kv_;
  CollectiveNoop coll_{1};
  sched::PlacementPlan plan_{};
  size_t dram_budget_ = 0;
};

}  // namespace llmoc::exec::cpu
