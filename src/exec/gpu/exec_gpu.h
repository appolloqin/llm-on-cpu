#pragma once
#include <memory>
#include <unordered_map>
#include <utility>

#include "contracts/exec_backend.h"
#include "contracts/expert_runtime.h"
#include "contracts/linear_op.h"
#include "exec/gpu/residency_gpu.h"
#include "sched/placement_planner.h"

namespace llmoc::exec::gpu {

class GemmGpu final : public contracts::IGemm {
 public:
  bool gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) override;
};

class ExpertRuntimeGpu final : public contracts::IExpertRuntime {
 public:
  ExpertRuntimeGpu(contracts::DeviceMesh mesh, int n_experts);
  void set_n_experts(int n) { n_experts_ = n; }
  void set_local_rank(int r) { local_rank_ = r; }
  ExpertSlotPool& pool() { return pool_; }

  // Bind host weight blob for subsequent pin/prefetch H2D.
  void bind_host(contracts::ExpertId id, const void* host, size_t bytes);

  void prefetch(int layer, const contracts::ExpertId* ids, int n) override;
  void pin(int layer, const contracts::ExpertId* ids, int n, contracts::BlockHandle* out) override;
  void gemm_swiglu(const contracts::BlockHandle&, const contracts::TensorView&,
                   contracts::TensorView&) override {}
  void release(const contracts::BlockHandle* hs, int n) override;
  bool owns(contracts::ExpertId id) const override;

  int slot_capacity() const { return pool_.capacity(); }
  void set_slot_capacity(int c) { pool_.configure(c, vram_budget_); }
  void set_vram_budget(size_t b) {
    vram_budget_ = b;
    pool_.configure(pool_.capacity(), b);
  }
  int pinned_slots() const { return pool_.used(); }

 private:
  contracts::DeviceMesh mesh_;
  int n_experts_ = 0;
  int local_rank_ = 0;
  size_t vram_budget_ = 0;
  ExpertSlotPool pool_;
  std::unordered_map<uint64_t, std::pair<const void*, size_t>> host_bind_;

  static uint64_t key(contracts::ExpertId id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(id.layer)) << 32) |
           static_cast<uint32_t>(id.expert);
  }
};

class KvPolicyDevice final : public contracts::IKvDevicePolicy {
 public:
  bool kv_on_device() const override { return true; }
};

class ExecGpu final : public contracts::IExecBackend {
 public:
  ExecGpu(contracts::DeviceMesh mesh, int n_experts_hint);
  contracts::ExecMode mode() const override { return contracts::ExecMode::kPureGpu; }
  contracts::ExecCaps caps() const override;
  const contracts::DeviceMesh& mesh() const override { return mesh_; }
  void configure(const sched::PlacementPlan& plan) override;
  contracts::IGemm* gemm() override { return &gemm_; }
  contracts::IExpertRuntime* experts() override { return &experts_; }
  contracts::IKvDevicePolicy* kv_policy() override { return &kv_; }
  contracts::ICollective* coll() override { return coll_.get(); }
  ExpertRuntimeGpu& experts_gpu() { return experts_; }

 private:
  contracts::DeviceMesh mesh_;
  GemmGpu gemm_;
  ExpertRuntimeGpu experts_;
  KvPolicyDevice kv_;
  std::unique_ptr<contracts::ICollective> coll_;
  sched::PlacementPlan plan_{};
  size_t vram_budget_ = 0;
  size_t dram_budget_ = 0;
};

}  // namespace llmoc::exec::gpu
