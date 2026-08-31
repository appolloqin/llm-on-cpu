#pragma once
// llm-on-cpu :: contracts/exec_backend.h

#include <cstddef>
#include <memory>

#include "contracts/collective.h"
#include "contracts/device_mesh.h"
#include "contracts/exec_mode.h"
#include "contracts/expert_runtime.h"
#include "contracts/linear_op.h"

namespace llmoc::sched {
struct PlacementPlan;  // forward — configured after planner
}

namespace llmoc::contracts {

struct ExecCaps {
  bool experts_on_gpu = false;
  bool attn_on_gpu = false;
  int world_size = 1;
  int ep_size = 1;
  int tp_size = 1;
  size_t vram_budget_per_rank = 0;
  size_t dram_budget = 0;
};

// Optional KV placement hint (host vs device pages).
class IKvDevicePolicy {
 public:
  virtual ~IKvDevicePolicy() = default;
  virtual bool kv_on_device() const = 0;
};

class IExecBackend {
 public:
  virtual ~IExecBackend() = default;
  virtual ExecMode mode() const = 0;
  virtual ExecCaps caps() const = 0;
  virtual const DeviceMesh& mesh() const = 0;
  virtual void configure(const sched::PlacementPlan& plan) = 0;
  virtual IGemm* gemm() = 0;
  virtual IExpertRuntime* experts() = 0;
  virtual IKvDevicePolicy* kv_policy() = 0;
  virtual ICollective* coll() = 0;
};

}  // namespace llmoc::contracts
