// llm-on-cpu :: exec/gpu/exec_gpu.cpp
#include "exec/gpu/exec_gpu.h"

#include "exec/mesh_nccl.h"
#include "hal/cpu_ops.h"
#include "hal/cuda_backend.h"

namespace llmoc::exec::gpu {

bool GemmGpu::gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) {
  if (llmoc::hal::cuda::enabled() &&
      llmoc::hal::cuda::try_gemm_w16(x, reinterpret_cast<const uint16_t*>(W), y, M, K, is_f16)) {
    return true;
  }
  const auto dt = is_f16 ? llmoc::hal::WDtype::kF16 : llmoc::hal::WDtype::kBF16;
  llmoc::hal::gemm_bias_free(x, reinterpret_cast<const uint16_t*>(W), y, M, K, dt);
  return true;
}

ExpertRuntimeGpu::ExpertRuntimeGpu(contracts::DeviceMesh mesh, int n_experts)
    : mesh_(std::move(mesh)), n_experts_(n_experts) {
  pool_.configure(8, 0);
}

bool ExpertRuntimeGpu::owns(contracts::ExpertId id) const {
  if (mesh_.ep_size <= 1) return true;
  return mesh_.owns_expert(local_rank_ % mesh_.ep_size, id.expert, n_experts_);
}

void ExpertRuntimeGpu::bind_host(contracts::ExpertId id, const void* host, size_t bytes) {
  host_bind_[key(id)] = {host, bytes};
}

void ExpertRuntimeGpu::prefetch(int /*layer*/, const contracts::ExpertId* ids, int n) {
  for (int i = 0; i < n; ++i) {
    if (!owns(ids[i])) continue;
    auto it = host_bind_.find(key(ids[i]));
    if (it == host_bind_.end()) continue;
    pool_.pin_upload(ids[i], it->second.first, it->second.second);
  }
}

void ExpertRuntimeGpu::pin(int layer, const contracts::ExpertId* ids, int n,
                           contracts::BlockHandle* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = {};
    out[i].id = ids ? ids[i] : contracts::ExpertId{layer, i};
    if (!owns(out[i].id)) continue;
    auto it = host_bind_.find(key(out[i].id));
    void* d = nullptr;
    size_t nbytes = 0;
    if (it != host_bind_.end()) {
      nbytes = it->second.second;
      d = pool_.pin_upload(out[i].id, it->second.first, nbytes);
    }
    if (!d) d = pool_.lookup(out[i].id);
    out[i].ptr = d;
    out[i].nbytes = d ? (nbytes ? nbytes : 1) : 0;
    out[i].device_ordinal = d ? mesh_.device_at(local_rank_) : -1;
  }
}

void ExpertRuntimeGpu::release(const contracts::BlockHandle* hs, int n) {
  for (int i = 0; i < n; ++i) {
    if (hs[i].valid()) pool_.release(hs[i].id);
  }
}

ExecGpu::ExecGpu(contracts::DeviceMesh mesh, int n_experts_hint)
    : mesh_(std::move(mesh)),
      experts_(mesh_, n_experts_hint),
      coll_(make_collective(mesh_)) {}

contracts::ExecCaps ExecGpu::caps() const {
  contracts::ExecCaps c;
  c.experts_on_gpu = true;
  c.attn_on_gpu = true;
  c.world_size = mesh_.world_size;
  c.ep_size = mesh_.ep_size;
  c.tp_size = mesh_.tp_size;
  c.vram_budget_per_rank = vram_budget_;
  c.dram_budget = dram_budget_;
  return c;
}

void ExecGpu::configure(const sched::PlacementPlan& plan) {
  plan_ = plan;
  vram_budget_ = plan.vram_budget_per_rank > 0 ? plan.vram_budget_per_rank : plan.vram_budget;
  dram_budget_ = plan.dram_budget;
  experts_.set_vram_budget(vram_budget_ / 2);  // half for expert slots
  if (plan.expert_slots_per_rank > 0) {
    experts_.set_slot_capacity(static_cast<int>(plan.expert_slots_per_rank));
  }
  if (plan.n_experts > 0) experts_.set_n_experts(plan.n_experts);
}

}  // namespace llmoc::exec::gpu
