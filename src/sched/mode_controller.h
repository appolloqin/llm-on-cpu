#pragma once
// llm-on-cpu :: sched/mode_controller.h
// Resolves pure_cpu | hybrid_gpu | pure_gpu | auto (+ device mesh G8)

#include <string>

#include "contracts/device_mesh.h"
#include "contracts/exec_mode.h"

namespace llmoc::sched {

using ExecMode = contracts::ExecMode;

inline ExecMode parse_mode(const std::string& s) { return contracts::parse_mode(s); }
inline const char* mode_name(ExecMode m) { return contracts::mode_name(m); }

// Resolve load-time mode. pure_gpu without CUDA → err set; still returns kPureGpu.
// hybrid_gpu without CUDA → degrade to pure_cpu and set *degraded=true.
// auto → hybrid_gpu if CUDA else pure_cpu.
ExecMode resolve_mode(ExecMode requested, bool cuda_ok, bool* degraded = nullptr,
                      std::string* err = nullptr);

// Build mesh for GPU modes. pure_cpu → forces world_size=1.
// visible_gpus: from cuda probe (0 if none). nccl_available: runtime NCCL probe.
bool resolve_mesh_for_mode(ExecMode mode, const contracts::DeviceMeshSpec& spec, int visible_gpus,
                           bool nccl_available, bool moe_family, contracts::DeviceMesh* out,
                           std::string* err);

}  // namespace llmoc::sched
