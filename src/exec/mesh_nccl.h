#pragma once
// llm-on-cpu :: exec/mesh_nccl.h — dynamic NCCL collective (single-node)

#include <memory>

#include "contracts/collective.h"
#include "contracts/device_mesh.h"

namespace llmoc::exec {

// Probe libnccl without initializing a communicator.
bool nccl_probe_load();

// world_size==1 → CollectiveNoop. Else NCCL if loaded, else host stub (tests).
std::unique_ptr<contracts::ICollective> make_collective(const contracts::DeviceMesh& mesh);

}  // namespace llmoc::exec
