#pragma once
// llm-on-cpu :: exec/nccl_probe.h — dynamic NCCL availability (stub until mesh_nccl)

namespace llmoc::exec {

// Returns true if libnccl can be loaded. Currently false until mesh_nccl lands;
// world_size==1 never needs it.
bool nccl_available();

}  // namespace llmoc::exec
