// llm-on-cpu :: exec/nccl_probe.cpp
#include "exec/nccl_probe.h"

#include "exec/mesh_nccl.h"

namespace llmoc::exec {

bool nccl_available() { return nccl_probe_load(); }

}  // namespace llmoc::exec
