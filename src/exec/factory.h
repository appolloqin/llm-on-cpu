#pragma once
// llm-on-cpu :: exec/factory.h — make_exec(mode, mesh, ...)

#include <memory>
#include <string>

#include "contracts/device_mesh.h"
#include "contracts/exec_backend.h"
#include "contracts/exec_mode.h"

namespace llmoc::exec {

struct MakeExecOptions {
  contracts::DeviceMesh mesh;
  size_t vram_budget_per_rank = 0;
  int n_experts_hint = 0;
};

// Returns backend for resolved mode. Does not enable CUDA (caller does).
std::unique_ptr<contracts::IExecBackend> make_exec(contracts::ExecMode mode,
                                                   const MakeExecOptions& opt,
                                                   std::string* err = nullptr);

}  // namespace llmoc::exec
