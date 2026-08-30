#pragma once
// llm-on-cpu :: sched/mode_controller.h
// M5: pure_cpu | hybrid_gpu | pure_gpu | auto

#include <string>

namespace llmoc::sched {

enum class ExecMode { kPureCpu, kHybridGpu, kPureGpu, kAuto };

inline ExecMode parse_mode(const std::string& s) {
  if (s == "hybrid_gpu") return ExecMode::kHybridGpu;
  if (s == "pure_gpu") return ExecMode::kPureGpu;
  if (s == "auto") return ExecMode::kAuto;
  return ExecMode::kPureCpu;
}

inline const char* mode_name(ExecMode m) {
  switch (m) {
    case ExecMode::kHybridGpu: return "hybrid_gpu";
    case ExecMode::kPureGpu: return "pure_gpu";
    case ExecMode::kAuto: return "auto";
    default: return "pure_cpu";
  }
}

// Resolve load-time mode. pure_gpu without CUDA → throws via out_error if provided.
// hybrid_gpu without CUDA → degrade to pure_cpu and set *degraded=true.
// auto → hybrid_gpu if CUDA else pure_cpu.
ExecMode resolve_mode(ExecMode requested, bool cuda_ok, bool* degraded = nullptr,
                      std::string* err = nullptr);

}  // namespace llmoc::sched
