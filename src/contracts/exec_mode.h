#pragma once
// llm-on-cpu :: contracts/exec_mode.h — shared exec mode enum (IMPLEMENTATION 2026-08-31)
// + layer_stream（DESIGN_LAYER_STREAM：装不下时按层加载，可运行优先）

#include <string>

namespace llmoc::contracts {

enum class ExecMode { kPureCpu, kHybridGpu, kPureGpu, kAuto, kLayerStream };

inline ExecMode parse_mode(const std::string& s) {
  if (s == "hybrid_gpu") return ExecMode::kHybridGpu;
  if (s == "pure_gpu") return ExecMode::kPureGpu;
  if (s == "auto") return ExecMode::kAuto;
  if (s == "layer_stream") return ExecMode::kLayerStream;
  return ExecMode::kPureCpu;
}

inline const char* mode_name(ExecMode m) {
  switch (m) {
    case ExecMode::kHybridGpu: return "hybrid_gpu";
    case ExecMode::kPureGpu: return "pure_gpu";
    case ExecMode::kAuto: return "auto";
    case ExecMode::kLayerStream: return "layer_stream";
    default: return "pure_cpu";
  }
}

}  // namespace llmoc::contracts
