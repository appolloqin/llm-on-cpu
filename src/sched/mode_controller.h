#pragma once
// llm-on-cpu :: sched/mode_controller.h
// M4/M5 模式开关占位: pure_cpu | hybrid_gpu | pure_gpu

#include <string>

namespace llmoc::sched {

enum class ExecMode { kPureCpu, kHybridGpu, kPureGpu, kAuto };

inline ExecMode parse_mode(const std::string& s) {
  if (s == "hybrid_gpu") return ExecMode::kHybridGpu;
  if (s == "pure_gpu") return ExecMode::kPureGpu;
  if (s == "auto") return ExecMode::kAuto;
  return ExecMode::kPureCpu;
}

inline ExecMode resolve_mode(ExecMode m) {
  // B/C 期仅交付 pure_cpu; auto → pure_cpu
  if (m == ExecMode::kAuto) return ExecMode::kPureCpu;
  return m;
}

}  // namespace llmoc::sched
