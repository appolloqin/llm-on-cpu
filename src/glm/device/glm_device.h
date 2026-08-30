#pragma once
// llm-on-cpu :: glm/device/glm_device.h
// 三种模式设备抽象：pure_cpu / hybrid_gpu / pure_gpu（CUDA 可选编译）

#include <cstddef>
#include <memory>
#include <string>

#include "glm/glm_config.h"

namespace llmoc::glm {

struct DeviceCaps {
  bool has_cuda = false;
  size_t cpu_ram_hint_bytes = 0;
  size_t gpu_vram_hint_bytes = 0;
  std::string backend_name;  // "cpu" | "cuda" | "hybrid"
};

class IGlmDevice {
 public:
  virtual ~IGlmDevice() = default;
  virtual const DeviceCaps& caps() const = 0;
  virtual const char* name() const = 0;
  // 将专家/层权重安置策略：CPU 常驻、GPU 常驻、或分层
  virtual void configure_memory(double dram_hot_gb, double gpu_vram_gb) = 0;
};

// 按配置解析模式；无 CUDA 时 pure_gpu 抛错，hybrid_gpu 降级 pure_cpu 并打日志由调用方处理。
std::unique_ptr<IGlmDevice> make_device(ExecMode mode, bool* degraded_to_cpu);

bool cuda_runtime_available();

}  // namespace llmoc::glm
