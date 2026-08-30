// llm-on-cpu :: glm/device/glm_device.cpp
#include "glm/device/glm_device.h"

#include <stdexcept>

#include "common/log.h"
#include "glm/hal/glm_cuda_ops.h"
#include "hal/cuda_backend.h"

namespace llmoc::glm {
namespace {

class CpuDevice final : public IGlmDevice {
 public:
  explicit CpuDevice(const char* nm) {
    caps_.has_cuda = false;
    caps_.backend_name = nm;
  }
  const DeviceCaps& caps() const override { return caps_; }
  const char* name() const override { return caps_.backend_name.c_str(); }
  void configure_memory(double dram_hot_gb, double) override {
    caps_.cpu_ram_hint_bytes = static_cast<size_t>(dram_hot_gb * (1ull << 30));
    caps_.gpu_vram_hint_bytes = 0;
  }

 private:
  DeviceCaps caps_;
};

class CudaDevice final : public IGlmDevice {
 public:
  CudaDevice(bool hybrid) {
    caps_.has_cuda = true;
    caps_.backend_name = hybrid ? "hybrid" : "cuda";
  }
  const DeviceCaps& caps() const override { return caps_; }
  const char* name() const override { return caps_.backend_name.c_str(); }
  void configure_memory(double dram_hot_gb, double gpu_vram_gb) override {
    caps_.cpu_ram_hint_bytes = static_cast<size_t>(dram_hot_gb * (1ull << 30));
    caps_.gpu_vram_hint_bytes = static_cast<size_t>(gpu_vram_gb * (1ull << 30));
    hal::GlmCudaContext::instance().init(caps_.gpu_vram_hint_bytes);
  }

 private:
  DeviceCaps caps_;
};

}  // namespace

bool cuda_runtime_available() {
  // Probe only — must not enable GEMM (pure_cpu / degrade paths stay CPU).
  return llmoc::hal::cuda::probe_available();
}

std::unique_ptr<IGlmDevice> make_device(ExecMode mode, bool* degraded_to_cpu) {
  if (degraded_to_cpu) *degraded_to_cpu = false;
  const bool cuda_ok = cuda_runtime_available();

  if (mode == ExecMode::kPureCpu) {
    return std::make_unique<CpuDevice>("cpu");
  }
  if (mode == ExecMode::kPureGpu) {
    if (!cuda_ok) {
      throw std::runtime_error(
          "glm: mode=pure_gpu requires CUDA Runtime + cuBLAS (cudart64_12/cublas64_12); "
          "use pure_cpu or hybrid_gpu");
    }
    auto d = std::make_unique<CudaDevice>(false);
    hal::log_cuda_status();
    return d;
  }
  // hybrid_gpu
  if (!cuda_ok) {
    if (degraded_to_cpu) *degraded_to_cpu = true;
    LOG_WARN("glm: hybrid_gpu requested but CUDA unavailable — degraded to pure_cpu (%s)",
             hal::GlmCudaContext::instance().status());
    return std::make_unique<CpuDevice>("cpu");
  }
  auto d = std::make_unique<CudaDevice>(true);
  hal::log_cuda_status();
  return d;
}

}  // namespace llmoc::glm
