// llm-on-cpu :: glm/hal/glm_cuda_ops.cpp — thin wrapper over shared hal/cuda_backend
#include "glm/hal/glm_cuda_ops.h"

#include "common/log.h"
#include "hal/cuda_backend.h"

namespace llmoc::glm::hal {

GlmCudaContext& GlmCudaContext::instance() {
  static GlmCudaContext ctx;
  return ctx;
}

bool GlmCudaContext::init(size_t vram_budget_bytes) {
  ok_ = llmoc::hal::cuda::enable(vram_budget_bytes);
  status_ = llmoc::hal::cuda::status();
  vram_budget_ = vram_budget_bytes;
  return ok_;
}

void GlmCudaContext::shutdown() {
  llmoc::hal::cuda::disable();
  ok_ = false;
  status_ = "shutdown";
}

size_t GlmCudaContext::vram_used() const { return llmoc::hal::cuda::vram_used(); }

int GlmCudaContext::upload_bf16(const std::string& /*key*/, const uint16_t* W, int M, int K) {
  if (!ok_ || !W) return -1;
  return llmoc::hal::cuda::prefetch_w16(W, M, K, false) ? 1 : -1;
}

bool GlmCudaContext::gemm_bf16(const std::string& /*key*/, const float* x, const uint16_t* W_host,
                               float* y, int M, int K) {
  if (!ok_) return false;
  return llmoc::hal::cuda::try_gemm_w16(x, W_host, y, M, K, false);
}

bool cuda_ops_available() { return llmoc::hal::cuda::probe_available(); }

void log_cuda_status() {
  LOG_INFO("glm cuda: %s", llmoc::hal::cuda::status());
  llmoc::hal::cuda::log_status();
}

bool gemm_bf16_cuda(const float* x, const uint16_t* W, float* y, int M, int K) {
  return llmoc::hal::cuda::try_gemm_w16(x, W, y, M, K, false);
}

}  // namespace llmoc::glm::hal
