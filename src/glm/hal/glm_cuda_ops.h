#pragma once
// llm-on-cpu :: glm/hal/glm_cuda_ops.h — wraps shared hal/cuda_backend

#include <cstdint>
#include <cstddef>
#include <string>

namespace llmoc::glm::hal {

class GlmCudaContext {
 public:
  static GlmCudaContext& instance();
  bool init(size_t vram_budget_bytes);
  void shutdown();
  bool available() const { return ok_; }
  const char* status() const { return status_.c_str(); }
  size_t vram_used() const;
  size_t vram_budget() const { return vram_budget_; }

  int upload_bf16(const std::string& key, const uint16_t* W, int M, int K);
  bool gemm_bf16(const std::string& key, const float* x, const uint16_t* W_host, float* y, int M,
                 int K);

 private:
  GlmCudaContext() = default;
  bool ok_ = false;
  std::string status_ = "uninitialized";
  size_t vram_budget_ = 0;
};

bool cuda_ops_available();
void log_cuda_status();
bool gemm_bf16_cuda(const float* x, const uint16_t* W, float* y, int M, int K);

}  // namespace llmoc::glm::hal
