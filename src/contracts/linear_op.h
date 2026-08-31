#pragma once
// llm-on-cpu :: contracts/linear_op.h

#include "contracts/tensor_view.h"

namespace llmoc::contracts {

class IGemm {
 public:
  virtual ~IGemm() = default;
  // y[M] = W[M,K] @ x[K]  (row-major W); returns false → caller may fall back
  virtual bool gemm_w16(const float* x, const void* W, float* y, int M, int K, bool is_f16) = 0;
};

}  // namespace llmoc::contracts
