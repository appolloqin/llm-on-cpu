#pragma once
// llm-on-cpu :: contracts/tensor_view.h

#include <cstddef>
#include <cstdint>

namespace llmoc::contracts {

enum class DType : uint8_t { kF32, kBF16, kF16, kInt4, kNvfp4 };

struct TensorView {
  void* ptr = nullptr;
  int device_ordinal = -1;  // -1 = host
  DType dtype = DType::kF32;
  int64_t numel = 0;
  int rank = 0;
  int64_t shape[8]{};

  bool on_device() const { return device_ordinal >= 0; }
  size_t nbytes() const {
    size_t b = 4;
    if (dtype == DType::kBF16 || dtype == DType::kF16) b = 2;
    else if (dtype == DType::kInt4 || dtype == DType::kNvfp4) b = 1;  // packed approx
    return static_cast<size_t>(numel) * b;
  }
};

}  // namespace llmoc::contracts
