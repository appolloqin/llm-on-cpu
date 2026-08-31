#pragma once
// llm-on-cpu :: contracts/weight_block.h

#include <cstdint>

namespace llmoc::contracts {

struct ExpertId {
  int layer = 0;
  int expert = 0;
  bool operator==(const ExpertId& o) const { return layer == o.layer && expert == o.expert; }
};

struct BlockHandle {
  ExpertId id;
  void* ptr = nullptr;   // host or device pointer (backend-defined)
  int device_ordinal = -1;
  uint64_t nbytes = 0;
  bool valid() const { return ptr != nullptr && nbytes > 0; }
};

}  // namespace llmoc::contracts
