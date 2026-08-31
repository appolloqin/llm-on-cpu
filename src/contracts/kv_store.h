#pragma once
// llm-on-cpu :: contracts/kv_store.h — family-specific stores implement this

#include <cstdint>
#include <vector>

namespace llmoc::contracts {

struct TokenSpan {
  const int32_t* data = nullptr;
  int n = 0;
};

class IKvStore {
 public:
  virtual ~IKvStore() = default;
  virtual void clear() = 0;
  virtual int prefix_reuse(const TokenSpan& tokens) = 0;
};

}  // namespace llmoc::contracts
