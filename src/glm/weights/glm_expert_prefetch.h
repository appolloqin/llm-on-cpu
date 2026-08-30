#pragma once
// llm-on-cpu :: glm/weights/glm_expert_prefetch.h — 独立专家预取（不依赖 weights/prefetch_pipeline）

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmoc::glm {

class GlmExpertPrefetch {
 public:
  struct Config {
    size_t slot_bytes = 64ull << 20;
    unsigned slots = 4;
  };

  void configure(const Config& cfg);
  // 占位：后续接 GLMQ expert groups + 异步 IO
  void plan(int layer, const std::vector<int>& expert_ids);
  void acquire(int layer, std::vector<const uint8_t*>& out_ptrs);
  void release(int layer);

 private:
  Config cfg_;
  std::mutex mu_;
};

}  // namespace llmoc::glm
