// llm-on-cpu :: glm/weights/glm_expert_prefetch.cpp
#include "glm/weights/glm_expert_prefetch.h"

namespace llmoc::glm {

void GlmExpertPrefetch::configure(const Config& cfg) { cfg_ = cfg; }

void GlmExpertPrefetch::plan(int, const std::vector<int>&) {}

void GlmExpertPrefetch::acquire(int, std::vector<const uint8_t*>& out_ptrs) {
  out_ptrs.clear();
}

void GlmExpertPrefetch::release(int) {}

}  // namespace llmoc::glm
