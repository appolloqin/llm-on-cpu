#pragma once
// llm-on-cpu :: glm/weights/glm_format.h
// 独立容器魔数，不与 LWC/QLWC 共用。

#include <cstdint>
#include <cstring>

namespace llmoc::glm {

constexpr uint32_t kGlmqMagic = 0x514D4C47u;  // "GLMQ"
constexpr uint16_t kGlmqVersion = 2;

enum class GlmqQuant : uint16_t {
  kAwqInt4 = 1,
  kNvfp4 = 2,
  kBf16 = 3,
};

enum class GlmqDtype : uint16_t {
  kBF16 = 1,
  kF32 = 2,
  kAWQ4 = 3,
  kNVFP4 = 4,
};

#pragma pack(push, 1)
struct GlmqFileHeader {
  uint32_t magic = kGlmqMagic;
  uint16_t version = kGlmqVersion;
  uint16_t quant = static_cast<uint16_t>(GlmqQuant::kBf16);
  uint32_t n_tensors = 0;
  uint32_t n_expert_groups = 0;
  uint64_t catalog_bytes = 0;
  uint64_t data_bytes = 0;
  uint32_t hidden = 0;
  uint32_t layers = 0;
  uint32_t vocab = 0;
  uint32_t n_heads = 0;
  uint32_t n_kv = 0;
  uint32_t head_dim = 0;
  uint32_t n_experts = 0;
  uint32_t topk = 0;
  uint32_t moe_inter = 0;
  uint32_t rms_eps_bits = 0;
  char reserved[16] = {};
};

struct GlmqTensorRec {
  char name[96] = {};
  uint16_t dtype = static_cast<uint16_t>(GlmqDtype::kBF16);
  uint16_t ndim = 0;
  uint32_t shape[4] = {};
  uint64_t offset = 0;
  uint64_t nbytes = 0;
};
#pragma pack(pop)

inline bool is_glmq_magic(uint32_t m) { return m == kGlmqMagic; }

inline float rms_eps_from_hdr(const GlmqFileHeader& h) {
  float e = 1e-6f;
  if (h.rms_eps_bits) std::memcpy(&e, &h.rms_eps_bits, 4);
  return e;
}

inline void set_rms_eps(GlmqFileHeader& h, float e) {
  std::memcpy(&h.rms_eps_bits, &e, 4);
}

}  // namespace llmoc::glm
