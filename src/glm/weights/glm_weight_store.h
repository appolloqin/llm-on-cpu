#pragma once
#include "glm/glm_config.h"
#include "glm/hal/glm_awq_int4_ops.h"
#include "glm/hal/glm_nvfp4_ops.h"
#include "glm/weights/glm_format.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmoc::glm {

class GlmWeightStore {
 public:
  GlmWeightStore() = default;
  ~GlmWeightStore() { close(); }
  GlmWeightStore(const GlmWeightStore&) = delete;
  GlmWeightStore& operator=(const GlmWeightStore&) = delete;

  void open(const std::string& path, QuantKind expect);
  void close();
  bool is_open() const { return open_; }
  const GlmqFileHeader& header() const { return hdr_; }
  QuantKind quant() const { return quant_; }
  int awq_group_size() const { return awq_group_size_; }

  const GlmqTensorRec* find(const std::string& name) const;
  const uint8_t* data_of(const GlmqTensorRec& t) const;
  const uint16_t* bf16(const std::string& name) const;
  const uint16_t* require_bf16(const std::string& name) const;

  bool awq_view(const std::string& name, hal::AwqView& out) const;
  bool nvfp4_view(const std::string& name, hal::Nvfp4View& out) const;

 private:
  bool open_ = false;
  GlmqFileHeader hdr_{};
  QuantKind quant_ = QuantKind::kBf16;
  int awq_group_size_ = 128;
  std::vector<GlmqTensorRec> catalog_;
  // Payload view (mmap or owned buffer). Tensor offsets are relative to data_base_.
  const uint8_t* data_base_ = nullptr;
  size_t data_bytes_ = 0;
  // Full-file map (preferred for large NVFP4/AWQ packs — no RAM copy).
  void* map_view_ = nullptr;
  size_t map_bytes_ = 0;
#if defined(_WIN32)
  void* win_file_ = nullptr;   // HANDLE
  void* win_mapping_ = nullptr;
#else
  int map_fd_ = -1;
#endif
  // Fallback when mmap unavailable: small files / tests.
  std::vector<uint8_t> owned_blob_;
  std::unordered_map<std::string, size_t> index_;
};

bool write_glmq_file(const std::string& path, const GlmqFileHeader& hdr,
                     const std::vector<GlmqTensorRec>& catalog, const std::vector<uint8_t>& data);

}  // namespace llmoc::glm
