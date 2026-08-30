#pragma once
#include "glm/glm_config.h"
#include "glm/hal/glm_awq_int4_ops.h"
#include "glm/hal/glm_nvfp4_ops.h"
#include "glm/weights/glm_format.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace llmoc::glm {

class GlmWeightStore {
 public:
  void open(const std::string& path, QuantKind expect);
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
  std::vector<uint8_t> blob_;
  std::unordered_map<std::string, size_t> index_;
};

bool write_glmq_file(const std::string& path, const GlmqFileHeader& hdr,
                     const std::vector<GlmqTensorRec>& catalog, const std::vector<uint8_t>& data);

}  // namespace llmoc::glm
