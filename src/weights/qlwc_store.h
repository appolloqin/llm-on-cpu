#pragma once
// llm-on-cpu :: weights/qlwc_store.h
// INT4 QLWC 权重仓库（独立于 WeightManager）。

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "weights/qlwc_format.h"

namespace llmoc::qlwc {

struct Int4View {
  const uint8_t* qweight = nullptr;  // packed
  const uint16_t* scales = nullptr;  // fp16 bits
  const uint16_t* zeros = nullptr;   // fp16 bits or null (AWQ)
  // open 时预转 f32（M * ceil(K/gs)）；避免每次 GEMM 重转（尤其 lm_head）
  const float* scales_f32 = nullptr;
  int M = 0, K = 0;
  int group_size = 128;
  Scheme scheme = Scheme::kAwqSym;
};

struct PassView {
  const uint16_t* data = nullptr;
  size_t n_elem = 0;
  PassDtype dtype = PassDtype::kBF16;
};

class QlwcStore {
 public:
  void open(const std::filesystem::path& file);
  void close();

  const Header& header() const { return hdr_; }
  bool has(const std::string& name) const {
    return int4_views_.count(name) != 0 || pass_views_.count(name) != 0;
  }
  bool is_int4(const std::string& name) const { return int4_views_.count(name) != 0; }
  Int4View get_int4(const std::string& name) const;
  PassView get_pass(const std::string& name) const;

 private:
  Header hdr_;
  std::filesystem::path file_;
  // 全量驻留以降低 IO（INT4 体积小）
  std::unordered_map<std::string, std::vector<uint8_t>> blobs_;
  // open 时建好视图，避免 decode 热路径反复 find/字符串拼接
  std::unordered_map<std::string, Int4View> int4_views_;
  std::unordered_map<std::string, PassView> pass_views_;
};

}  // namespace llmoc::qlwc
