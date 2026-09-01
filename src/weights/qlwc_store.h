#pragma once
// llm-on-cpu :: weights/qlwc_store.h
// INT4 QLWC 权重仓库（独立于 WeightManager）。
// lazy=true：仅读目录，按 ensure/release 做层窗口（layer_stream）。

#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "weights/qlwc_format.h"

namespace llmoc::qlwc {

struct Int4View {
  const uint8_t* qweight = nullptr;
  const uint16_t* scales = nullptr;
  const uint16_t* zeros = nullptr;
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

struct OpenOptions {
  bool lazy = false;  // false = 全量驻留（默认，兼容旧路径）
};

class QlwcStore {
 public:
  void open(const std::filesystem::path& file);
  void open(const std::filesystem::path& file, OpenOptions opt);
  void close();

  const Header& header() const { return hdr_; }
  bool lazy() const { return lazy_; }
  bool has(const std::string& name) const;
  bool is_int4(const std::string& name) const;
  bool is_loaded(const std::string& name) const;
  Int4View get_int4(const std::string& name) const;
  PassView get_pass(const std::string& name) const;

  // lazy：装入张量（已装则命中）；非 lazy 为 no-op
  void ensure(const std::string& name);
  // 释放不在 keep 集合内、且非 always_resident 的已装张量
  void release_except(const std::unordered_set<std::string>& keep);
  void mark_always_resident(const std::string& name);
  uint64_t loaded_bytes() const { return loaded_bytes_; }

 private:
  void load_tensor(const TensorMeta& t);
  void drop_tensor(const std::string& name);

  Header hdr_;
  std::filesystem::path file_;
  bool lazy_ = false;
  uint64_t loaded_bytes_ = 0;
  std::unordered_set<std::string> always_resident_;
  std::unordered_map<std::string, std::vector<uint8_t>> blobs_;
  std::unordered_map<std::string, Int4View> int4_views_;
  std::unordered_map<std::string, PassView> pass_views_;
};

}  // namespace llmoc::qlwc
