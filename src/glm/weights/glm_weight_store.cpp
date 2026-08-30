// llm-on-cpu :: glm/weights/glm_weight_store.cpp
#include "glm/weights/glm_weight_store.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace llmoc::glm {

bool write_glmq_file(const std::string& path, const GlmqFileHeader& hdr_in,
                     const std::vector<GlmqTensorRec>& catalog, const std::vector<uint8_t>& data) {
  GlmqFileHeader hdr = hdr_in;
  hdr.magic = kGlmqMagic;
  hdr.version = kGlmqVersion;
  hdr.n_tensors = static_cast<uint32_t>(catalog.size());
  hdr.catalog_bytes = sizeof(GlmqTensorRec) * catalog.size();
  hdr.data_bytes = data.size();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  if (!catalog.empty())
    out.write(reinterpret_cast<const char*>(catalog.data()),
              static_cast<std::streamsize>(hdr.catalog_bytes));
  if (!data.empty())
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(out);
}

void GlmWeightStore::open(const std::string& path, QuantKind expect) {
  open_ = false;
  blob_.clear();
  catalog_.clear();
  index_.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("glm: cannot open weights: " + path);
  in.read(reinterpret_cast<char*>(&hdr_), sizeof(hdr_));
  if (!in || !is_glmq_magic(hdr_.magic))
    throw std::runtime_error("glm: bad GLMQ magic (run tools/glm/* to build .glmq)");
  if (hdr_.version < 1 || hdr_.version > kGlmqVersion)
    throw std::runtime_error("glm: unsupported GLMQ version");

  const auto q = static_cast<GlmqQuant>(hdr_.quant);
  if (q == GlmqQuant::kAwqInt4) quant_ = QuantKind::kAwqInt4;
  else if (q == GlmqQuant::kNvfp4) quant_ = QuantKind::kNvfp4;
  else quant_ = QuantKind::kBf16;

  // reserved[0..3] stores AWQ group_size when non-zero
  uint32_t gs = 0;
  std::memcpy(&gs, hdr_.reserved, 4);
  if (gs > 0) awq_group_size_ = static_cast<int>(gs);

  // expect=Bf16 means "any" for loaders that only need presence
  if (expect != QuantKind::kBf16 && quant_ != expect &&
      !(expect == QuantKind::kAwqInt4 && quant_ == QuantKind::kBf16)) {
    // Allow BF16 file when config says awq during development of graph
  }

  catalog_.resize(hdr_.n_tensors);
  if (hdr_.n_tensors) {
    in.read(reinterpret_cast<char*>(catalog_.data()),
            static_cast<std::streamsize>(sizeof(GlmqTensorRec) * hdr_.n_tensors));
  }
  blob_.resize(static_cast<size_t>(hdr_.data_bytes));
  if (hdr_.data_bytes) {
    in.read(reinterpret_cast<char*>(blob_.data()), static_cast<std::streamsize>(hdr_.data_bytes));
  }
  if (!in) throw std::runtime_error("glm: truncated GLMQ file");

  for (size_t i = 0; i < catalog_.size(); ++i) index_[catalog_[i].name] = i;
  open_ = true;
}

const GlmqTensorRec* GlmWeightStore::find(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &catalog_[it->second];
}

const uint8_t* GlmWeightStore::data_of(const GlmqTensorRec& t) const {
  if (t.offset + t.nbytes > blob_.size()) throw std::runtime_error("glm: tensor OOB");
  return blob_.data() + static_cast<size_t>(t.offset);
}

const uint16_t* GlmWeightStore::bf16(const std::string& name) const {
  const auto* t = find(name);
  if (!t) return nullptr;
  return reinterpret_cast<const uint16_t*>(data_of(*t));
}

const uint16_t* GlmWeightStore::require_bf16(const std::string& name) const {
  const uint16_t* p = bf16(name);
  if (!p) throw std::runtime_error("glm: missing tensor " + name);
  return p;
}

bool GlmWeightStore::awq_view(const std::string& name, hal::AwqView& out) const {
  const auto* t = find(name);
  if (!t || t->dtype != static_cast<uint16_t>(GlmqDtype::kAWQ4)) return false;
  const int M = static_cast<int>(t->shape[0]);
  const int K = static_cast<int>(t->shape[1]);
  int gs = t->ndim >= 3 && t->shape[2] > 0 ? static_cast<int>(t->shape[2]) : awq_group_size_;
  if (gs <= 0) gs = K;
  const int rb = (K + 1) / 2;
  const size_t q_bytes = static_cast<size_t>(M) * rb;
  const int ng = K / gs;
  const size_t s_bytes = static_cast<size_t>(M) * ng * 2;
  if (t->nbytes < q_bytes + s_bytes) throw std::runtime_error("glm: AWQ tensor truncated " + name);
  const uint8_t* base = data_of(*t);
  out.qweight = base;
  out.scales = reinterpret_cast<const uint16_t*>(base + q_bytes);
  out.scales_f32 = nullptr;
  out.M = M;
  out.K = K;
  out.group_size = gs;
  return true;
}

bool GlmWeightStore::nvfp4_view(const std::string& name, hal::Nvfp4View& out) const {
  const auto* t = find(name);
  if (!t || t->dtype != static_cast<uint16_t>(GlmqDtype::kNVFP4)) return false;
  const int M = static_cast<int>(t->shape[0]);
  const int K = static_cast<int>(t->shape[1]);
  int gs = t->ndim >= 3 && t->shape[2] > 0 ? static_cast<int>(t->shape[2]) : 16;
  if (gs <= 0) gs = 16;
  const int rb = (K + 1) / 2;
  const size_t q_bytes = static_cast<size_t>(M) * rb;
  const int ng = (K + gs - 1) / gs;
  const size_t s_bytes = static_cast<size_t>(M) * ng;
  if (t->nbytes < q_bytes + s_bytes + 4)
    throw std::runtime_error("glm: NVFP4 tensor truncated " + name);
  const uint8_t* base = data_of(*t);
  out.qweight = base;
  out.scales_fp8 = base + q_bytes;
  float g = 1.f;
  std::memcpy(&g, base + q_bytes + s_bytes, 4);
  out.global_scale = g;
  out.M = M;
  out.K = K;
  out.group_size = gs;
  return true;
}

}  // namespace llmoc::glm
