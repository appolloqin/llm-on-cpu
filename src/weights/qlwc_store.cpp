// llm-on-cpu :: weights/qlwc_store.cpp
#include "weights/qlwc_store.h"

#include <cstring>
#include <stdexcept>

#include "hal/cpu_ops.h"

namespace llmoc::qlwc {

void QlwcStore::open(const std::filesystem::path& file) {
  file_ = file;
  hdr_ = ReadHeader(file);
  blobs_.clear();
  int4_views_.clear();
  pass_views_.clear();
  for (const auto& t : hdr_.tensors) {
    if (t.kind == TensorKind::kPassthrough) {
      auto& buf = blobs_[t.name + "#data"];
      buf = ReadBlob(file, t.data_offset, t.data_nbytes);
      PassView v;
      v.data = reinterpret_cast<const uint16_t*>(buf.data());
      v.n_elem = t.data_nbytes / 2;
      v.dtype = t.pass_dtype;
      pass_views_.emplace(t.name, v);
    } else {
      auto& q = blobs_[t.name + "#q"];
      auto& s = blobs_[t.name + "#s"];
      q = ReadBlob(file, t.q_offset, t.q_nbytes);
      s = ReadBlob(file, t.scales_offset, t.scales_nbytes);
      Int4View v;
      v.qweight = q.data();
      v.scales = reinterpret_cast<const uint16_t*>(s.data());
      if (t.zeros_nbytes) {
        auto& z = blobs_[t.name + "#z"];
        z = ReadBlob(file, t.zeros_offset, t.zeros_nbytes);
        v.zeros = reinterpret_cast<const uint16_t*>(z.data());
      }
      v.M = static_cast<int>(t.shape[0]);
      v.K = static_cast<int>(t.shape[1]);
      v.group_size = static_cast<int>(t.group_size);
      v.scheme = hdr_.scheme;
      const int ng = (v.K + v.group_size - 1) / v.group_size;
      auto& sf = blobs_[t.name + "#sf"];
      sf.resize(static_cast<size_t>(v.M) * ng * sizeof(float));
      auto* sf32 = reinterpret_cast<float*>(sf.data());
      const size_t nsc = static_cast<size_t>(v.M) * ng;
      for (size_t i = 0; i < nsc; ++i) sf32[i] = hal::f16_to_f32(v.scales[i]);
      v.scales_f32 = sf32;
      int4_views_.emplace(t.name, v);
    }
  }
}

void QlwcStore::close() {
  blobs_.clear();
  int4_views_.clear();
  pass_views_.clear();
  hdr_ = {};
}

Int4View QlwcStore::get_int4(const std::string& name) const {
  auto it = int4_views_.find(name);
  if (it == int4_views_.end()) throw std::runtime_error("not int4 tensor: " + name);
  return it->second;
}

PassView QlwcStore::get_pass(const std::string& name) const {
  auto it = pass_views_.find(name);
  if (it == pass_views_.end()) throw std::runtime_error("not passthrough tensor: " + name);
  return it->second;
}

}  // namespace llmoc::qlwc
