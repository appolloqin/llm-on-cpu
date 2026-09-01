// llm-on-cpu :: weights/qlwc_store.cpp
#include "weights/qlwc_store.h"

#include <cstring>
#include <stdexcept>

#include "hal/cpu_ops.h"

namespace llmoc::qlwc {
namespace {

bool name_is_layer(const std::string& n) {
  // 仅语言模型层可换出；mtp.* / visual.* 常驻
  if (n.rfind("mtp.", 0) == 0) return false;
  if (n.rfind("visual.", 0) == 0) return false;
  if (n.find("language_model.layers.") != std::string::npos) return true;
  // bare "layers.N." but not mtp
  if (n.rfind("layers.", 0) == 0) return true;
  return false;
}

}  // namespace

void QlwcStore::open(const std::filesystem::path& file) { open(file, OpenOptions{}); }

void QlwcStore::open(const std::filesystem::path& file, OpenOptions opt) {
  close();
  file_ = file;
  lazy_ = opt.lazy;
  hdr_ = ReadHeader(file);
  if (!lazy_) {
    for (const auto& t : hdr_.tensors) load_tensor(t);
    return;
  }
  // lazy：目录已在 hdr_；常驻候选（非 layers.*）先装入
  for (const auto& t : hdr_.tensors) {
    if (!name_is_layer(t.name)) {
      always_resident_.insert(t.name);
      load_tensor(t);
    }
  }
}

void QlwcStore::close() {
  blobs_.clear();
  int4_views_.clear();
  pass_views_.clear();
  always_resident_.clear();
  loaded_bytes_ = 0;
  lazy_ = false;
  hdr_ = {};
  file_.clear();
}

bool QlwcStore::has(const std::string& name) const {
  return hdr_.find(name) != nullptr;
}

bool QlwcStore::is_int4(const std::string& name) const {
  const auto* t = hdr_.find(name);
  return t && t->kind == TensorKind::kInt4;
}

bool QlwcStore::is_loaded(const std::string& name) const {
  return int4_views_.count(name) != 0 || pass_views_.count(name) != 0;
}

void QlwcStore::mark_always_resident(const std::string& name) {
  always_resident_.insert(name);
  if (lazy_ && has(name) && !is_loaded(name)) ensure(name);
}

void QlwcStore::ensure(const std::string& name) {
  if (is_loaded(name)) return;
  const auto* t = hdr_.find(name);
  if (!t) throw std::runtime_error("QLWC missing tensor: " + name);
  load_tensor(*t);
}

void QlwcStore::load_tensor(const TensorMeta& t) {
  if (is_loaded(t.name)) return;
  if (t.kind == TensorKind::kPassthrough) {
    auto& buf = blobs_[t.name + "#data"];
    buf = ReadBlob(file_, t.data_offset, t.data_nbytes);
    loaded_bytes_ += buf.size();
    PassView v;
    v.data = reinterpret_cast<const uint16_t*>(buf.data());
    v.n_elem = t.data_nbytes / 2;
    v.dtype = t.pass_dtype;
    pass_views_.emplace(t.name, v);
  } else {
    auto& q = blobs_[t.name + "#q"];
    auto& s = blobs_[t.name + "#s"];
    q = ReadBlob(file_, t.q_offset, t.q_nbytes);
    s = ReadBlob(file_, t.scales_offset, t.scales_nbytes);
    loaded_bytes_ += q.size() + s.size();
    Int4View v;
    v.qweight = q.data();
    v.scales = reinterpret_cast<const uint16_t*>(s.data());
    if (t.zeros_nbytes) {
      auto& z = blobs_[t.name + "#z"];
      z = ReadBlob(file_, t.zeros_offset, t.zeros_nbytes);
      loaded_bytes_ += z.size();
      v.zeros = reinterpret_cast<const uint16_t*>(z.data());
    }
    v.M = static_cast<int>(t.shape[0]);
    v.K = static_cast<int>(t.shape[1]);
    v.group_size = static_cast<int>(t.group_size);
    v.scheme = hdr_.scheme;
    const int ng = (v.K + v.group_size - 1) / v.group_size;
    auto& sf = blobs_[t.name + "#sf"];
    sf.resize(static_cast<size_t>(v.M) * ng * sizeof(float));
    loaded_bytes_ += sf.size();
    auto* sf32 = reinterpret_cast<float*>(sf.data());
    const size_t nsc = static_cast<size_t>(v.M) * ng;
    for (size_t i = 0; i < nsc; ++i) sf32[i] = hal::f16_to_f32(v.scales[i]);
    v.scales_f32 = sf32;
    int4_views_.emplace(t.name, v);
  }
}

void QlwcStore::drop_tensor(const std::string& name) {
  if (always_resident_.count(name)) return;
  auto erase_blob = [&](const std::string& key) {
    auto it = blobs_.find(key);
    if (it == blobs_.end()) return;
    if (loaded_bytes_ >= it->second.size()) loaded_bytes_ -= it->second.size();
    else loaded_bytes_ = 0;
    blobs_.erase(it);
  };
  if (int4_views_.erase(name)) {
    erase_blob(name + "#q");
    erase_blob(name + "#s");
    erase_blob(name + "#z");
    erase_blob(name + "#sf");
  }
  if (pass_views_.erase(name)) erase_blob(name + "#data");
}

void QlwcStore::release_except(const std::unordered_set<std::string>& keep) {
  if (!lazy_) return;
  std::vector<std::string> drop;
  for (const auto& kv : int4_views_) {
    if (!keep.count(kv.first) && !always_resident_.count(kv.first)) drop.push_back(kv.first);
  }
  for (const auto& kv : pass_views_) {
    if (!keep.count(kv.first) && !always_resident_.count(kv.first)) drop.push_back(kv.first);
  }
  for (const auto& n : drop) drop_tensor(n);
}

Int4View QlwcStore::get_int4(const std::string& name) const {
  auto it = int4_views_.find(name);
  if (it == int4_views_.end()) {
    throw std::runtime_error("not int4 / not loaded (call ensure): " + name);
  }
  return it->second;
}

PassView QlwcStore::get_pass(const std::string& name) const {
  auto it = pass_views_.find(name);
  if (it == pass_views_.end()) {
    throw std::runtime_error("not passthrough / not loaded (call ensure): " + name);
  }
  return it->second;
}

}  // namespace llmoc::qlwc
