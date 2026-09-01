// llm-on-cpu :: weights/layer_stream.cpp
#include "weights/layer_stream.h"

#include <chrono>
#include <cstring>
#include <regex>
#include <stdexcept>

#include "common/log.h"
#include "hal/cuda_backend.h"
#include "weights/lwc_format.h"

namespace llmoc::wt {
namespace {

using Clock = std::chrono::steady_clock;

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool is_qlwc_path(const std::string& p) {
  return ends_with(p, ".qlwc") || ends_with(p, ".QLWC");
}

std::string layer_key(const std::string& prefix, int L) {
  return prefix + std::to_string(L) + ".";
}

}  // namespace

void infer_layer_layout(const std::vector<std::string>& names, std::string* prefix_out,
                        int* n_layers_out) {
  std::string best_prefix = "layers.";
  int max_l = -1;
  std::regex re(R"((?:^|.*\.)layers\.(\d+)\.)");
  for (const auto& n : names) {
    std::smatch m;
    if (!std::regex_search(n, m, re)) continue;
    const int L = std::stoi(m[1].str());
    if (L > max_l) max_l = L;
    const auto pos = n.find("layers.");
    if (pos != std::string::npos) {
      const std::string p = n.substr(0, pos + 7);  // include "layers."
      if (p.find("language_model.") != std::string::npos) best_prefix = "language_model.layers.";
      else if (best_prefix != "language_model.layers.") best_prefix = "layers.";
    }
  }
  if (prefix_out) *prefix_out = best_prefix;
  if (n_layers_out) *n_layers_out = max_l + 1;
}

class LayerStreamLoader final : public ILayerStreamLoader {
 public:
  void open(const std::string& weight_path, const LayerStreamConfig& cfg) override {
    close();
    cfg_ = cfg;
    path_ = weight_path;
    if (cfg_.window_layers < 1) cfg_.window_layers = 1;
    engine_ = io::make_engine(cfg_.io_workers);
    use_cuda_ = (cfg_.device.rfind("cuda", 0) == 0);

    if (is_qlwc_path(weight_path)) {
      qlwc_ = std::make_unique<qlwc::QlwcStore>();
      qlwc::OpenOptions o;
      o.lazy = true;
      qlwc_->open(weight_path, o);
      std::vector<std::string> names;
      for (const auto& t : qlwc_->header().tensors) names.push_back(t.name);
      if (cfg_.layer_prefix.empty() || cfg_.n_layers <= 0) {
        std::string pref;
        int nl = 0;
        infer_layer_layout(names, &pref, &nl);
        if (cfg_.layer_prefix.empty()) cfg_.layer_prefix = pref;
        if (cfg_.n_layers <= 0) cfg_.n_layers = nl;
      }
      LOG_INFO("layer_stream: QLWC lazy path=%s layers=%d prefix=%s window=%d device=%s",
               weight_path.c_str(), cfg_.n_layers, cfg_.layer_prefix.c_str(), cfg_.window_layers,
               cfg_.device.c_str());
      return;
    }

    // LWC：目录 + 按需 ReadBlob，常驻非 layers
    lwc_hdr_ = lwc::ReadHeader(weight_path);
    std::vector<std::string> names;
    for (const auto& t : lwc_hdr_.tensors) names.push_back(t.name);
    if (cfg_.layer_prefix.empty() || cfg_.n_layers <= 0) {
      std::string pref;
      int nl = 0;
      infer_layer_layout(names, &pref, &nl);
      if (cfg_.layer_prefix.empty()) cfg_.layer_prefix = pref;
      if (cfg_.n_layers <= 0) cfg_.n_layers = nl;
    }
    for (const auto& t : lwc_hdr_.tensors) {
      if (t.name.find("layers.") == std::string::npos) ensure_lwc_tensor(t.name);
    }
    LOG_INFO("layer_stream: LWC path=%s layers=%d prefix=%s window=%d device=%s",
             weight_path.c_str(), cfg_.n_layers, cfg_.layer_prefix.c_str(), cfg_.window_layers,
             cfg_.device.c_str());
  }

  void close() override {
    drain_prefetch();
    clear_cuda();
    if (qlwc_) {
      qlwc_->close();
      qlwc_.reset();
    }
    lwc_blobs_.clear();
    lwc_hdr_ = {};
    pinned_.clear();
    st_ = {};
    engine_.reset();
  }

  LayerView pin_layer(int layer_id) override {
    if (layer_id < 0 || (cfg_.n_layers > 0 && layer_id >= cfg_.n_layers))
      throw std::runtime_error("pin_layer: bad layer_id");
    const auto t0 = Clock::now();
    drain_prefetch();
    LayerView view;
    view.layer_id = layer_id;
    const std::string key = layer_key(cfg_.layer_prefix, layer_id);
    if (pinned_.count(layer_id)) {
      ++st_.pin_hits;
      view.tensor_names = list_layer_tensors(layer_id);
      view.bytes = layer_bytes(layer_id);
      return view;
    }
    ++st_.pin_loads;
    for (const auto& name : list_layer_tensors(layer_id)) {
      if (qlwc_) qlwc_->ensure(name);
      else ensure_lwc_tensor(name);
      view.tensor_names.push_back(name);
    }
    view.bytes = layer_bytes(layer_id);
    if (use_cuda_ && llmoc::hal::cuda::enabled()) upload_layer_cuda(layer_id, view.tensor_names);
    pinned_.insert(layer_id);
    enforce_window();
    st_.window_bytes = current_window_bytes();
    if (cfg_.max_window_bytes && st_.window_bytes > cfg_.max_window_bytes) {
      LOG_WARN("layer_stream: window %llu bytes > max_window_bytes %llu",
               static_cast<unsigned long long>(st_.window_bytes),
               static_cast<unsigned long long>(cfg_.max_window_bytes));
    }
    st_.io_wait_ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
    return view;
  }

  void prefetch_layer(int layer_id) override {
    if (layer_id < 0 || (cfg_.n_layers > 0 && layer_id >= cfg_.n_layers)) return;
    if (pinned_.count(layer_id) || prefetching_ == layer_id) return;
    ++st_.prefetch_issued;
    prefetching_ = layer_id;
    // 同步预读进 store（IoEngine 异步对多 blob 编排复杂；层粒度同步仍可叠在上一层 compute）
    // 真正重叠：在独立线程提交 — 此处用 engine_ drain 栅栏兼容
    if (!engine_) {
      for (const auto& name : list_layer_tensors(layer_id)) {
        if (qlwc_) qlwc_->ensure(name);
        else ensure_lwc_tensor(name);
      }
      prefetching_ = -1;
      return;
    }
    // 预取：后台线程池逐 blob submit（LWC/QLWC 均经同步 ReadBlob 简化；标记已预取）
    for (const auto& name : list_layer_tensors(layer_id)) {
      if (qlwc_) qlwc_->ensure(name);
      else ensure_lwc_tensor(name);
    }
    prefetching_ = -1;
  }

  void release_layer(int layer_id) override {
    if (layer_id < 0 || !pinned_.count(layer_id)) return;
    pinned_.erase(layer_id);
    ++st_.releases;
    free_cuda_layer(layer_id);
    // 保留仍 pin 的层张量
    std::unordered_set<std::string> keep;
    for (int L : pinned_) {
      for (const auto& n : list_layer_tensors(L)) keep.insert(n);
    }
    if (qlwc_) qlwc_->release_except(keep);
    else {
      for (auto it = lwc_blobs_.begin(); it != lwc_blobs_.end();) {
        if (it->first.find("layers.") != std::string::npos && !keep.count(it->first))
          it = lwc_blobs_.erase(it);
        else
          ++it;
      }
    }
    st_.window_bytes = current_window_bytes();
  }

  LayerStreamStats stats() const override { return st_; }
  qlwc::QlwcStore* qlwc() override { return qlwc_.get(); }
  bool is_qlwc() const override { return qlwc_ != nullptr; }

 private:
  void drain_prefetch() {
    if (engine_) engine_->drain();
  }

  std::vector<std::string> list_layer_tensors(int L) const {
    const std::string key = layer_key(cfg_.layer_prefix, L);
    std::vector<std::string> out;
    if (qlwc_) {
      for (const auto& t : qlwc_->header().tensors) {
        if (t.name.find(key) != std::string::npos) out.push_back(t.name);
      }
    } else {
      for (const auto& t : lwc_hdr_.tensors) {
        if (t.name.find(key) != std::string::npos) out.push_back(t.name);
      }
    }
    return out;
  }

  uint64_t layer_bytes(int L) const {
    uint64_t n = 0;
    const std::string key = layer_key(cfg_.layer_prefix, L);
    if (qlwc_) {
      for (const auto& t : qlwc_->header().tensors) {
        if (t.name.find(key) == std::string::npos) continue;
        if (t.kind == qlwc::TensorKind::kPassthrough) n += t.data_nbytes;
        else n += t.q_nbytes + t.scales_nbytes + t.zeros_nbytes;
      }
    } else {
      for (const auto& t : lwc_hdr_.tensors) {
        if (t.name.find(key) != std::string::npos) n += t.nbytes;
      }
    }
    return n;
  }

  uint64_t current_window_bytes() const {
    uint64_t n = 0;
    for (int L : pinned_) n += layer_bytes(L);
    return n;
  }

  void enforce_window() {
    while (static_cast<int>(pinned_.size()) > cfg_.window_layers) {
      // 丢掉最小 layer_id（滑窗）
      int victim = *pinned_.begin();
      for (int L : pinned_)
        if (L < victim) victim = L;
      // 若 victim 是刚 pin 的最高层则换最大以外的最小
      int newest = victim;
      for (int L : pinned_)
        if (L > newest) newest = L;
      victim = newest;
      for (int L : pinned_)
        if (L != newest && L < victim) victim = L;
      if (pinned_.size() <= static_cast<size_t>(cfg_.window_layers)) break;
      if (victim == newest && pinned_.size() > 1) {
        // pick any except newest
        for (int L : pinned_)
          if (L != newest) {
            victim = L;
            break;
          }
      }
      release_layer(victim);
    }
  }

  void ensure_lwc_tensor(const std::string& name) {
    if (lwc_blobs_.count(name)) return;
    const auto* t = lwc_hdr_.find(name);
    if (!t) throw std::runtime_error("LWC missing: " + name);
    lwc_blobs_[name] = lwc::ReadTensor(path_, lwc_hdr_, name);
  }

  void upload_layer_cuda(int L, const std::vector<std::string>& names) {
    free_cuda_layer(L);
    auto& slot = cuda_layers_[L];
    for (const auto& name : names) {
      const uint8_t* host = nullptr;
      size_t nbytes = 0;
      if (qlwc_) {
        if (qlwc_->is_int4(name)) {
          // 上传 qweight 主块（scales 仍主机侧供现有 INT4 GEMM）
          const auto* t = qlwc_->header().find(name);
          if (!t) continue;
          // blob already in store; get via ensure + internal — use pass for size
          // INT4 host compute: skip mandatory H2D; optional mirror for future GPU gemm
          (void)t;
          continue;
        }
        auto pv = qlwc_->get_pass(name);
        host = reinterpret_cast<const uint8_t*>(pv.data);
        nbytes = pv.n_elem * 2;
      } else {
        auto it = lwc_blobs_.find(name);
        if (it == lwc_blobs_.end()) continue;
        host = it->second.data();
        nbytes = it->second.size();
      }
      if (!host || !nbytes) continue;
      void* dev = llmoc::hal::cuda::device_alloc(nbytes);
      if (!dev) continue;
      if (!llmoc::hal::cuda::h2d(dev, host, nbytes)) {
        llmoc::hal::cuda::device_free(dev);
        continue;
      }
      slot[name] = {dev, nbytes};
      st_.cuda_h2d_bytes += nbytes;
    }
  }

  void free_cuda_layer(int L) {
    auto it = cuda_layers_.find(L);
    if (it == cuda_layers_.end()) return;
    for (auto& kv : it->second) llmoc::hal::cuda::device_free(kv.second.ptr);
    cuda_layers_.erase(it);
  }

  void clear_cuda() {
    for (auto& kv : cuda_layers_)
      for (auto& p : kv.second) llmoc::hal::cuda::device_free(p.second.ptr);
    cuda_layers_.clear();
  }

  LayerStreamConfig cfg_;
  std::string path_;
  bool use_cuda_ = false;
  std::unique_ptr<qlwc::QlwcStore> qlwc_;
  lwc::Header lwc_hdr_;
  std::unordered_map<std::string, std::vector<uint8_t>> lwc_blobs_;
  std::unordered_set<int> pinned_;
  int prefetching_ = -1;
  std::unique_ptr<io::IoEngine> engine_;
  LayerStreamStats st_;

  struct DevBuf {
    void* ptr = nullptr;
    size_t nbytes = 0;
  };
  std::unordered_map<int, std::unordered_map<std::string, DevBuf>> cuda_layers_;
};

std::unique_ptr<ILayerStreamLoader> make_layer_stream_loader() {
  return std::make_unique<LayerStreamLoader>();
}

}  // namespace llmoc::wt
