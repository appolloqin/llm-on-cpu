#pragma once
// llm-on-cpu :: weights/layer_stream.h
// 层流式加载（跑起来优先）。见 docs/DESIGN_LAYER_STREAM.md

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "weights/io_engine.h"
#include "weights/lwc_format.h"
#include "weights/qlwc_store.h"

namespace llmoc::hal::cuda {
bool enabled();
void* device_alloc(size_t);
void device_free(void*);
bool h2d(void*, const void*, size_t);
}  // namespace llmoc::hal::cuda

namespace llmoc::wt {

struct LayerStreamConfig {
  int window_layers = 2;
  std::string device = "cpu";  // "cpu" | "cuda:0"
  uint64_t max_window_bytes = 0;
  unsigned io_workers = 2;
  int n_layers = 0;            // 0 = 从权重名推断
  std::string layer_prefix;    // e.g. "language_model.layers." or "layers."
};

struct LayerView {
  int layer_id = -1;
  std::vector<std::string> tensor_names;
  uint64_t bytes = 0;
};

struct LayerStreamStats {
  uint64_t pin_hits = 0;
  uint64_t pin_loads = 0;
  uint64_t prefetch_issued = 0;
  uint64_t releases = 0;
  uint64_t io_wait_ns = 0;
  uint64_t window_bytes = 0;
  uint64_t cuda_h2d_bytes = 0;
};

class ILayerStreamLoader {
 public:
  virtual ~ILayerStreamLoader() = default;
  virtual void open(const std::string& weight_path, const LayerStreamConfig& cfg) = 0;
  virtual void close() = 0;
  virtual LayerView pin_layer(int layer_id) = 0;
  virtual void prefetch_layer(int layer_id) = 0;
  virtual void release_layer(int layer_id) = 0;
  virtual LayerStreamStats stats() const = 0;
  // INT4 路径：暴露底层 store（lazy）
  virtual qlwc::QlwcStore* qlwc() { return nullptr; }
  virtual bool is_qlwc() const { return false; }
};

std::unique_ptr<ILayerStreamLoader> make_layer_stream_loader();

// 从 header 推断 layer 前缀与层数
void infer_layer_layout(const std::vector<std::string>& names, std::string* prefix_out,
                        int* n_layers_out);

}  // namespace llmoc::wt
