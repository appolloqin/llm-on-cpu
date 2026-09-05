#pragma once
// llm-on-cpu :: model/kv_cache.h
// B期: 连续 KV + GatedDeltaNet 递推/卷积状态。C期: 简单 radix 前缀树复用。

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmoc::model {

struct LinearAttnState {
  std::vector<float> conv;       // [conv_dim * kernel]
  std::vector<float> recurrent;  // [n_v_heads * dk * dv]
  bool has_state = false;
};

struct LayerKv {
  // full attention: K/V [max_seq, n_kv_heads * head_dim] flattened as [n_kv, seq, dim] in ops
  std::vector<float> k;
  std::vector<float> v;
  int seq = 0;
  LinearAttnState linear;
};

class SessionCache {
 public:
  void init(int n_layers, int max_seq, int n_kv_heads, int head_dim, int n_v_heads, int dk, int dv,
            int conv_dim, int conv_k);
  // need_full_kv[i]=false → skip K/V alloc (linear-attn layers); nullptr = all layers need KV
  void init(int n_layers, int max_seq, int n_kv_heads, int head_dim, int n_v_heads, int dk, int dv,
            int conv_dim, int conv_k, const std::vector<uint8_t>* need_full_kv);
  void clear();

  // P0: 投机 verify 快照 / 回滚（拒绝后缀不得污染已提交状态）
  struct Snapshot {
    std::vector<int> seq;  // per-layer
    std::vector<LinearAttnState> linear;
    std::vector<int32_t> tokens;
  };
  Snapshot snapshot() const;
  void restore(const Snapshot& snap);

  LayerKv& layer(int i) { return layers_[i]; }
  const LayerKv& layer(int i) const { return layers_[i]; }
  int n_layers() const { return static_cast<int>(layers_.size()); }
  int max_seq() const { return max_seq_; }

  // C期: 记录 token 前缀指纹以便 radix 命中(单会话内简单复用)
  uint64_t prefix_hash = 0;
  std::vector<int32_t> tokens;

 private:
  std::vector<LayerKv> layers_;
  int max_seq_ = 0;
  int n_kv_heads_ = 0;
  int head_dim_ = 0;
};

// 极简 radix: token 序列 -> 可复用的 SessionCache 快照长度
class RadixKvPool {
 public:
  void insert(const std::vector<int32_t>& tokens, int cached_len);
  int longest_prefix(const std::vector<int32_t>& tokens) const;
  size_t nodes() const { return map_.size(); }

 private:
  std::unordered_map<std::string, int> map_;
  static std::string key_of(const std::vector<int32_t>& t, size_t n);
};

}  // namespace llmoc::model
