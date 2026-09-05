// llm-on-cpu :: model/kv_cache.cpp
#include "model/kv_cache.h"

#include <cstring>
#include <string>

namespace llmoc::model {

void SessionCache::init(int n_layers, int max_seq, int n_kv_heads, int head_dim, int n_v_heads,
                        int dk, int dv, int conv_dim, int conv_k) {
  init(n_layers, max_seq, n_kv_heads, head_dim, n_v_heads, dk, dv, conv_dim, conv_k, nullptr);
}

void SessionCache::init(int n_layers, int max_seq, int n_kv_heads, int head_dim, int n_v_heads,
                        int dk, int dv, int conv_dim, int conv_k,
                        const std::vector<uint8_t>* need_full_kv) {
  max_seq_ = max_seq;
  n_kv_heads_ = n_kv_heads;
  head_dim_ = head_dim;
  layers_.assign(n_layers, {});
  for (int i = 0; i < n_layers; ++i) {
    auto& L = layers_[static_cast<size_t>(i)];
    const bool want_kv =
        !need_full_kv || static_cast<int>(need_full_kv->size()) <= i || (*need_full_kv)[static_cast<size_t>(i)];
    if (want_kv) {
      L.k.assign(static_cast<size_t>(n_kv_heads) * max_seq * head_dim, 0.f);
      L.v.assign(static_cast<size_t>(n_kv_heads) * max_seq * head_dim, 0.f);
    }
    L.seq = 0;
    L.linear.conv.assign(static_cast<size_t>(conv_dim) * conv_k, 0.f);
    L.linear.recurrent.assign(static_cast<size_t>(n_v_heads) * dk * dv, 0.f);
    L.linear.has_state = false;
  }
  tokens.clear();
  prefix_hash = 0;
}

void SessionCache::clear() {
  for (auto& L : layers_) {
    L.seq = 0;
    L.linear.has_state = false;
    std::fill(L.linear.conv.begin(), L.linear.conv.end(), 0.f);
    std::fill(L.linear.recurrent.begin(), L.linear.recurrent.end(), 0.f);
  }
  tokens.clear();
  prefix_hash = 0;
}

SessionCache::Snapshot SessionCache::snapshot() const {
  Snapshot s;
  s.seq.resize(layers_.size());
  s.linear.resize(layers_.size());
  for (size_t i = 0; i < layers_.size(); ++i) {
    s.seq[i] = layers_[i].seq;
    s.linear[i] = layers_[i].linear;  // 深拷贝 vector
  }
  s.tokens = tokens;
  return s;
}

void SessionCache::restore(const Snapshot& snap) {
  if (snap.seq.size() != layers_.size()) return;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i].seq = snap.seq[i];
    layers_[i].linear = snap.linear[i];
    // K/V 槽位：seq 缩短后旧内容可忽略（decode 只读 [0,seq)）
  }
  tokens = snap.tokens;
}

std::string RadixKvPool::key_of(const std::vector<int32_t>& t, size_t n) {
  std::string s;
  s.resize(n * sizeof(int32_t));
  if (n) std::memcpy(s.data(), t.data(), n * sizeof(int32_t));
  return s;
}

void RadixKvPool::insert(const std::vector<int32_t>& tokens, int cached_len) {
  if (cached_len <= 0) return;
  map_[key_of(tokens, static_cast<size_t>(cached_len))] = cached_len;
}

int RadixKvPool::longest_prefix(const std::vector<int32_t>& tokens) const {
  int best = 0;
  for (size_t n = 1; n <= tokens.size(); ++n) {
    auto it = map_.find(key_of(tokens, n));
    if (it != map_.end()) best = it->second;
  }
  return best;
}

}  // namespace llmoc::model
