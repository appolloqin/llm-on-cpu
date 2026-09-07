// llm-on-cpu :: moe/offload_cache.cpp
#include "moe/offload_cache.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "common/log.h"
#include "hal/cuda_backend.h"
#include "moe/qlwc_expert_bytes.h"

namespace llmoc::moe {

void OffloadMoeCache::configure(const OffloadCacheConfig& cfg, QlwcExpertHostBanks* banks) {
  cfg_ = cfg;
  banks_ = banks;
  if (cfg_.num_layers <= 0 || cfg_.num_experts <= 0 || cfg_.cache_size <= 0) {
    throw std::runtime_error("OffloadMoeCache: invalid config");
  }
  if (cfg_.prefill_overlap && cfg_.cache_size < 2 * cfg_.num_experts) {
    throw std::runtime_error(
        "OffloadMoeCache: prefill_overlap requires cache_size >= 2 * num_experts");
  }
  const int LE = cfg_.num_layers * cfg_.num_experts;
  slot_for_id_.assign(static_cast<size_t>(LE), -1);
  id_of_slot_.assign(static_cast<size_t>(cfg_.cache_size), -1);
  usage_.assign(static_cast<size_t>(cfg_.cache_size), 0);
  step_ = 0;
  misses_.clear();
  overflow_.clear();
  prefill_active_ = false;
  prefill_buf_ = 0;
  prefill_layer_of_buf_.assign(2, -1);
  ready_ = banks_ != nullptr && banks_->ready();
}

void OffloadMoeCache::clear_slot(int slot) {
  if (slot < 0 || slot >= cfg_.cache_size) return;
  const int flat = id_of_slot_[static_cast<size_t>(slot)];
  if (flat >= 0 && flat < static_cast<int>(slot_for_id_.size()) &&
      slot_for_id_[static_cast<size_t>(flat)] == slot) {
    slot_for_id_[static_cast<size_t>(flat)] = -1;
  }
  id_of_slot_[static_cast<size_t>(slot)] = -1;
  usage_[static_cast<size_t>(slot)] = 0;
}

int OffloadMoeCache::alloc_slot_lru(int flat, const int* protect, int n_protect) {
  // Prefer free slot outside prefill double-buffer region when possible.
  const int reserved = cfg_.prefill_overlap ? (2 * cfg_.num_experts) : 0;
  auto is_protected = [&](int s) {
    for (int i = 0; i < n_protect; ++i)
      if (protect[i] == s) return true;
    return false;
  };

  int best = -1;
  int64_t best_u = INT64_MAX;
  for (int s = reserved; s < cfg_.cache_size; ++s) {
    if (is_protected(s)) continue;
    if (id_of_slot_[static_cast<size_t>(s)] < 0) {
      best = s;
      break;
    }
    if (usage_[static_cast<size_t>(s)] < best_u) {
      best_u = usage_[static_cast<size_t>(s)];
      best = s;
    }
  }
  if (best < 0) {
    // Fall back to any slot including reserved.
    for (int s = 0; s < cfg_.cache_size; ++s) {
      if (is_protected(s)) continue;
      if (id_of_slot_[static_cast<size_t>(s)] < 0) {
        best = s;
        break;
      }
      if (usage_[static_cast<size_t>(s)] < best_u) {
        best_u = usage_[static_cast<size_t>(s)];
        best = s;
      }
    }
  }
  if (best < 0) throw std::runtime_error("OffloadMoeCache: no slot available");
  clear_slot(best);
  id_of_slot_[static_cast<size_t>(best)] = flat;
  slot_for_id_[static_cast<size_t>(flat)] = best;
  usage_[static_cast<size_t>(best)] = ++step_;
  return best;
}

void OffloadMoeCache::ensure_experts(int layer, int* topk_ids, int K) {
  misses_.clear();
  overflow_.clear();
  ensured_experts_.assign(static_cast<size_t>(K), -1);
  if (!ready_ || !topk_ids || K <= 0) return;
  if (cfg_.backend == MoeBackend::kCpu || !banks_->layer_gpu_ok(layer)) {
    for (int i = 0; i < K; ++i) {
      ensured_experts_[static_cast<size_t>(i)] = topk_ids[i];
      overflow_.push_back(topk_ids[i]);
      topk_ids[i] = -1;
    }
    return;
  }

  std::vector<int> protect;
  protect.reserve(static_cast<size_t>(K));
  for (int i = 0; i < K; ++i) {
    const int e = topk_ids[i];
    ensured_experts_[static_cast<size_t>(i)] = e;
    if (e < 0 || e >= cfg_.num_experts) {
      topk_ids[i] = -1;
      continue;
    }
    const int flat = flat_id(layer, e);
    int slot = slot_for_id_[static_cast<size_t>(flat)];
    if (slot >= 0) {
      usage_[static_cast<size_t>(slot)] = ++step_;
      topk_ids[i] = slot;
      protect.push_back(slot);
    } else {
      slot = alloc_slot_lru(flat, protect.data(), static_cast<int>(protect.size()));
      topk_ids[i] = slot;
      protect.push_back(slot);
      misses_.push_back(slot);
    }
  }
}

void OffloadMoeCache::ensure_experts_hybrid(int layer, int* topk_ids, int K) {
  if (cfg_.backend != MoeBackend::kHybrid) {
    ensure_experts(layer, topk_ids, K);
    return;
  }
  misses_.clear();
  overflow_.clear();
  ensured_experts_.assign(static_cast<size_t>(K), -1);
  if (!ready_ || !topk_ids || K <= 0) return;
  if (!banks_->layer_gpu_ok(layer)) {
    for (int i = 0; i < K; ++i) {
      ensured_experts_[static_cast<size_t>(i)] = topk_ids[i];
      overflow_.push_back(topk_ids[i]);
      topk_ids[i] = -1;
    }
    return;
  }

  std::vector<int> hit_slots;
  std::vector<int> miss_experts;
  hit_slots.reserve(static_cast<size_t>(K));
  miss_experts.reserve(static_cast<size_t>(K));

  for (int i = 0; i < K; ++i) {
    const int e = topk_ids[i];
    ensured_experts_[static_cast<size_t>(i)] = e;
    if (e < 0 || e >= cfg_.num_experts) {
      topk_ids[i] = -1;
      continue;
    }
    const int flat = flat_id(layer, e);
    const int slot = slot_for_id_[static_cast<size_t>(flat)];
    if (slot >= 0) {
      usage_[static_cast<size_t>(slot)] = ++step_;
      topk_ids[i] = slot;
      hit_slots.push_back(slot);
    } else {
      miss_experts.push_back(e);
      topk_ids[i] = -2;  // placeholder: miss, decide fetch vs CPU below
    }
  }

  const int nmiss = static_cast<int>(miss_experts.size());
  float frac = cfg_.hybrid_fetch_frac;
  if (frac < 0.f) frac = 0.f;
  if (frac > 1.f) frac = 1.f;
  int nfetch = static_cast<int>(frac * static_cast<float>(nmiss) + 0.5f);
  if (nmiss > 0 && frac > 0.f && nfetch == 0) nfetch = 1;
  if (nfetch > nmiss) nfetch = nmiss;
  last_fetch_count_ = nfetch;

  // Fetch first nfetch misses (stable order); rest → CPU overflow.
  std::vector<int> protect = hit_slots;
  int fetched = 0;
  for (int i = 0; i < K; ++i) {
    if (topk_ids[i] != -2) continue;
    const int e = ensured_experts_[static_cast<size_t>(i)];
    if (fetched < nfetch) {
      const int flat = flat_id(layer, e);
      const int slot = alloc_slot_lru(flat, protect.data(), static_cast<int>(protect.size()));
      topk_ids[i] = slot;
      protect.push_back(slot);
      misses_.push_back(slot);
      ++fetched;
    } else {
      topk_ids[i] = -1;
      overflow_.push_back(e);
    }
  }
}

int OffloadMoeCache::copy_missing(int layer) {
  last_copy_bytes_ = 0;
  if (!ready_ || !hal::cuda::enabled() || misses_.empty()) return 0;
  if (!banks_->layer_gpu_ok(layer)) return 0;

  int n = 0;
  for (int slot : misses_) {
    if (slot < 0 || slot >= cfg_.cache_size) continue;
    const int flat = id_of_slot_[static_cast<size_t>(slot)];
    if (flat < 0) continue;
    const int e = flat % cfg_.num_experts;
    const int L = flat / cfg_.num_experts;
    if (L != layer) continue;
    const auto& g = banks_->gate(L, e);
    const auto& u = banks_->up(L, e);
    const auto& d = banks_->down(L, e);
    const bool ok = hal::cuda::prefetch_int4_weight(g) && hal::cuda::prefetch_int4_weight(u) &&
                    hal::cuda::prefetch_int4_weight(d);
    if (ok) {
      ++n;
      last_copy_bytes_ += expert_device_bytes(banks_->config().hidden, banks_->config().intermediate,
                                              banks_->config().group_size, banks_->config().has_zeros);
    }
  }
  return n;
}

bool OffloadMoeCache::views_for_slot(int slot, qlwc::Int4View& gate, qlwc::Int4View& up,
                                     qlwc::Int4View& down) const {
  if (!ready_ || slot < 0 || slot >= cfg_.cache_size) return false;
  const int flat = id_of_slot_[static_cast<size_t>(slot)];
  if (flat < 0) return false;
  const int e = flat % cfg_.num_experts;
  const int L = flat / cfg_.num_experts;
  return views_for_expert(L, e, gate, up, down);
}

bool OffloadMoeCache::views_for_expert(int layer, int expert, qlwc::Int4View& gate,
                                       qlwc::Int4View& up, qlwc::Int4View& down) const {
  if (!ready_ || !banks_) return false;
  if (!banks_->has_layer(layer)) return false;
  if (expert < 0 || expert >= cfg_.num_experts) return false;
  gate = banks_->gate(layer, expert);
  up = banks_->up(layer, expert);
  down = banks_->down(layer, expert);
  return gate.qweight != nullptr;
}

void OffloadMoeCache::begin_prefill() {
  prefill_active_ = cfg_.prefill_overlap;
  prefill_buf_ = 0;
  prefill_layer_of_buf_[0] = -1;
  prefill_layer_of_buf_[1] = -1;
}

void OffloadMoeCache::prefetch_prefill_layer(int layer) {
  if (!prefill_active_ || !ready_ || !banks_) return;
  if (layer < 0 || layer >= cfg_.num_layers) return;
  if (!banks_->layer_gpu_ok(layer)) return;

  const int E = cfg_.num_experts;
  const int buf = prefill_buf_;
  const int base = buf * E;
  // Evict previous occupants of this buffer half.
  for (int s = base; s < base + E; ++s) clear_slot(s);

  for (int e = 0; e < E; ++e) {
    const int flat = flat_id(layer, e);
    // If expert already mapped elsewhere, leave it; still fill buffer slot for stable layout.
    const int slot = base + e;
    clear_slot(slot);
    id_of_slot_[static_cast<size_t>(slot)] = flat;
    slot_for_id_[static_cast<size_t>(flat)] = slot;
    usage_[static_cast<size_t>(slot)] = ++step_;

    if (hal::cuda::enabled()) {
      (void)hal::cuda::prefetch_int4_weight(banks_->gate(layer, e));
      (void)hal::cuda::prefetch_int4_weight(banks_->up(layer, e));
      (void)hal::cuda::prefetch_int4_weight(banks_->down(layer, e));
    }
  }
  prefill_layer_of_buf_[static_cast<size_t>(buf)] = layer;
  prefill_buf_ = 1 - prefill_buf_;
}

void OffloadMoeCache::wait_prefill_layer(int layer) {
  // Synchronous prefetch_int4_weight — nothing to wait. Keep API for future async.
  (void)layer;
}

void OffloadMoeCache::release_prefill_layer(int layer) {
  if (!prefill_active_) return;
  for (int b = 0; b < 2; ++b) {
    if (prefill_layer_of_buf_[static_cast<size_t>(b)] != layer) continue;
    const int base = b * cfg_.num_experts;
    for (int s = base; s < base + cfg_.num_experts; ++s) clear_slot(s);
    prefill_layer_of_buf_[static_cast<size_t>(b)] = -1;
  }
}

}  // namespace llmoc::moe
