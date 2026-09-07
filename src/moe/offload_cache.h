#pragma once
// llm-on-cpu :: moe/offload_cache.h
// FreeToken-isomorphic VRAM expert slot LRU + prefill double-buffer + hybrid split.

#include <cstdint>
#include <vector>

#include "moe/host_banks.h"
#include "weights/qlwc_store.h"

namespace llmoc::moe {

enum class MoeBackend : uint8_t { kOffload = 0, kHybrid = 1, kCpu = 2 };

struct OffloadCacheConfig {
  int num_layers = 0;
  int num_experts = 0;
  int cache_size = 0;  // slots; must be >= 2*E if prefill_overlap
  bool prefill_overlap = true;
  MoeBackend backend = MoeBackend::kHybrid;
  float hybrid_fetch_frac = 0.5f;
};

class OffloadMoeCache {
 public:
  void configure(const OffloadCacheConfig& cfg, QlwcExpertHostBanks* banks);
  const OffloadCacheConfig& config() const { return cfg_; }
  bool ready() const { return ready_; }

  int cache_size() const { return cfg_.cache_size; }
  int num_experts() const { return cfg_.num_experts; }

  // Decode: rewrite topk_ids[K] → slot ids (or -1 for hybrid CPU overflow).
  // Records misses for copy_missing / overflow_experts for hybrid CPU.
  void ensure_experts(int layer, int* topk_ids, int K);
  void ensure_experts_hybrid(int layer, int* topk_ids, int K);

  // H2D miss experts from host banks via cuda prefetch_int4_weight.
  // Returns number of experts uploaded.
  int copy_missing(int layer);

  // Host Int4Views for a slot (or expert if topk still expert-id before ensure).
  bool views_for_slot(int slot, qlwc::Int4View& gate, qlwc::Int4View& up,
                      qlwc::Int4View& down) const;
  bool views_for_expert(int layer, int expert, qlwc::Int4View& gate, qlwc::Int4View& up,
                        qlwc::Int4View& down) const;

  // Experts that should run on CPU this step (hybrid overflow); expert ids.
  const std::vector<int>& overflow_experts() const { return overflow_; }
  const std::vector<int>& miss_slots() const { return misses_; }

  // Prefill double-buffer (borrows slots [0, 2E)).
  void begin_prefill();
  void prefetch_prefill_layer(int layer);
  void wait_prefill_layer(int layer);
  void release_prefill_layer(int layer);
  bool prefill_active() const { return prefill_active_; }

  // Stats
  int64_t step() const { return step_; }
  size_t last_copy_bytes() const { return last_copy_bytes_; }
  int last_fetch_count() const { return last_fetch_count_; }

 private:
  int flat_id(int layer, int expert) const {
    return layer * cfg_.num_experts + expert;
  }
  int alloc_slot_lru(int flat, const int* protect, int n_protect);
  void clear_slot(int slot);

  OffloadCacheConfig cfg_{};
  QlwcExpertHostBanks* banks_ = nullptr;
  bool ready_ = false;

  std::vector<int> slot_for_id_;  // [L*E] → slot or -1
  std::vector<int> id_of_slot_;   // [C] → flat id or -1
  std::vector<int64_t> usage_;    // [C]
  int64_t step_ = 0;

  std::vector<int> misses_;    // slot indices to H2D
  std::vector<int> overflow_;  // expert ids for CPU
  // Parallel to last ensure: original expert ids for each topk position (length K).
  std::vector<int> ensured_experts_;

  bool prefill_active_ = false;
  int prefill_buf_ = 0;  // 0 or 1
  std::vector<int> prefill_layer_of_buf_;  // [2] layer id or -1

  size_t last_copy_bytes_ = 0;
  int last_fetch_count_ = 0;
};

}  // namespace llmoc::moe
