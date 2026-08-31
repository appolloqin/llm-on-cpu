#pragma once
// llm-on-cpu :: exec/gpu/residency_gpu.h — VRAM expert slot pool

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "contracts/weight_block.h"

namespace llmoc::exec::gpu {

struct ExpertSlot {
  contracts::ExpertId id{};
  void* device_ptr = nullptr;
  size_t bytes = 0;
  bool occupied = false;
};

class ExpertSlotPool {
 public:
  void configure(int capacity, size_t vram_budget);
  // Upload host bytes into a slot (LRU evict if full). Returns device ptr or nullptr.
  void* pin_upload(contracts::ExpertId id, const void* host, size_t bytes);
  void* lookup(contracts::ExpertId id) const;
  void release(contracts::ExpertId id);
  int capacity() const { return capacity_; }
  int used() const { return used_; }

 private:
  static uint64_t key(contracts::ExpertId id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(id.layer)) << 32) |
           static_cast<uint32_t>(id.expert);
  }
  void evict_one();

  mutable std::mutex mu_;
  int capacity_ = 8;
  int used_ = 0;
  size_t budget_ = 0;
  size_t used_bytes_ = 0;
  std::unordered_map<uint64_t, ExpertSlot> map_;
  std::vector<uint64_t> lru_;
};

}  // namespace llmoc::exec::gpu
