// llm-on-cpu :: exec/gpu/residency_gpu.cpp
#include "exec/gpu/residency_gpu.h"

#include "hal/cuda_backend.h"

namespace llmoc::exec::gpu {

void ExpertSlotPool::configure(int capacity, size_t vram_budget) {
  std::lock_guard<std::mutex> lock(mu_);
  capacity_ = capacity < 1 ? 1 : capacity;
  budget_ = vram_budget;
}

void ExpertSlotPool::evict_one() {
  if (lru_.empty()) return;
  const uint64_t k = lru_.front();
  lru_.erase(lru_.begin());
  auto it = map_.find(k);
  if (it == map_.end()) return;
  if (it->second.device_ptr) {
    llmoc::hal::cuda::device_free(it->second.device_ptr);
    if (used_bytes_ >= it->second.bytes) used_bytes_ -= it->second.bytes;
  }
  map_.erase(it);
  --used_;
}

void* ExpertSlotPool::lookup(contracts::ExpertId id) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = map_.find(key(id));
  if (it == map_.end() || !it->second.occupied) return nullptr;
  return it->second.device_ptr;
}

void ExpertSlotPool::release(contracts::ExpertId id) {
  std::lock_guard<std::mutex> lock(mu_);
  const uint64_t k = key(id);
  auto it = map_.find(k);
  if (it == map_.end()) return;
  if (it->second.device_ptr) {
    llmoc::hal::cuda::device_free(it->second.device_ptr);
    if (used_bytes_ >= it->second.bytes) used_bytes_ -= it->second.bytes;
  }
  map_.erase(it);
  --used_;
  for (auto i = lru_.begin(); i != lru_.end(); ++i) {
    if (*i == k) {
      lru_.erase(i);
      break;
    }
  }
}

void* ExpertSlotPool::pin_upload(contracts::ExpertId id, const void* host, size_t bytes) {
  if (!host || bytes == 0) return nullptr;
  std::lock_guard<std::mutex> lock(mu_);
  const uint64_t k = key(id);
  auto it = map_.find(k);
  if (it != map_.end() && it->second.occupied && it->second.bytes == bytes) {
    // refresh LRU
    for (auto i = lru_.begin(); i != lru_.end(); ++i) {
      if (*i == k) {
        lru_.erase(i);
        break;
      }
    }
    lru_.push_back(k);
    return it->second.device_ptr;
  }
  while (used_ >= capacity_ || (budget_ > 0 && used_bytes_ + bytes > budget_)) {
    if (lru_.empty()) break;
    evict_one();
  }
  void* d = llmoc::hal::cuda::device_alloc(bytes);
  if (!d) return nullptr;
  if (!llmoc::hal::cuda::h2d(d, host, bytes)) {
    llmoc::hal::cuda::device_free(d);
    return nullptr;
  }
  ExpertSlot s;
  s.id = id;
  s.device_ptr = d;
  s.bytes = bytes;
  s.occupied = true;
  map_[k] = s;
  lru_.push_back(k);
  ++used_;
  used_bytes_ += bytes;
  return d;
}

}  // namespace llmoc::exec::gpu
