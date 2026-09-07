// llm-on-cpu :: moe/host_banks.cpp
#include "moe/host_banks.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/log.h"
#include "common/platform.h"
#include "hal/cuda_backend.h"
#include "moe/qlwc_expert_bytes.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace llmoc::moe {

QlwcExpertHostBanks::~QlwcExpertHostBanks() { free_all(); }

void QlwcExpertHostBanks::free_all() {
  for (auto& L : layers_) {
    if (L.arena) {
      if (L.residency == HostResidency::kPinned) {
        hal::cuda::host_unregister(L.arena);
      } else if (L.residency == HostResidency::kLocked) {
        sys::unlock_pages(L.arena, L.arena_bytes);
      }
      free_arena(L.arena, L.arena_bytes);
      L.arena = nullptr;
      L.arena_bytes = 0;
    }
    L.gate.clear();
    L.up.clear();
    L.down.clear();
  }
  layers_.clear();
  ready_ = false;
  pinned_bytes_ = 0;
  locked_bytes_ = 0;
  total_arena_bytes_ = 0;
}

uint8_t* QlwcExpertHostBanks::alloc_arena(size_t nbytes) {
  if (nbytes == 0) return nullptr;
#if defined(_WIN32)
  void* p = VirtualAlloc(nullptr, nbytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!p) throw std::bad_alloc();
  return static_cast<uint8_t*>(p);
#else
  void* p = nullptr;
  if (posix_memalign(&p, 4096, nbytes) != 0 || !p) throw std::bad_alloc();
  return static_cast<uint8_t*>(p);
#endif
}

void QlwcExpertHostBanks::free_arena(uint8_t* p, size_t nbytes) {
  if (!p) return;
  (void)nbytes;
#if defined(_WIN32)
  VirtualFree(p, 0, MEM_RELEASE);
#else
  free(p);
#endif
}

void QlwcExpertHostBanks::configure(const HostBanksConfig& cfg) {
  free_all();
  cfg_ = cfg;
  if (cfg_.num_layers <= 0 || cfg_.num_experts <= 0 || cfg_.hidden <= 0 ||
      cfg_.intermediate <= 0) {
    throw std::runtime_error("QlwcExpertHostBanks: invalid geometry");
  }
}

void QlwcExpertHostBanks::allocate() {
  free_all();
  const int L = cfg_.num_layers;
  layers_.resize(static_cast<size_t>(L));
  for (int li = 0; li < L; ++li) {
    layers_[static_cast<size_t>(li)].n_experts = cfg_.num_experts;
    layers_[static_cast<size_t>(li)].residency = HostResidency::kPageable;
  }
  ready_ = false;
}

void QlwcExpertHostBanks::alloc_layer_arena(int li) {
  auto& lb = layers_[static_cast<size_t>(li)];
  if (lb.arena) return;
  const int E = cfg_.num_experts;
  const int H = cfg_.hidden;
  const int I = cfg_.intermediate;
  const size_t per_expert = expert_storage_bytes(H, I, cfg_.group_size, cfg_.has_zeros);
  const size_t layer_bytes = per_expert * static_cast<size_t>(E);
  lb.n_experts = E;
  lb.arena_bytes = layer_bytes;
  lb.arena = alloc_arena(layer_bytes);
  std::memset(lb.arena, 0, layer_bytes);
  lb.gate.resize(static_cast<size_t>(E));
  lb.up.resize(static_cast<size_t>(E));
  lb.down.resize(static_cast<size_t>(E));

  const size_t gate_sz = int4_proj_storage_bytes(I, H, cfg_.group_size, cfg_.has_zeros);
  const size_t up_sz = gate_sz;
  const size_t down_sz = int4_proj_storage_bytes(H, I, cfg_.group_size, cfg_.has_zeros);
  size_t off = 0;
  for (int e = 0; e < E; ++e) {
    lb.gate[static_cast<size_t>(e)].base = lb.arena + off;
    lb.gate[static_cast<size_t>(e)].nbytes = gate_sz;
    off += gate_sz;
    lb.up[static_cast<size_t>(e)].base = lb.arena + off;
    lb.up[static_cast<size_t>(e)].nbytes = up_sz;
    off += up_sz;
    lb.down[static_cast<size_t>(e)].base = lb.arena + off;
    lb.down[static_cast<size_t>(e)].nbytes = down_sz;
    off += down_sz;
  }
  total_arena_bytes_ += layer_bytes;
  build_views_for_layer(li);
}

void QlwcExpertHostBanks::build_views_for_layer(int only_L) {
  const int H = cfg_.hidden;
  const int I = cfg_.intermediate;
  const int gs = cfg_.group_size;
  auto wire = [&](ExpertProjBank& b, int M, int K) {
    uint8_t* p = b.base;
    const size_t qn = int4_q_bytes(M, K);
    const size_t sn = int4_scales_f16_bytes(M, K, gs);
    const size_t zn = int4_zeros_f16_bytes(M, K, gs, cfg_.has_zeros);
    b.view.qweight = p;
    p += qn;
    b.view.scales = reinterpret_cast<const uint16_t*>(p);
    p += sn;
    if (zn) {
      b.view.zeros = reinterpret_cast<const uint16_t*>(p);
      p += zn;
    } else {
      b.view.zeros = nullptr;
    }
    b.view.scales_f32 = nullptr;  // CPU gemm converts fp16 on demand
    b.view.M = M;
    b.view.K = K;
    b.view.group_size = gs;
    b.view.scheme = cfg_.scheme;
    b.view.awq_zp = cfg_.awq_zp;
  };

  for (int li = 0; li < cfg_.num_layers; ++li) {
    if (only_L >= 0 && li != only_L) continue;
    auto& lb = layers_[static_cast<size_t>(li)];
    for (int e = 0; e < cfg_.num_experts; ++e) {
      wire(lb.gate[static_cast<size_t>(e)], I, H);
      wire(lb.up[static_cast<size_t>(e)], I, H);
      wire(lb.down[static_cast<size_t>(e)], H, I);
    }
  }
}

void QlwcExpertHostBanks::fill_from_qlwc(qlwc::QlwcStore& store) {
  if (layers_.empty()) allocate();
  const int L = cfg_.num_layers;
  const int E = cfg_.num_experts;
  const auto& hdr = store.header();

  // Copy one projection into the bank, then drop the QLWC blob immediately.
  // Without drop, fill keeps a second full copy of every expert in store → OOM (~2× DRAM).
  auto copy_proj = [&](ExpertProjBank& dst, const std::string& name) {
    store.ensure(name);
    const auto src = store.get_int4(name);
    if (src.M != dst.view.M || src.K != dst.view.K) {
      throw std::runtime_error("expert shape mismatch: " + name);
    }
    const size_t qn = int4_q_bytes(src.M, src.K);
    const size_t sn = int4_scales_f16_bytes(src.M, src.K, src.group_size);
    std::memcpy(const_cast<uint8_t*>(dst.view.qweight), src.qweight, qn);
    std::memcpy(const_cast<uint16_t*>(dst.view.scales), src.scales, sn);
    if (cfg_.has_zeros) {
      const size_t zn = int4_zeros_f16_bytes(src.M, src.K, src.group_size, true);
      if (src.zeros) {
        std::memcpy(const_cast<uint16_t*>(dst.view.zeros), src.zeros, zn);
      } else {
        std::memset(const_cast<uint16_t*>(dst.view.zeros), 0, zn);
      }
    }
    store.drop(name);
  };

  int n_moe = 0;
  const size_t layer_bytes_est =
      expert_storage_bytes(cfg_.hidden, cfg_.intermediate, cfg_.group_size, cfg_.has_zeros) *
      static_cast<size_t>(E);
  for (int li = 0; li < L; ++li) {
    const std::string probe = cfg_.name_prefix + "layers." + std::to_string(li) +
                              ".mlp.experts.0.gate_proj.weight";
    if (!store.has(probe)) continue;

    if (cfg_.dram_budget_bytes > 0 &&
        total_arena_bytes_ + layer_bytes_est > cfg_.dram_budget_bytes) {
      LOG_WARN("moe host_banks: dram_hot budget reached before layer %d "
               "(arena=%.2fGiB budget=%.2fGiB) — remaining MoE layers use disk ensure",
               li, total_arena_bytes_ / double(1ull << 30),
               cfg_.dram_budget_bytes / double(1ull << 30));
      break;
    }

    try {
      alloc_layer_arena(li);
    } catch (const std::bad_alloc&) {
      LOG_WARN("moe host_banks: OOM allocating layer %d (arena=%.2fGiB so far, store=%.2fGiB) — "
               "stop fill; remaining MoE layers fall back to disk ensure",
               li, total_arena_bytes_ / double(1ull << 30),
               store.loaded_bytes() / double(1ull << 30));
      break;
    }

    auto& lb = layers_[static_cast<size_t>(li)];
    try {
      for (int e = 0; e < E; ++e) {
        const std::string base = cfg_.name_prefix + "layers." + std::to_string(li) +
                                 ".mlp.experts." + std::to_string(e) + ".";
        copy_proj(lb.gate[static_cast<size_t>(e)], base + "gate_proj.weight");
        copy_proj(lb.up[static_cast<size_t>(e)], base + "up_proj.weight");
        copy_proj(lb.down[static_cast<size_t>(e)], base + "down_proj.weight");
      }
    } catch (const std::bad_alloc&) {
      LOG_WARN("moe host_banks: OOM while filling layer %d expert blob — stop fill "
               "(arena=%.2fGiB store=%.2fGiB)",
               li, total_arena_bytes_ / double(1ull << 30),
               store.loaded_bytes() / double(1ull << 30));
      // Drop partial layer arena so residency maps stay consistent.
      if (lb.arena) {
        free_arena(lb.arena, lb.arena_bytes);
        if (total_arena_bytes_ >= lb.arena_bytes) total_arena_bytes_ -= lb.arena_bytes;
        lb.arena = nullptr;
        lb.arena_bytes = 0;
        lb.gate.clear();
        lb.up.clear();
        lb.down.clear();
      }
      break;
    }

    ++n_moe;
    if (n_moe % 4 == 0 || li + 1 == L) {
      LOG_INFO("moe host_banks fill: moe_layers=%d last=%d/%d arena=%.2fGiB store=%.2fGiB "
               "(scheme=%u gs=%u)",
               n_moe, li + 1, L, total_arena_bytes_ / double(1ull << 30),
               store.loaded_bytes() / double(1ull << 30), static_cast<unsigned>(hdr.scheme),
               hdr.group_size);
    }
  }
  ready_ = n_moe > 0;
  LOG_INFO("moe host_banks fill done: moe_layers=%d arena=%.2fGiB store_left=%.2fGiB", n_moe,
           total_arena_bytes_ / double(1ull << 30), store.loaded_bytes() / double(1ull << 30));
}

bool QlwcExpertHostBanks::settle_layer(int L, HostResidency want) {
  auto& lb = layers_[static_cast<size_t>(L)];
  if (!lb.arena || lb.arena_bytes == 0) return false;

  if (want == HostResidency::kPinned) {
    if (hal::cuda::host_register(lb.arena, lb.arena_bytes)) {
      lb.residency = HostResidency::kPinned;
      pinned_bytes_ += lb.arena_bytes;
      return true;
    }
    // Fall through to lock.
    want = HostResidency::kLocked;
  }
  if (want == HostResidency::kLocked) {
    if (sys::lock_pages(lb.arena, lb.arena_bytes)) {
      lb.residency = HostResidency::kLocked;
      locked_bytes_ += lb.arena_bytes;
      return true;
    }
  }
  lb.residency = HostResidency::kPageable;
  return false;
}

void QlwcExpertHostBanks::pin() {
  pinned_bytes_ = 0;
  locked_bytes_ = 0;
  size_t used = 0;
  const size_t budget = cfg_.dram_budget_bytes;
  const bool want_pin = cfg_.host_pin;

  for (int li = 0; li < cfg_.num_layers; ++li) {
    auto& lb = layers_[static_cast<size_t>(li)];
    if (!lb.arena) continue;
    HostResidency want = want_pin ? HostResidency::kPinned : HostResidency::kLocked;
    if (budget > 0 && used + lb.arena_bytes > budget) {
      want = HostResidency::kLocked;
      LOG_WARN("moe host_banks: layer %d exceeds dram_hot (%.2f GiB used / %.2f GiB); "
               "CPU-only for this layer",
               li, used / double(1ull << 30), budget / double(1ull << 30));
    }
    if (!settle_layer(li, want)) {
      LOG_WARN("moe host_banks: layer %d settle failed → PAGEABLE (CPU decode)", li);
    } else {
      used += lb.arena_bytes;
    }
  }
  LOG_INFO("moe host_pin=%.2fGiB locked=%.2fGiB arena=%.2fGiB layers=%d experts=%d",
           pinned_bytes_ / double(1ull << 30), locked_bytes_ / double(1ull << 30),
           total_arena_bytes_ / double(1ull << 30), cfg_.num_layers, cfg_.num_experts);
}

void QlwcExpertHostBanks::prepare_synthetic_for_test(int n_moe_layers) {
  allocate();
  const int n = (std::min)(n_moe_layers, cfg_.num_layers);
  for (int li = 0; li < n; ++li) {
    alloc_layer_arena(li);
    layers_[static_cast<size_t>(li)].residency = HostResidency::kPinned;
  }
  ready_ = n > 0;
}

HostResidency QlwcExpertHostBanks::layer_residency(int layer) const {
  if (layer < 0 || layer >= cfg_.num_layers) return HostResidency::kPageable;
  return layers_[static_cast<size_t>(layer)].residency;
}

bool QlwcExpertHostBanks::has_layer(int layer) const {
  if (layer < 0 || layer >= cfg_.num_layers) return false;
  return layers_[static_cast<size_t>(layer)].arena != nullptr;
}

bool QlwcExpertHostBanks::layer_gpu_ok(int layer) const {
  return has_layer(layer) && layer_residency(layer) == HostResidency::kPinned;
}

const qlwc::Int4View& QlwcExpertHostBanks::gate(int layer, int expert) const {
  return layers_.at(static_cast<size_t>(layer)).gate.at(static_cast<size_t>(expert)).view;
}
const qlwc::Int4View& QlwcExpertHostBanks::up(int layer, int expert) const {
  return layers_.at(static_cast<size_t>(layer)).up.at(static_cast<size_t>(expert)).view;
}
const qlwc::Int4View& QlwcExpertHostBanks::down(int layer, int expert) const {
  return layers_.at(static_cast<size_t>(layer)).down.at(static_cast<size_t>(expert)).view;
}

}  // namespace llmoc::moe
