#pragma once
// llm-on-cpu :: moe/host_banks.h
// FreeToken-style pin-after-fill host banks for QLWC INT4 MoE experts.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "weights/qlwc_store.h"

namespace llmoc::moe {

enum class HostResidency : uint8_t {
  kPageable = 0,
  kLocked = 1,   // VirtualLock/mlock — CPU decode OK, no CUDA pin
  kPinned = 2,   // cudaHostRegister (or lock-only fallback)
};

struct ExpertProjBank {
  uint8_t* base = nullptr;  // owned by LayerExpertBanks arena
  size_t nbytes = 0;
  qlwc::Int4View view{};
};

struct LayerExpertBanks {
  int n_experts = 0;
  HostResidency residency = HostResidency::kPageable;
  // Contiguous arena: E * (gate|up|down storage)
  uint8_t* arena = nullptr;
  size_t arena_bytes = 0;
  std::vector<ExpertProjBank> gate;
  std::vector<ExpertProjBank> up;
  std::vector<ExpertProjBank> down;
};

struct HostBanksConfig {
  int num_layers = 0;
  int num_experts = 0;
  int hidden = 0;
  int intermediate = 0;  // moe intermediate
  int group_size = 128;
  bool has_zeros = true;  // GPTQ asym
  qlwc::Scheme scheme = qlwc::Scheme::kGptqAsym;
  int awq_zp = 0;
  std::string name_prefix = "language_model.";  // + layers.L.mlp.experts.E.…
  bool host_pin = true;
  size_t dram_budget_bytes = 0;  // 0 = unlimited for arena fill
  // Cap cudaHostRegister (WDDM pin quota). Excess layers use VirtualLock/PAGEABLE.
  // 0 = auto (~2GiB). Huge register blocks cudaMalloc/JIT on Windows.
  size_t cuda_pin_budget_bytes = 0;
};

class QlwcExpertHostBanks {
 public:
  QlwcExpertHostBanks() = default;
  ~QlwcExpertHostBanks();

  QlwcExpertHostBanks(const QlwcExpertHostBanks&) = delete;
  QlwcExpertHostBanks& operator=(const QlwcExpertHostBanks&) = delete;

  void configure(const HostBanksConfig& cfg);
  const HostBanksConfig& config() const { return cfg_; }

  // Allocate arenas (unpinned). Call before fill.
  void allocate();

  // Ensure + copy each expert tensor from QLWC into arenas; build Int4Views.
  void fill_from_qlwc(qlwc::QlwcStore& store);

  // Pin-after-fill: cudaHostRegister when CUDA available, else VirtualLock/mlock.
  // Layers that exceed dram_budget or fail pin become LOCKED (CPU-only decode).
  void pin();

  void set_cuda_pin_budget(size_t bytes) { cfg_.cuda_pin_budget_bytes = bytes; }
  void set_host_pin(bool on) { cfg_.host_pin = on; }

  // Unit-test helper: allocate N MoE layers with empty weights and mark ready/PINNED.
  void prepare_synthetic_for_test(int n_moe_layers);

  bool ready() const { return ready_; }
  int num_layers() const { return cfg_.num_layers; }
  int num_experts() const { return cfg_.num_experts; }
  bool has_layer(int layer) const;

  HostResidency layer_residency(int layer) const;
  bool layer_gpu_ok(int layer) const;  // PINNED only

  const qlwc::Int4View& gate(int layer, int expert) const;
  const qlwc::Int4View& up(int layer, int expert) const;
  const qlwc::Int4View& down(int layer, int expert) const;

  size_t pinned_bytes() const { return pinned_bytes_; }
  size_t locked_bytes() const { return locked_bytes_; }
  size_t total_arena_bytes() const { return total_arena_bytes_; }

 private:
  void free_all();
  uint8_t* alloc_arena(size_t nbytes);
  void free_arena(uint8_t* p, size_t nbytes);
  void alloc_layer_arena(int li);
  void build_views_for_layer(int L);
  bool settle_layer(int L, HostResidency want);

  HostBanksConfig cfg_{};
  std::vector<LayerExpertBanks> layers_;
  bool ready_ = false;
  size_t pinned_bytes_ = 0;
  size_t locked_bytes_ = 0;
  size_t total_arena_bytes_ = 0;
};

}  // namespace llmoc::moe
