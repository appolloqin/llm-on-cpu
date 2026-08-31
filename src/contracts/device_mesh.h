#pragma once
// llm-on-cpu :: contracts/device_mesh.h — single-node multi-GPU mesh (G8)

#include <cstdint>
#include <string>
#include <vector>

namespace llmoc::contracts {

enum class MeshStrategy { kAuto, kEp, kTp, kEpTp };

inline MeshStrategy parse_mesh_strategy(const std::string& s) {
  if (s == "ep") return MeshStrategy::kEp;
  if (s == "tp") return MeshStrategy::kTp;
  if (s == "ep_tp") return MeshStrategy::kEpTp;
  return MeshStrategy::kAuto;
}

inline const char* mesh_strategy_name(MeshStrategy s) {
  switch (s) {
    case MeshStrategy::kEp: return "ep";
    case MeshStrategy::kTp: return "tp";
    case MeshStrategy::kEpTp: return "ep_tp";
    default: return "auto";
  }
}

enum class EpShard { kMod, kContiguous };

inline EpShard parse_ep_shard(const std::string& s) {
  if (s == "contiguous") return EpShard::kContiguous;
  return EpShard::kMod;
}

struct DeviceMeshSpec {
  std::vector<int> ids;  // empty or {0} => single device
  MeshStrategy strategy = MeshStrategy::kAuto;
  int ep_size = 0;  // 0 = derive
  int tp_size = 0;
  EpShard ep_shard = EpShard::kMod;
  bool require_nccl = true;
  bool has_moe = true;  // family hint for auto strategy
};

struct DeviceMesh {
  std::vector<int> ids;
  int world_size = 1;
  int ep_size = 1;
  int tp_size = 1;
  MeshStrategy strategy = MeshStrategy::kTp;
  EpShard ep_shard = EpShard::kMod;
  bool nccl_ok = false;

  int local_rank = 0;  // this process view (single-process: iterate ranks in loops)

  int device_at(int rank) const {
    if (ids.empty()) return 0;
    if (rank < 0 || rank >= static_cast<int>(ids.size())) return ids.front();
    return ids[static_cast<size_t>(rank)];
  }

  // Expert ownership for EP: returns rank in [0, ep_size)
  int owner_rank(int expert_id, int n_experts) const {
    if (ep_size <= 1 || n_experts <= 0) return 0;
    if (ep_shard == EpShard::kContiguous) {
      const int per = (n_experts + ep_size - 1) / ep_size;
      int r = expert_id / per;
      if (r >= ep_size) r = ep_size - 1;
      return r;
    }
    return ((expert_id % ep_size) + ep_size) % ep_size;
  }

  bool owns_expert(int rank, int expert_id, int n_experts) const {
    return owner_rank(expert_id, n_experts) == (rank % ep_size);
  }

  std::string summary() const {
    std::string s = "mesh=";
    s += mesh_strategy_name(strategy);
    s += " world=" + std::to_string(world_size);
    s += " ep=" + std::to_string(ep_size);
    s += " tp=" + std::to_string(tp_size);
    s += " devices=";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i) s += ",";
      s += std::to_string(ids[i]);
    }
    return s;
  }
};

// Resolve mesh from spec + visible GPU count. Fills *err on failure.
// world_size==1 never requires NCCL.
bool resolve_device_mesh(const DeviceMeshSpec& spec, int visible_gpus, bool nccl_available,
                        bool moe_family, DeviceMesh* out, std::string* err);

}  // namespace llmoc::contracts
