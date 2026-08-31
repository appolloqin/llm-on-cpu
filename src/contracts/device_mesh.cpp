// llm-on-cpu :: contracts/device_mesh.cpp
#include "contracts/device_mesh.h"

#include <algorithm>

namespace llmoc::contracts {

bool resolve_device_mesh(const DeviceMeshSpec& spec, int visible_gpus, bool nccl_available,
                        bool moe_family, DeviceMesh* out, std::string* err) {
  if (!out) {
    if (err) *err = "resolve_device_mesh: null out";
    return false;
  }
  DeviceMesh mesh;
  mesh.ep_shard = spec.ep_shard;

  std::vector<int> ids = spec.ids;
  if (ids.empty()) {
    ids.push_back(0);
  } else if (ids.size() == 1 && ids[0] < 0) {
    // sentinel: auto all visible
    ids.clear();
    for (int i = 0; i < visible_gpus; ++i) ids.push_back(i);
    if (ids.empty()) ids.push_back(0);
  }

  // Dedup & clamp
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  for (int& d : ids) {
    if (d < 0) d = 0;
    if (visible_gpus > 0 && d >= visible_gpus) {
      if (err) *err = "devices.ids out of range for visible GPUs";
      return false;
    }
  }

  mesh.ids = ids;
  mesh.world_size = static_cast<int>(ids.size());
  if (mesh.world_size < 1) mesh.world_size = 1;

  MeshStrategy strat = spec.strategy;
  if (strat == MeshStrategy::kAuto) {
    if (mesh.world_size == 1) {
      strat = moe_family ? MeshStrategy::kEp : MeshStrategy::kTp;
    } else if (moe_family || spec.has_moe) {
      strat = MeshStrategy::kEp;
    } else {
      strat = MeshStrategy::kTp;
    }
  }
  mesh.strategy = strat;

  int ep = spec.ep_size;
  int tp = spec.tp_size;
  if (ep <= 0 && tp <= 0) {
    if (strat == MeshStrategy::kEp) {
      ep = mesh.world_size;
      tp = 1;
    } else if (strat == MeshStrategy::kTp) {
      ep = 1;
      tp = mesh.world_size;
    } else {  // ep_tp: split as evenly as possible favoring ep for MoE
      ep = mesh.world_size;
      tp = 1;
      for (int e = mesh.world_size; e >= 1; --e) {
        if (mesh.world_size % e == 0) {
          ep = e;
          tp = mesh.world_size / e;
          break;
        }
      }
    }
  } else {
    if (ep <= 0) ep = 1;
    if (tp <= 0) tp = mesh.world_size / ep;
  }

  if (ep * tp != mesh.world_size) {
    if (err) {
      *err = "ep_size * tp_size must equal world_size (" + std::to_string(ep) + "*" +
             std::to_string(tp) + "!=" + std::to_string(mesh.world_size) + ")";
    }
    return false;
  }
  mesh.ep_size = ep;
  mesh.tp_size = tp;

  if (mesh.world_size > 1) {
    if (spec.require_nccl && !nccl_available) {
      if (err) *err = "pure_gpu/hybrid mesh: world_size>1 but NCCL unavailable";
      return false;
    }
    // require_nccl=false → allow mesh with host collective stub (tests / no NCCL yet)
    mesh.nccl_ok = nccl_available || !spec.require_nccl;
  } else {
    mesh.nccl_ok = false;
  }

  *out = mesh;
  return true;
}

}  // namespace llmoc::contracts
