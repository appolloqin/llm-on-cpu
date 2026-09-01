#pragma once
// llm-on-cpu :: families/kimi_k3/kimi_stub_model.h
// Kimi-STUB-v0：latent MoE；单卡 pure_gpu 装不下 → 降级 layer_stream（可运行）

#include "families/deepseek_v4/ds_stub_model.h"

namespace llmoc::families::kimi {

using KimiStubGeometry = deepseek::DsStubGeometry;
using KimiStubModel = deepseek::DsStubModel;

void write_fake_kimiq(const std::string& path, KimiStubGeometry g = {});

// true if Kimi product active-set cannot fit pure_gpu on a single card
bool pure_gpu_single_card_does_not_fit(contracts::ExecMode mode, int world_size);

// pure_gpu + 单卡且装不下 → kLayerStream；否则原 mode。*degraded=true 时需打 WARN。
contracts::ExecMode resolve_kimi_exec_mode(contracts::ExecMode requested, int world_size,
                                           bool* degraded = nullptr);

inline bool reject_pure_gpu_single_card(contracts::ExecMode mode, int world_size) {
  return pure_gpu_single_card_does_not_fit(mode, world_size);
}

}  // namespace llmoc::families::kimi
