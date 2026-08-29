#pragma once
// llm-on-cpu :: model/vision/qwen_vision_encoder.h
// Qwen3.5 visual.* 前向（无 DeepStack）：PatchEmbed → ViT → Merger → LLM hidden。

#include <string>
#include <vector>

#include "hal/cpu_ops.h"
#include "model/vision/image_preprocess.h"
#include "weights/qlwc_store.h"

namespace llmoc::model::vision {

struct VisionConfig {
  int depth = 24;
  int hidden = 1024;
  int intermediate = 4096;
  int num_heads = 16;
  int patch_size = 16;
  int temporal_patch_size = 2;
  int spatial_merge = 2;
  int out_hidden = 2560;
  int in_channels = 3;
  int num_position_embeddings = 2304;
  float ln_eps = 1e-6f;
  float rope_theta = 10000.f;
};

class QwenVisionEncoder {
 public:
  void load(qlwc::QlwcStore* store, const VisionConfig& cfg);
  bool ready() const { return store_ != nullptr && store_->has("visual.patch_embed.proj.weight"); }

  // out_embeds: [num_merged_tokens, out_hidden]
  void encode(const PreparedImage& img, std::vector<float>& out_embeds);

  const VisionConfig& cfg() const { return cfg_; }

 private:
  qlwc::QlwcStore* store_ = nullptr;
  VisionConfig cfg_;
  hal::WDtype pass_wd_ = hal::WDtype::kBF16;
  int num_grid_per_side_ = 48;

  bool is_int4(const std::string& name) const;
  const uint16_t* pass(const std::string& name);
  void gemm_w(const float* x, const std::string& wname, float* y, int M, int K);
  void gemm_w_bias(const float* x, const std::string& wname, const std::string& bname, float* y,
                   int M, int K);
  void dequant_row(const std::string& name, int row, float* out, int K);

  void patch_embed(const PreparedImage& img, std::vector<float>& hidden);
  void add_pos_embed(int grid_t, int grid_h, int grid_w, std::vector<float>& hidden);
  void build_rotary(int grid_t, int grid_h, int grid_w, std::vector<float>& cos,
                    std::vector<float>& sin);
  void block_forward(int layer, std::vector<float>& hidden, int seq, const float* cos,
                     const float* sin);
  void merger_forward(std::vector<float>& hidden, int seq, std::vector<float>& out);
};

}  // namespace llmoc::model::vision
