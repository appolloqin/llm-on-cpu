#pragma once
// llm-on-cpu :: model/vision/image_preprocess.h
// Qwen2VL / Qwen3.5 图像预处理：smart_resize + normalize + merge-order patchify。

#include <cstdint>
#include <string>
#include <vector>

namespace llmoc::model::vision {

struct ImagePrepConfig {
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;
  // Qwen preprocessor: shortest/longest_edge 即像素总数上下界
  // 旧默认 max=min=65536（~256²）导致截图文字不可读、模型靠先验乱编
  int min_pixels = 65536;     // ~256²
  int max_pixels = 262144;    // ~512²，CPU 默认识图平衡；可被 preprocessor 覆盖后再软顶
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float std[3] = {0.5f, 0.5f, 0.5f};
};

struct PreparedImage {
  // [seq, C * temporal * patch * patch]，merge 邻域已排好序
  std::vector<float> pixel_values;
  int grid_t = 1;
  int grid_h = 0;
  int grid_w = 0;
  int patch_dim = 0;

  int num_patches() const { return grid_t * grid_h * grid_w; }
  int num_merged_tokens() const {
    const int m2 = merge_size_cache_;
    return num_patches() / (m2 * m2);
  }
  int merge_size_cache_ = 2;
};

// 解码 PNG/JPEG（stb），失败抛异常
PreparedImage prepare_image_bytes(const uint8_t* data, size_t nbytes, const ImagePrepConfig& cfg);
PreparedImage prepare_image_file(const std::string& path, const ImagePrepConfig& cfg);

}  // namespace llmoc::model::vision
