// llm-on-cpu :: model/vision/image_preprocess.cpp
#include "model/vision/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "stb_image.h"

namespace llmoc::model::vision {
namespace {

int round_by_factor(int x, int factor) { return (x + factor / 2) / factor * factor; }
int floor_by_factor(int x, int factor) { return x / factor * factor; }
int ceil_by_factor(int x, int factor) { return (x + factor - 1) / factor * factor; }

void smart_resize(int h, int w, int factor, int min_pixels, int max_pixels, int& out_h, int& out_w) {
  if (h <= 0 || w <= 0) throw std::runtime_error("invalid image size");
  const double ratio = static_cast<double>(std::max(h, w)) / static_cast<double>(std::min(h, w));
  if (ratio > 200.0) throw std::runtime_error("image aspect ratio too extreme");
  int h_bar = std::max(factor, round_by_factor(h, factor));
  int w_bar = std::max(factor, round_by_factor(w, factor));
  if (static_cast<int64_t>(h_bar) * w_bar > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(h) * w / max_pixels);
    h_bar = std::max(factor, floor_by_factor(static_cast<int>(h / beta), factor));
    w_bar = std::max(factor, floor_by_factor(static_cast<int>(w / beta), factor));
  } else if (static_cast<int64_t>(h_bar) * w_bar < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(h) * w));
    h_bar = ceil_by_factor(static_cast<int>(h * beta), factor);
    w_bar = ceil_by_factor(static_cast<int>(w * beta), factor);
  }
  out_h = h_bar;
  out_w = w_bar;
}

// bilinear resize RGB uint8 → float CHW normalized
void resize_normalize(const uint8_t* src, int sh, int sw, int channels, int dh, int dw,
                      const float mean[3], const float stdv[3], std::vector<float>& chw) {
  chw.assign(static_cast<size_t>(3) * dh * dw, 0.f);
  for (int y = 0; y < dh; ++y) {
    const float fy = (dh == 1) ? 0.f : static_cast<float>(y) * (sh - 1) / static_cast<float>(dh - 1);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float wy = fy - y0;
    for (int x = 0; x < dw; ++x) {
      const float fx = (dw == 1) ? 0.f : static_cast<float>(x) * (sw - 1) / static_cast<float>(dw - 1);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, sw - 1);
      const float wx = fx - x0;
      for (int c = 0; c < 3; ++c) {
        auto sample = [&](int yy, int xx) -> float {
          if (channels >= 3)
            return src[(yy * sw + xx) * channels + c] / 255.f;
          const float g = src[yy * sw + xx] / 255.f;
          return g;
        };
        const float v = (1 - wy) * (1 - wx) * sample(y0, x0) + (1 - wy) * wx * sample(y0, x1) +
                        wy * (1 - wx) * sample(y1, x0) + wy * wx * sample(y1, x1);
        chw[static_cast<size_t>(c) * dh * dw + y * dw + x] = (v - mean[c]) / stdv[c];
      }
    }
  }
}

PreparedImage patchify(const std::vector<float>& chw, int height, int width, const ImagePrepConfig& cfg) {
  const int C = 3;
  const int ps = cfg.patch_size;
  const int ms = cfg.merge_size;
  const int tps = cfg.temporal_patch_size;
  if (height % ps || width % ps) throw std::runtime_error("size not divisible by patch_size");
  if ((height / ps) % ms || (width / ps) % ms)
    throw std::runtime_error("grid not divisible by merge_size");

  const int grid_h = height / ps;
  const int grid_w = width / ps;
  const int grid_t = 1;
  const int patch_dim = C * tps * ps * ps;
  const int seq = grid_t * grid_h * grid_w;

  PreparedImage out;
  out.grid_t = grid_t;
  out.grid_h = grid_h;
  out.grid_w = grid_w;
  out.patch_dim = patch_dim;
  out.merge_size_cache_ = ms;
  out.pixel_values.assign(static_cast<size_t>(seq) * patch_dim, 0.f);

  // patches layout (image duplicated temporally):
  // reshape [T=tps, C, gh/ms, ms, ps, gw/ms, ms, ps] then transpose (0,3,6,4,7,2,1,5,8)
  // → order: gt, gh/ms, gw/ms, ms, ms, C, tps, ps, ps
  const int gh_m = grid_h / ms;
  const int gw_m = grid_w / ms;
  int out_i = 0;
  for (int th = 0; th < gh_m; ++th) {
    for (int tw = 0; tw < gw_m; ++tw) {
      for (int mh = 0; mh < ms; ++mh) {
        for (int mw = 0; mw < ms; ++mw) {
          const int gh = th * ms + mh;
          const int gw = tw * ms + mw;
          float* dst = out.pixel_values.data() + static_cast<size_t>(out_i++) * patch_dim;
          int d = 0;
          for (int c = 0; c < C; ++c) {
            for (int t = 0; t < tps; ++t) {
              (void)t;  // duplicate frame
              for (int ph = 0; ph < ps; ++ph) {
                for (int pw = 0; pw < ps; ++pw) {
                  const int y = gh * ps + ph;
                  const int x = gw * ps + pw;
                  dst[d++] = chw[static_cast<size_t>(c) * height * width + y * width + x];
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

}  // namespace

PreparedImage prepare_image_bytes(const uint8_t* data, size_t nbytes, const ImagePrepConfig& cfg) {
  int w = 0, h = 0, comp = 0;
  unsigned char* img = stbi_load_from_memory(data, static_cast<int>(nbytes), &w, &h, &comp, 0);
  if (!img) throw std::runtime_error(std::string("stbi decode failed: ") + stbi_failure_reason());
  const int factor = cfg.patch_size * cfg.merge_size;
  int rh = 0, rw = 0;
  smart_resize(h, w, factor, cfg.min_pixels, cfg.max_pixels, rh, rw);
  std::vector<float> chw;
  resize_normalize(img, h, w, comp, rh, rw, cfg.mean, cfg.std, chw);
  stbi_image_free(img);
  return patchify(chw, rh, rw, cfg);
}

PreparedImage prepare_image_file(const std::string& path, const ImagePrepConfig& cfg) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open image: " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return prepare_image_bytes(buf.data(), buf.size(), cfg);
}

}  // namespace llmoc::model::vision
