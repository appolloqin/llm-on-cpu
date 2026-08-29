// llm-on-cpu :: model/vision/qwen_vision_encoder.cpp
#include "model/vision/qwen_vision_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "hal/int4_ops.h"

namespace llmoc::model::vision {
namespace {

float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + std::tanh(c * (x + 0.044715f * x * x * x)));
}
float gelu_erf(float x) {
  return 0.5f * x * (1.f + std::erf(x * 0.7071067811865476f));
}

void layernorm(const float* x, const uint16_t* w, const uint16_t* b, float* y, int n, float eps,
               hal::WDtype dt) {
  double mean = 0.0;
  for (int i = 0; i < n; ++i) mean += x[i];
  mean /= n;
  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = x[i] - mean;
    var += d * d;
  }
  var /= n;
  const float inv = 1.f / std::sqrt(static_cast<float>(var) + eps);
  for (int i = 0; i < n; ++i) {
    const float nx = (x[i] - static_cast<float>(mean)) * inv;
    y[i] = nx * hal::load_w(w + i, dt) + hal::load_w(b + i, dt);
  }
}

void attn_full(const float* q, const float* k, const float* v, float* out, int seq, int n_heads,
               int head_dim, float scale) {
  std::vector<float> scores(static_cast<size_t>(seq) * seq);
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < seq; ++i) {
      const float* qi = q + (static_cast<size_t>(i) * n_heads + h) * head_dim;
      float* row = scores.data() + static_cast<size_t>(i) * seq;
      float m = -1e30f;
      for (int j = 0; j < seq; ++j) {
        const float* kj = k + (static_cast<size_t>(j) * n_heads + h) * head_dim;
        float s = 0.f;
        for (int d = 0; d < head_dim; ++d) s += qi[d] * kj[d];
        s *= scale;
        row[j] = s;
        if (s > m) m = s;
      }
      float sum = 0.f;
      for (int j = 0; j < seq; ++j) {
        row[j] = std::exp(row[j] - m);
        sum += row[j];
      }
      const float inv = 1.f / sum;
      for (int j = 0; j < seq; ++j) row[j] *= inv;
      float* oi = out + (static_cast<size_t>(i) * n_heads + h) * head_dim;
      std::memset(oi, 0, sizeof(float) * head_dim);
      for (int j = 0; j < seq; ++j) {
        const float* vj = v + (static_cast<size_t>(j) * n_heads + h) * head_dim;
        const float a = row[j];
        for (int d = 0; d < head_dim; ++d) oi[d] += a * vj[d];
      }
    }
  }
}

}  // namespace

void QwenVisionEncoder::load(qlwc::QlwcStore* store, const VisionConfig& cfg) {
  store_ = store;
  cfg_ = cfg;
  num_grid_per_side_ = static_cast<int>(std::sqrt(static_cast<float>(cfg_.num_position_embeddings)));
  pass_wd_ = hal::WDtype::kBF16;
  for (const auto& tm : store_->header().tensors) {
    if (tm.kind == qlwc::TensorKind::kPassthrough) {
      pass_wd_ = tm.pass_dtype == qlwc::PassDtype::kF16 ? hal::WDtype::kF16 : hal::WDtype::kBF16;
      break;
    }
  }
  if (!ready()) throw std::runtime_error("QLWC missing visual.patch_embed.proj.weight");
}

bool QwenVisionEncoder::is_int4(const std::string& name) const { return store_->is_int4(name); }

const uint16_t* QwenVisionEncoder::pass(const std::string& name) {
  return store_->get_pass(name).data;
}

void QwenVisionEncoder::gemm_w(const float* x, const std::string& wname, float* y, int M, int K) {
  if (is_int4(wname))
    hal::gemm_int4(x, store_->get_int4(wname), y);
  else
    hal::gemm_bias_free(x, pass(wname), y, M, K, pass_wd_);
}

void QwenVisionEncoder::gemm_w_bias(const float* x, const std::string& wname, const std::string& bname,
                                    float* y, int M, int K) {
  gemm_w(x, wname, y, M, K);
  const uint16_t* b = pass(bname);
  for (int m = 0; m < M; ++m) y[m] += hal::load_w(b + m, pass_wd_);
}

void QwenVisionEncoder::dequant_row(const std::string& name, int row, float* out, int K) {
  if (is_int4(name)) {
    hal::dequant_int4_row(store_->get_int4(name), row, out);
    return;
  }
  const uint16_t* p = pass(name) + static_cast<size_t>(row) * K;
  for (int i = 0; i < K; ++i) out[i] = hal::load_w(p + i, pass_wd_);
}

void QwenVisionEncoder::patch_embed(const PreparedImage& img, std::vector<float>& hidden) {
  const int seq = img.num_patches();
  const int H = cfg_.hidden;
  const int Kd = img.patch_dim;
  hidden.assign(static_cast<size_t>(seq) * H, 0.f);

  // Conv3d weight [out, in, t, h, w] flattened as [H, Kd]
  const uint16_t* W = pass("visual.patch_embed.proj.weight");
  const uint16_t* B = pass("visual.patch_embed.proj.bias");
  std::vector<float> Wf(static_cast<size_t>(H) * Kd);
  for (size_t i = 0; i < Wf.size(); ++i) Wf[i] = hal::load_w(W + i, pass_wd_);

  for (int t = 0; t < seq; ++t) {
    const float* x = img.pixel_values.data() + static_cast<size_t>(t) * Kd;
    float* y = hidden.data() + static_cast<size_t>(t) * H;
    for (int o = 0; o < H; ++o) {
      float acc = hal::load_w(B + o, pass_wd_);
      const float* wr = Wf.data() + static_cast<size_t>(o) * Kd;
      for (int k = 0; k < Kd; ++k) acc += wr[k] * x[k];
      y[o] = acc;
    }
  }
}

void QwenVisionEncoder::add_pos_embed(int grid_t, int grid_h, int grid_w,
                                     std::vector<float>& hidden) {
  const int merge = cfg_.spatial_merge;
  const int H = cfg_.hidden;
  const int side = num_grid_per_side_;
  const int seq = grid_t * grid_h * grid_w;
  if (static_cast<int>(hidden.size()) != seq * H) throw std::runtime_error("pos_embed size mismatch");

  // Build interpolated pos in raster h*w, then permute to merge order (same as fast_pos_embed)
  std::vector<float> raster(static_cast<size_t>(grid_h) * grid_w * H, 0.f);
  std::vector<float> row(H);

  auto lerp_idx = [&](int h, int w_idx, float* out) {
    const float h_idx = (grid_h == 1) ? 0.f : static_cast<float>(h) * (side - 1) / (grid_h - 1);
    const float w_f = (grid_w == 1) ? 0.f : static_cast<float>(w_idx) * (side - 1) / (grid_w - 1);
    const int h0 = static_cast<int>(h_idx);
    const int w0 = static_cast<int>(w_f);
    const int h1 = std::min(h0 + 1, side - 1);
    const int w1 = std::min(w0 + 1, side - 1);
    const float dh = h_idx - h0;
    const float dw = w_f - w0;
    const int i00 = h0 * side + w0;
    const int i01 = h0 * side + w1;
    const int i10 = h1 * side + w0;
    const int i11 = h1 * side + w1;
    std::vector<float> e00(H), e01(H), e10(H), e11(H);
    dequant_row("visual.pos_embed.weight", i00, e00.data(), H);
    dequant_row("visual.pos_embed.weight", i01, e01.data(), H);
    dequant_row("visual.pos_embed.weight", i10, e10.data(), H);
    dequant_row("visual.pos_embed.weight", i11, e11.data(), H);
    for (int d = 0; d < H; ++d) {
      out[d] = (1 - dh) * (1 - dw) * e00[d] + (1 - dh) * dw * e01[d] + dh * (1 - dw) * e10[d] +
               dh * dw * e11[d];
    }
  };

  for (int hi = 0; hi < grid_h; ++hi) {
    for (int wi = 0; wi < grid_w; ++wi) {
      lerp_idx(hi, wi, row.data());
      std::memcpy(raster.data() + (static_cast<size_t>(hi) * grid_w + wi) * H, row.data(),
                  sizeof(float) * H);
    }
  }

  // permute to merge order and add (with temporal repeat)
  const int gh_m = grid_h / merge;
  const int gw_m = grid_w / merge;
  int dst = 0;
  for (int t = 0; t < grid_t; ++t) {
    (void)t;
    for (int th = 0; th < gh_m; ++th) {
      for (int tw = 0; tw < gw_m; ++tw) {
        for (int mh = 0; mh < merge; ++mh) {
          for (int mw = 0; mw < merge; ++mw) {
            const int hi = th * merge + mh;
            const int wi = tw * merge + mw;
            const float* src = raster.data() + (static_cast<size_t>(hi) * grid_w + wi) * H;
            float* h = hidden.data() + static_cast<size_t>(dst++) * H;
            for (int d = 0; d < H; ++d) h[d] += src[d];
          }
        }
      }
    }
  }
}

void QwenVisionEncoder::build_rotary(int grid_t, int grid_h, int grid_w, std::vector<float>& cos,
                                     std::vector<float>& sin) {
  const int merge = cfg_.spatial_merge;
  const int head_dim = cfg_.hidden / cfg_.num_heads;
  const int rope_dim = head_dim / 2;  // VisionRotaryEmbedding(head_dim//2)
  const int freq_dim = rope_dim / 2;  // inv_freq length
  const int seq = grid_t * grid_h * grid_w;

  std::vector<float> inv_freq(freq_dim);
  for (int i = 0; i < freq_dim; ++i)
    inv_freq[i] = 1.f / std::pow(cfg_.rope_theta, static_cast<float>(i * 2) / rope_dim);

  // position ids in merge order (h, w)
  std::vector<int> hpos(seq), wpos(seq);
  const int gh_m = grid_h / merge;
  const int gw_m = grid_w / merge;
  int idx = 0;
  for (int t = 0; t < grid_t; ++t) {
    (void)t;
    for (int th = 0; th < gh_m; ++th) {
      for (int tw = 0; tw < gw_m; ++tw) {
        for (int mh = 0; mh < merge; ++mh) {
          for (int mw = 0; mw < merge; ++mw) {
            hpos[idx] = th * merge + mh;
            wpos[idx] = tw * merge + mw;
            ++idx;
          }
        }
      }
    }
  }

  // rotary freqs (seq, rope_dim) then cat → (seq, head_dim)
  cos.assign(static_cast<size_t>(seq) * head_dim, 0.f);
  sin.assign(static_cast<size_t>(seq) * head_dim, 0.f);
  for (int t = 0; t < seq; ++t) {
    std::vector<float> freqs(rope_dim);
    for (int i = 0; i < freq_dim; ++i) {
      freqs[i] = hpos[t] * inv_freq[i];
      freqs[i + freq_dim] = wpos[t] * inv_freq[i];
    }
    // cat (freqs, freqs) → head_dim
    for (int i = 0; i < rope_dim; ++i) {
      cos[static_cast<size_t>(t) * head_dim + i] = std::cos(freqs[i]);
      sin[static_cast<size_t>(t) * head_dim + i] = std::sin(freqs[i]);
      cos[static_cast<size_t>(t) * head_dim + rope_dim + i] = std::cos(freqs[i]);
      sin[static_cast<size_t>(t) * head_dim + rope_dim + i] = std::sin(freqs[i]);
    }
  }
}

void QwenVisionEncoder::block_forward(int layer, std::vector<float>& hidden, int seq,
                                      const float* cos, const float* sin) {
  const int H = cfg_.hidden;
  const int I = cfg_.intermediate;
  const int nh = cfg_.num_heads;
  const int hd = H / nh;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const std::string base = "visual.blocks." + std::to_string(layer) + ".";

  std::vector<float> normed(static_cast<size_t>(seq) * H);
  std::vector<float> attn_in(static_cast<size_t>(seq) * H);
  const uint16_t* n1w = pass(base + "norm1.weight");
  const uint16_t* n1b = pass(base + "norm1.bias");
  for (int t = 0; t < seq; ++t)
    layernorm(hidden.data() + t * H, n1w, n1b, normed.data() + t * H, H, cfg_.ln_eps, pass_wd_);

  // qkv
  std::vector<float> qkv(static_cast<size_t>(seq) * 3 * H);
  for (int t = 0; t < seq; ++t)
    gemm_w_bias(normed.data() + t * H, base + "attn.qkv.weight", base + "attn.qkv.bias",
                qkv.data() + t * 3 * H, 3 * H, H);

  std::vector<float> q(static_cast<size_t>(seq) * H), k(static_cast<size_t>(seq) * H),
      v(static_cast<size_t>(seq) * H);
  for (int t = 0; t < seq; ++t) {
    const float* row = qkv.data() + t * 3 * H;
    // reshape (seq, 3, nh, hd) permute → q/k/v as (seq, nh, hd) packed [seq*nh*hd]
    for (int h = 0; h < nh; ++h) {
      for (int d = 0; d < hd; ++d) {
        q[(t * nh + h) * hd + d] = row[0 * H + h * hd + d];
        k[(t * nh + h) * hd + d] = row[1 * H + h * hd + d];
        v[(t * nh + h) * hd + d] = row[2 * H + h * hd + d];
      }
    }
  }

  // apply rope: q_embed = q*cos + rotate_half(q)*sin
  for (int t = 0; t < seq; ++t) {
    for (int h = 0; h < nh; ++h) {
      float* qq = q.data() + (t * nh + h) * hd;
      float* kk = k.data() + (t * nh + h) * hd;
      const float* c = cos + static_cast<size_t>(t) * hd;
      const float* s = sin + static_cast<size_t>(t) * hd;
      const int half = hd / 2;
      std::vector<float> q_old(hd), k_old(hd);
      std::memcpy(q_old.data(), qq, sizeof(float) * hd);
      std::memcpy(k_old.data(), kk, sizeof(float) * hd);
      for (int i = 0; i < half; ++i) {
        const float rq = -q_old[i + half];
        const float rq2 = q_old[i];
        const float rk = -k_old[i + half];
        const float rk2 = k_old[i];
        qq[i] = q_old[i] * c[i] + rq * s[i];
        qq[i + half] = q_old[i + half] * c[i + half] + rq2 * s[i + half];
        kk[i] = k_old[i] * c[i] + rk * s[i];
        kk[i + half] = k_old[i + half] * c[i + half] + rk2 * s[i + half];
      }
    }
  }

  std::vector<float> attn_out(static_cast<size_t>(seq) * H);
  attn_full(q.data(), k.data(), v.data(), attn_out.data(), seq, nh, hd, scale);

  std::vector<float> proj(static_cast<size_t>(seq) * H);
  for (int t = 0; t < seq; ++t) {
    // flatten heads already in attn_out as [seq, nh, hd]
    gemm_w_bias(attn_out.data() + t * H, base + "attn.proj.weight", base + "attn.proj.bias",
                proj.data() + t * H, H, H);
    for (int d = 0; d < H; ++d) hidden[t * H + d] += proj[t * H + d];
  }

  // MLP
  const uint16_t* n2w = pass(base + "norm2.weight");
  const uint16_t* n2b = pass(base + "norm2.bias");
  for (int t = 0; t < seq; ++t)
    layernorm(hidden.data() + t * H, n2w, n2b, normed.data() + t * H, H, cfg_.ln_eps, pass_wd_);

  std::vector<float> mid(I), down(H);
  for (int t = 0; t < seq; ++t) {
    gemm_w_bias(normed.data() + t * H, base + "mlp.linear_fc1.weight", base + "mlp.linear_fc1.bias",
                mid.data(), I, H);
    for (int i = 0; i < I; ++i) mid[i] = gelu_tanh(mid[i]);
    gemm_w_bias(mid.data(), base + "mlp.linear_fc2.weight", base + "mlp.linear_fc2.bias", down.data(),
                H, I);
    for (int d = 0; d < H; ++d) hidden[t * H + d] += down[d];
  }
  (void)attn_in;
}

void QwenVisionEncoder::merger_forward(std::vector<float>& hidden, int seq, std::vector<float>& out) {
  const int H = cfg_.hidden;
  const int merge = cfg_.spatial_merge;
  const int m2 = merge * merge;
  if (seq % m2) throw std::runtime_error("vision seq not divisible by merge^2");
  const int n_out = seq / m2;
  const int mid = H * m2;  // 4096
  const int Oh = cfg_.out_hidden;

  const uint16_t* nw = pass("visual.merger.norm.weight");
  const uint16_t* nb = pass("visual.merger.norm.bias");

  out.assign(static_cast<size_t>(n_out) * Oh, 0.f);
  std::vector<float> normed(H), folded(mid), h1(mid), h2(Oh);

  for (int i = 0; i < n_out; ++i) {
    // LN each of 4 tokens, then fold
    for (int j = 0; j < m2; ++j) {
      layernorm(hidden.data() + (i * m2 + j) * H, nw, nb, normed.data(), H, cfg_.ln_eps, pass_wd_);
      std::memcpy(folded.data() + j * H, normed.data(), sizeof(float) * H);
    }
    gemm_w_bias(folded.data(), "visual.merger.linear_fc1.weight", "visual.merger.linear_fc1.bias",
                h1.data(), mid, mid);
    for (int k = 0; k < mid; ++k) h1[k] = gelu_erf(h1[k]);
    gemm_w_bias(h1.data(), "visual.merger.linear_fc2.weight", "visual.merger.linear_fc2.bias",
                h2.data(), Oh, mid);
    std::memcpy(out.data() + static_cast<size_t>(i) * Oh, h2.data(), sizeof(float) * Oh);
  }
}

void QwenVisionEncoder::encode(const PreparedImage& img, std::vector<float>& out_embeds) {
  if (!ready()) throw std::runtime_error("vision encoder not loaded");
  std::vector<float> hidden;
  patch_embed(img, hidden);
  add_pos_embed(img.grid_t, img.grid_h, img.grid_w, hidden);
  std::vector<float> cos, sin;
  build_rotary(img.grid_t, img.grid_h, img.grid_w, cos, sin);
  const int seq = img.num_patches();
  for (int L = 0; L < cfg_.depth; ++L) block_forward(L, hidden, seq, cos.data(), sin.data());
  merger_forward(hidden, seq, out_embeds);
}

}  // namespace llmoc::model::vision
