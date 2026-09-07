// llm-on-cpu :: tests/unit/test_gated_delta_chunk.cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hal/cpu_ops.h"
#include "test_main.h"

using namespace llmoc;

namespace {

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

TINY_TEST(GatedDelta, ChunkMatchesSerial) {
  // Heads-outer path must match token-serial (single-head loop) bit-stable within float noise.
  const int seq = 192, H = 4, dk = 32, dv = 32;
  std::vector<float> q(seq * H * dk), k(seq * H * dk), v(seq * H * dv), g(seq * H), beta(seq * H);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.01f * static_cast<float>((i * 3) % 17 - 8);
    k[i] = 0.02f * static_cast<float>((i * 5) % 13 - 6);
  }
  for (size_t i = 0; i < v.size(); ++i) v[i] = 0.03f * static_cast<float>((i * 7) % 11 - 5);
  for (int i = 0; i < seq * H; ++i) {
    g[i] = -0.05f * static_cast<float>((i % 5) + 1);
    beta[i] = 0.2f + 0.1f * static_cast<float>(i % 4);
  }

  std::vector<float> st_par(H * dk * dv, 0.f), st_ser = st_par;
  std::vector<float> o_par(seq * H * dv), o_ser(seq * H * dv);

  hal::gated_delta_chunked(q.data(), k.data(), v.data(), g.data(), beta.data(), st_par.data(),
                           o_par.data(), seq, H, dk, dv, true);

  // Serial reference: one head at a time via chunked with H'=1 views is awkward; call recurrent
  // on H=1 slices.
  for (int h = 0; h < H; ++h) {
    std::vector<float> q1(seq * dk), k1(seq * dk), v1(seq * dv), g1(seq), b1(seq);
    for (int t = 0; t < seq; ++t) {
      std::memcpy(q1.data() + t * dk, q.data() + (t * H + h) * dk, sizeof(float) * dk);
      std::memcpy(k1.data() + t * dk, k.data() + (t * H + h) * dk, sizeof(float) * dk);
      std::memcpy(v1.data() + t * dv, v.data() + (t * H + h) * dv, sizeof(float) * dv);
      g1[t] = g[t * H + h];
      b1[t] = beta[t * H + h];
    }
    std::vector<float> st1(dk * dv, 0.f), o1(seq * dv);
    hal::gated_delta_recurrent(q1.data(), k1.data(), v1.data(), g1.data(), b1.data(), st1.data(),
                               o1.data(), seq, /*n_heads=*/1, dk, dv, true);
    std::memcpy(st_ser.data() + h * dk * dv, st1.data(), sizeof(float) * dk * dv);
    for (int t = 0; t < seq; ++t)
      std::memcpy(o_ser.data() + (t * H + h) * dv, o1.data() + t * dv, sizeof(float) * dv);
  }

  const float e_out = max_abs_diff(o_par, o_ser);
  const float e_st = max_abs_diff(st_par, st_ser);
  EXPECT_TRUE(e_out < 1e-5f);
  EXPECT_TRUE(e_st < 1e-5f);
}

TINY_TEST(Attn, PrefillMatchesNaiveSmall) {
  const int seq = 48, nh = 4, nkv = 2, hd = 32;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  std::vector<float> q(seq * nh * hd), k(seq * nkv * hd), v(seq * nkv * hd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.01f * static_cast<float>((i % 9) - 4);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.02f * static_cast<float>((i % 7) - 3);
    v[i] = 0.015f * static_cast<float>((i % 5) - 2);
  }
  std::vector<float> out(seq * nh * hd), ref(seq * nh * hd);

  const int grp = nh / nkv;
  for (int tq = 0; tq < seq; ++tq) {
    for (int h = 0; h < nh; ++h) {
      const int hkv = h / grp;
      std::vector<float> scores(tq + 1);
      const float* qh = q.data() + (tq * nh + h) * hd;
      for (int tk = 0; tk <= tq; ++tk) {
        const float* kt = k.data() + (tk * nkv + hkv) * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
        scores[tk] = dot * scale;
      }
      float m = scores[0];
      for (int i = 1; i <= tq; ++i) m = std::max(m, scores[i]);
      float s = 0.f;
      for (int i = 0; i <= tq; ++i) {
        scores[i] = std::exp(scores[i] - m);
        s += scores[i];
      }
      for (int i = 0; i <= tq; ++i) scores[i] /= s;
      float* oh = ref.data() + (tq * nh + h) * hd;
      std::fill(oh, oh + hd, 0.f);
      for (int tk = 0; tk <= tq; ++tk) {
        const float* vt = v.data() + (tk * nkv + hkv) * hd;
        for (int d = 0; d < hd; ++d) oh[d] += scores[tk] * vt[d];
      }
    }
  }

  hal::attn_prefill(q.data(), k.data(), v.data(), out.data(), seq, nh, nkv, hd, scale);
  float e = 0.f;
  for (size_t i = 0; i < out.size(); ++i) e = std::max(e, std::fabs(out[i] - ref[i]));
  EXPECT_TRUE(e < 1e-4f);
}
