// llm-on-cpu :: families/deepseek_v4/ds_stub_model.cpp
#include "families/deepseek_v4/ds_stub_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "common/log.h"
#include "hal/cuda_backend.h"
#include "glm/hal/glm_nvfp4_ops.h"

namespace llmoc::families::deepseek {
namespace {

uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}

void fill_bf16(std::vector<uint16_t>& v, size_t n, float scale) {
  v.resize(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = f32_to_bf16(scale * (static_cast<float>((i * 17) % 11) - 5.f) / 5.f);
}

void rmsnorm_f(const float* x, const uint16_t* w, float* y, int n) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float inv = 1.f / std::sqrt(static_cast<float>(ss / n) + 1e-6f);
  for (int i = 0; i < n; ++i) {
    uint32_t u = static_cast<uint32_t>(w[i]) << 16;
    float wi;
    std::memcpy(&wi, &u, 4);
    y[i] = x[i] * inv * wi;
  }
}

}  // namespace

void DsStubModel::gemm_bf16(const float* x, const uint16_t* W, float* y, int M, int K) {
  if (use_gpu_ && hal::cuda::try_gemm_w16(x, W, y, M, K, false)) return;
  for (int m = 0; m < M; ++m) {
    float acc = 0.f;
    const uint16_t* row = W + static_cast<size_t>(m) * K;
    for (int k = 0; k < K; ++k) {
      uint32_t u = static_cast<uint32_t>(row[k]) << 16;
      float wk;
      std::memcpy(&wk, &u, 4);
      acc += x[k] * wk;
    }
    y[m] = acc;
  }
}

void DsStubModel::gemm_nv(const float* x, const hal::Nvfp4View& W, float* y) {
  if (use_gpu_ && hal::cuda::try_gemm_nvfp4(x, W, y)) return;
  glm::hal::gemm_nvfp4(x, W, y);
}

void DsStubModel::finish_load(contracts::ExecMode mode) {
  mode_ = mode;
  meta_.hidden = g_.hidden;
  meta_.layers = g_.layers;
  meta_.vocab = g_.vocab;
  meta_.n_kv = 1;
  meta_.head_dim = g_.d_latent;
  meta_.is_moe = true;
  meta_.kind = "deepseek_v4_stub";
  use_gpu_ = false;
  LOG_INFO("DsStub: H=%d L=%d V=%d E=%d topk=%d d_c=%d mode=%s", g_.hidden, g_.layers, g_.vocab,
           g_.n_experts, g_.top_k, g_.d_latent, contracts::mode_name(mode_));
}

static void pack_nv(DsStubModel::ExpW& e, int M, int K) {
  e.q.assign(static_cast<size_t>(M) * ((K + 1) / 2), 0x12);
  const int gs = 16;
  const int ng = (K + gs - 1) / gs;
  e.scales.assign(static_cast<size_t>(M) * ng, 0x38);
  e.view.qweight = e.q.data();
  e.view.scales_fp8 = e.scales.data();
  e.view.global_scale = 0.05f;
  e.view.M = M;
  e.view.K = K;
  e.view.group_size = gs;
}

void DsStubModel::load_synthetic(DsStubGeometry g, contracts::ExecMode mode) {
  g_ = g;
  const int H = g_.hidden, V = g_.vocab, L = g_.layers, E = g_.n_experts;
  const int dc = g_.d_latent, I = g_.intermediate;
  fill_bf16(embed_, static_cast<size_t>(V) * H, 0.02f);
  fill_bf16(lm_head_, static_cast<size_t>(V) * H, 0.02f);
  fill_bf16(final_norm_, H, 1.f);
  ln1_.assign(L, {});
  ln2_.assign(L, {});
  w_qc_.assign(L, {});
  w_out_.assign(L, {});
  w_gate_.assign(L, {});
  exp_gate_.assign(L, std::vector<ExpW>(E));
  exp_up_.assign(L, std::vector<ExpW>(E));
  exp_down_.assign(L, std::vector<ExpW>(E));
  for (int l = 0; l < L; ++l) {
    fill_bf16(ln1_[l], H, 1.f);
    fill_bf16(ln2_[l], H, 1.f);
    fill_bf16(w_qc_[l], static_cast<size_t>(dc) * H, 0.01f);
    fill_bf16(w_out_[l], static_cast<size_t>(H) * dc, 0.01f);
    fill_bf16(w_gate_[l], static_cast<size_t>(E) * H, 0.01f);
    for (int e = 0; e < E; ++e) {
      pack_nv(exp_gate_[l][e], I, H);
      pack_nv(exp_up_[l][e], I, H);
      pack_nv(exp_down_[l][e], H, I);
    }
  }
  finish_load(mode);
}

void DsStubModel::load_file(const std::string& path, contracts::ExecMode mode) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("ds stub open failed: " + path);
  char magic[4];
  in.read(magic, 4);
  if (std::memcmp(magic, "DSS1", 4) != 0 && std::memcmp(magic, "KIM1", 4) != 0)
    throw std::runtime_error("bad stub magic (want DSS1/KIM1)");
  uint32_t ver = 0;
  in.read(reinterpret_cast<char*>(&ver), 4);
  DsStubGeometry g;
  in.read(reinterpret_cast<char*>(&g.hidden), 4);
  in.read(reinterpret_cast<char*>(&g.layers), 4);
  in.read(reinterpret_cast<char*>(&g.vocab), 4);
  in.read(reinterpret_cast<char*>(&g.n_experts), 4);
  in.read(reinterpret_cast<char*>(&g.top_k), 4);
  in.read(reinterpret_cast<char*>(&g.d_latent), 4);
  in.read(reinterpret_cast<char*>(&g.intermediate), 4);
  (void)ver;
  load_synthetic(g, mode);
  LOG_INFO("DsStub: loaded header from %s (synthetic fill)", path.c_str());
}

void DsStubModel::warm_gpu_weights() {
  use_gpu_ = false;
  if (mode_ == contracts::ExecMode::kPureCpu || !hal::cuda::enabled()) return;
  use_gpu_ = true;
  int n = 0;
  const int H = g_.hidden, dc = g_.d_latent, E = g_.n_experts;
  for (int l = 0; l < g_.layers; ++l) {
    if (hal::cuda::prefetch_w16(w_qc_[l].data(), dc, H, false)) ++n;
    if (hal::cuda::prefetch_w16(w_out_[l].data(), H, dc, false)) ++n;
    if (hal::cuda::prefetch_w16(w_gate_[l].data(), E, H, false)) ++n;
    if (mode_ == contracts::ExecMode::kPureGpu) {
      for (int e = 0; e < E; ++e) {
        if (hal::cuda::prefetch_nvfp4_weight(exp_gate_[l][e].view)) ++n;
        if (hal::cuda::prefetch_nvfp4_weight(exp_up_[l][e].view)) ++n;
        if (hal::cuda::prefetch_nvfp4_weight(exp_down_[l][e].view)) ++n;
      }
    }
  }
  LOG_INFO("DsStub: warm_gpu n=%d used=%.2fGiB", n, hal::cuda::vram_used() / double(1ull << 30));
}

void DsStubModel::init_cache(model::SessionCache& cache, int max_seq) const {
  cache.init(g_.layers, max_seq, 1, g_.d_latent, 0, 0, 0, 0, 4);
}

void DsStubModel::forward(const std::vector<int32_t>& tokens, model::SessionCache& cache,
                          std::vector<float>& logits, bool is_prefill) {
  (void)is_prefill;
  if (tokens.empty()) {
    logits.assign(g_.vocab, 0.f);
    return;
  }
  const int H = g_.hidden, dc = g_.d_latent, E = g_.n_experts, I = g_.intermediate;
  const int max_seq = cache.max_seq();
  std::vector<float> h(H), n1(H), n2(H), c(dc), attn(dc), o(H), gate(E);
  std::vector<float> eg(I), eu(I), mid(I), down(H), acc(H);

  for (int32_t tok : tokens) {
    if (tok < 0 || tok >= g_.vocab) tok = 0;
    for (int i = 0; i < H; ++i) {
      uint32_t u = static_cast<uint32_t>(embed_[static_cast<size_t>(tok) * H + i]) << 16;
      std::memcpy(&h[i], &u, 4);
    }
    // use layer0 seq as global pos
    const int pos = cache.layer(0).seq;
    for (int L = 0; L < g_.layers; ++L) {
      auto& Lkv = cache.layer(L);
      rmsnorm_f(h.data(), ln1_[L].data(), n1.data(), H);
      gemm_bf16(n1.data(), w_qc_[L].data(), c.data(), dc, H);
      float* kdst = Lkv.k.data() + static_cast<size_t>(pos) * dc;
      std::memcpy(kdst, c.data(), sizeof(float) * dc);

      std::fill(attn.begin(), attn.end(), 0.f);
      const int T = pos + 1;
      float nrm = 0.f;
      for (int t = 0; t < T; ++t) {
        const float* kt = Lkv.k.data() + static_cast<size_t>(t) * dc;
        float score = 0.f;
        for (int d = 0; d < dc; ++d) score += c[d] * kt[d];
        score = std::exp(score / std::sqrt(static_cast<float>(dc)));
        nrm += score;
        for (int d = 0; d < dc; ++d) attn[d] += score * kt[d];
      }
      if (nrm > 0) for (int d = 0; d < dc; ++d) attn[d] /= nrm;
      gemm_bf16(attn.data(), w_out_[L].data(), o.data(), H, dc);
      for (int i = 0; i < H; ++i) h[i] += o[i];

      rmsnorm_f(h.data(), ln2_[L].data(), n2.data(), H);
      gemm_bf16(n2.data(), w_gate_[L].data(), gate.data(), E, H);
      std::vector<int> idx(E);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + g_.top_k, idx.end(),
                        [&](int a, int b) { return gate[a] > gate[b]; });
      float wsum = 0.f;
      for (int i = 0; i < g_.top_k; ++i) wsum += std::exp(gate[idx[i]]);
      std::fill(acc.begin(), acc.end(), 0.f);
      for (int i = 0; i < g_.top_k; ++i) {
        const int e = idx[i];
        const float ww = std::exp(gate[e]) / (wsum > 0 ? wsum : 1.f);
        gemm_nv(n2.data(), exp_gate_[L][e].view, eg.data());
        gemm_nv(n2.data(), exp_up_[L][e].view, eu.data());
        for (int j = 0; j < I; ++j) {
          const float silu = eg[j] / (1.f + std::exp(-eg[j]));
          mid[j] = silu * eu[j];
        }
        gemm_nv(mid.data(), exp_down_[L][e].view, down.data());
        for (int j = 0; j < H; ++j) acc[j] += ww * down[j];
      }
      for (int i = 0; i < H; ++i) h[i] += acc[i];
      Lkv.seq = pos + 1;
      (void)max_seq;
    }
  }

  rmsnorm_f(h.data(), final_norm_.data(), n1.data(), H);
  logits.resize(g_.vocab);
  gemm_bf16(n1.data(), lm_head_.data(), logits.data(), g_.vocab, H);
}

void write_fake_dskq(const std::string& path, DsStubGeometry g) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("write_fake_dskq failed");
  out.write("DSS1", 4);
  uint32_t ver = 1;
  out.write(reinterpret_cast<const char*>(&ver), 4);
  out.write(reinterpret_cast<const char*>(&g.hidden), 4);
  out.write(reinterpret_cast<const char*>(&g.layers), 4);
  out.write(reinterpret_cast<const char*>(&g.vocab), 4);
  out.write(reinterpret_cast<const char*>(&g.n_experts), 4);
  out.write(reinterpret_cast<const char*>(&g.top_k), 4);
  out.write(reinterpret_cast<const char*>(&g.d_latent), 4);
  out.write(reinterpret_cast<const char*>(&g.intermediate), 4);
}

}  // namespace llmoc::families::deepseek
