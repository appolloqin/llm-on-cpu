// tools/glm/make_fake_glmq.cpp — tiny BF16 MoE GLMQ for local forward tests
// Layer0: sparse GQA (+ indexer); Layer1: KDA (+ conv/gates)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "glm/ops/glm_ops.h"
#include "glm/weights/glm_weight_store.h"

using namespace llmoc::glm;

static void fill_bf16(std::vector<uint8_t>& data, uint64_t& off, std::vector<GlmqTensorRec>& cat,
                      const std::string& name, const std::vector<uint32_t>& shape, uint32_t seed) {
  size_t n = 1;
  for (auto s : shape) n *= s;
  GlmqTensorRec r{};
  std::snprintf(r.name, sizeof(r.name), "%s", name.c_str());
  r.dtype = static_cast<uint16_t>(GlmqDtype::kBF16);
  r.ndim = static_cast<uint16_t>(shape.size());
  for (size_t i = 0; i < shape.size() && i < 4; ++i) r.shape[i] = shape[i];
  r.offset = off;
  r.nbytes = n * 2;
  cat.push_back(r);
  data.resize(data.size() + r.nbytes);
  auto* p = reinterpret_cast<uint16_t*>(data.data() + static_cast<size_t>(off));
  for (size_t i = 0; i < n; ++i) {
    float v = std::sin(0.01f * static_cast<float>(seed + i)) * 0.02f;
    p[i] = ops::f32_to_bf16(v);
  }
  if (name.find("layernorm") != std::string::npos || name.find("o_norm") != std::string::npos ||
      name == "norm.weight") {
    for (size_t i = 0; i < n; ++i) p[i] = ops::f32_to_bf16(1.f);
  }
  if (name.find("A_log") != std::string::npos) {
    for (size_t i = 0; i < n; ++i) p[i] = ops::f32_to_bf16(0.f);  // exp(0)=1
  }
  off += r.nbytes;
}

int main(int argc, char** argv) {
  std::string out = "models/_glm_selftest.glmq";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--out" && i + 1 < argc) out = argv[++i];
  }

  const uint32_t H = 64, L = 2, V = 128, nh = 4, nkv = 2, hd = 16, E = 4, topk = 2, I = 32;
  const uint32_t conv_k = 4, q_lora = 32, kv_lora = 16, d_idx = 16;
  GlmqFileHeader hdr{};
  hdr.quant = static_cast<uint16_t>(GlmqQuant::kBf16);
  hdr.hidden = H;
  hdr.layers = L;
  hdr.vocab = V;
  hdr.n_heads = nh;
  hdr.n_kv = nkv;
  hdr.head_dim = hd;
  hdr.n_experts = E;
  hdr.topk = topk;
  hdr.moe_inter = I;
  hdr.n_expert_groups = L * E;
  set_rms_eps(hdr, 1e-6f);

  std::vector<GlmqTensorRec> cat;
  std::vector<uint8_t> data;
  uint64_t off = 0;
  uint32_t seed = 1;

  fill_bf16(data, off, cat, "embed_tokens.weight", {V, H}, seed++);
  fill_bf16(data, off, cat, "norm.weight", {H}, seed++);
  fill_bf16(data, off, cat, "lm_head.weight", {V, H}, seed++);

  for (uint32_t layer = 0; layer < L; ++layer) {
    const std::string b = "layers." + std::to_string(layer) + ".";
    fill_bf16(data, off, cat, b + "input_layernorm.weight", {H}, seed++);
    fill_bf16(data, off, cat, b + "post_attention_layernorm.weight", {H}, seed++);

    if (layer == 0) {
      // Sparse MLA + indexer
      fill_bf16(data, off, cat, b + "self_attn.q_a_proj.weight", {q_lora, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.q_b_proj.weight", {nh * hd, q_lora}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.kv_a_proj_with_mqa.weight", {kv_lora, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.kv_b_proj.weight", {nkv * (hd + hd), kv_lora},
                seed++);
      fill_bf16(data, off, cat, b + "self_attn.indexer.weights_proj.weight", {d_idx, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.o_proj.weight", {H, nh * hd}, seed++);
    } else {
      // KDA full-ish path
      fill_bf16(data, off, cat, b + "self_attn.q_proj.weight", {nh * hd, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.k_proj.weight", {nh * hd, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.v_proj.weight", {nh * hd, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.b_proj.weight", {nh, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.q_conv1d.weight", {nh * hd, conv_k}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.k_conv1d.weight", {nh * hd, conv_k}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.v_conv1d.weight", {nh * hd, conv_k}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.f_a_proj.weight", {hd, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.f_b_proj.weight", {nh * hd, hd}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.g_a_proj.weight", {hd, H}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.g_b_proj.weight", {nh * hd, hd}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.A_log", {nh}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.dt_bias", {nh}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.o_norm.weight", {nh * hd}, seed++);
      fill_bf16(data, off, cat, b + "self_attn.o_proj.weight", {H, nh * hd}, seed++);
    }

    fill_bf16(data, off, cat, b + "mlp.gate.weight", {E, H}, seed++);
    for (uint32_t e = 0; e < E; ++e) {
      const std::string eb = b + "mlp.experts." + std::to_string(e) + ".";
      fill_bf16(data, off, cat, eb + "gate_proj.weight", {I, H}, seed++);
      fill_bf16(data, off, cat, eb + "up_proj.weight", {I, H}, seed++);
      fill_bf16(data, off, cat, eb + "down_proj.weight", {H, I}, seed++);
    }
    fill_bf16(data, off, cat, b + "mlp.shared_experts.gate_proj.weight", {I, H}, seed++);
    fill_bf16(data, off, cat, b + "mlp.shared_experts.up_proj.weight", {I, H}, seed++);
    fill_bf16(data, off, cat, b + "mlp.shared_experts.down_proj.weight", {H, I}, seed++);
  }

  if (!write_glmq_file(out, hdr, cat, data)) {
    std::cerr << "failed to write " << out << "\n";
    return 1;
  }

  const std::string meta = out + ".meta.json";
  std::ofstream mj(meta);
  mj << "{\n"
     << "  \"arch\": \"glm53_flash\",\n"
     << "  \"layer_types\": [\"deepseek_sparse_attention\", \"linear_attention\"],\n"
     << "  \"first_k_dense_replace\": 0,\n"
     << "  \"mhc\": false,\n"
     << "  \"index_topk\": 2,\n"
     << "  \"index_kpool\": 4,\n"
     << "  \"index_kpool_compress\": true,\n"
     << "  \"index_head_dim\": " << d_idx << "\n"
     << "}\n";
  mj.close();

  std::cout << "wrote " << out << " tensors=" << cat.size() << " bytes=" << data.size()
            << " H=" << H << " L=" << L << " V=" << V << " E=" << E << " + meta\n";
  return 0;
}
