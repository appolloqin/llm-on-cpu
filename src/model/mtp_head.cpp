// llm-on-cpu :: model/mtp_head.cpp
#include "model/mtp_head.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llmoc::model {
namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

}  // namespace

bool mtp_weights_present(const MtpWeightAccess& wa) {
  if (!wa.has || !wa.ctx) return false;
  return wa.has(wa.ctx, "mtp.fc.weight") && wa.has(wa.ctx, "mtp.layers.0.self_attn.q_proj.weight") &&
         wa.has(wa.ctx, "mtp.norm.weight") && wa.has(wa.ctx, "mtp.pre_fc_norm_embedding.weight") &&
         wa.has(wa.ctx, "mtp.pre_fc_norm_hidden.weight");
}

bool mtp_draft_propose(const MtpWeightAccess& wa, const std::vector<float>& last_hidden,
                       const std::vector<int32_t>& history, int draft_k,
                       std::vector<int32_t>& out, int32_t pin_first) {
  out.clear();
  if (draft_k <= 0 || history.empty()) return false;
  if (!mtp_weights_present(wa) || !wa.gemm || !wa.pass || !wa.embed) return false;
  const int H = wa.hidden;
  const int V = wa.vocab > 0 ? wa.vocab : 248320;
  if (static_cast<int>(last_hidden.size()) != H) return false;

  const int nh = wa.n_heads, nkv = wa.n_kv, hd = wa.head_dim, I = wa.intermediate;
  const int rotary_dim = static_cast<int>(hd * wa.partial_rotary) / 2 * 2;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const hal::WDtype dt = wa.pass_dt;

  SessionCache mcache;
  mcache.init(1, draft_k + 8, nkv, hd, /*n_v*/ 1, /*dk*/ 1, /*dv*/ 1, /*conv_dim*/ 1,
              /*conv_k*/ 1);

  std::vector<float> h = last_hidden;
  std::vector<float> emb(H), e_n(H), h_n(H), cat(H * 2), x(H), residual(H);
  std::vector<float> normed(H), qg(nh * hd * 2), kk(nkv * hd), vv(nkv * hd);
  std::vector<float> qq(nh * hd), gate(nh * hd), attn_heads(nh * hd), attn_out(H);
  std::vector<float> gproj(I), uproj(I), mid(I), down(H), logits(V);

  const uint16_t* n_emb = wa.pass(wa.ctx, "mtp.pre_fc_norm_embedding.weight");
  const uint16_t* n_hid = wa.pass(wa.ctx, "mtp.pre_fc_norm_hidden.weight");
  const uint16_t* n_in = wa.pass(wa.ctx, "mtp.layers.0.input_layernorm.weight");
  const uint16_t* qn = wa.pass(wa.ctx, "mtp.layers.0.self_attn.q_norm.weight");
  const uint16_t* kn = wa.pass(wa.ctx, "mtp.layers.0.self_attn.k_norm.weight");
  const uint16_t* n_post = wa.pass(wa.ctx, "mtp.layers.0.post_attention_layernorm.weight");
  const uint16_t* n_out = wa.pass(wa.ctx, "mtp.norm.weight");
  if (!n_emb || !n_hid || !n_in || !qn || !kn || !n_post || !n_out) return false;

  const char* emb_names[] = {"language_model.embed_tokens.weight", "embed_tokens.weight",
                             "embedding.weight"};
  const char* emb_w = nullptr;
  for (const char* en : emb_names) {
    if (wa.has(wa.ctx, en)) {
      emb_w = en;
      break;
    }
  }
  if (!emb_w) return false;

  std::vector<int> cand;
  const bool use_cand = wa.hint_logits && wa.embed_dot && wa.hint_top_m > 0;
  if (use_cand) {
    const int m = std::min(wa.hint_top_m, V);
    cand.resize(V);
    for (int i = 0; i < V; ++i) cand[i] = i;
    std::partial_sort(cand.begin(), cand.begin() + m, cand.end(),
                      [&](int a, int b) { return wa.hint_logits[a] > wa.hint_logits[b]; });
    cand.resize(m);
  }

  const int S = static_cast<int>(history.size());
  int32_t tok = history.back();

  auto propose_one = [&](int pos) -> int32_t {
    wa.embed(wa.ctx, tok, emb.data());
    hal::rmsnorm(emb.data(), n_emb, e_n.data(), H, wa.rms_eps, dt, wa.rms_one_plus);
    hal::rmsnorm(h.data(), n_hid, h_n.data(), H, wa.rms_eps, dt, wa.rms_one_plus);
    std::memcpy(cat.data(), e_n.data(), sizeof(float) * H);
    std::memcpy(cat.data() + H, h_n.data(), sizeof(float) * H);
    wa.gemm(wa.ctx, cat.data(), "mtp.fc.weight", x.data(), H, H * 2);

    std::memcpy(residual.data(), x.data(), sizeof(float) * H);
    hal::rmsnorm(x.data(), n_in, normed.data(), H, wa.rms_eps, dt, wa.rms_one_plus);
    wa.gemm(wa.ctx, normed.data(), "mtp.layers.0.self_attn.q_proj.weight", qg.data(), nh * hd * 2,
            H);
    wa.gemm(wa.ctx, normed.data(), "mtp.layers.0.self_attn.k_proj.weight", kk.data(), nkv * hd, H);
    wa.gemm(wa.ctx, normed.data(), "mtp.layers.0.self_attn.v_proj.weight", vv.data(), nkv * hd, H);

    auto& Lkv = mcache.layer(0);
    for (int hi = 0; hi < nh; ++hi) {
      float* qh = qq.data() + hi * hd;
      float* gh = gate.data() + hi * hd;
      const float* src = qg.data() + hi * hd * 2;
      std::memcpy(qh, src, sizeof(float) * hd);
      std::memcpy(gh, src + hd, sizeof(float) * hd);
      hal::rmsnorm(qh, qn, qh, hd, wa.rms_eps, dt, wa.rms_one_plus);
      hal::apply_rope_freqs(qh, hd, rotary_dim, pos, wa.rope_theta);
    }
    for (int hi = 0; hi < nkv; ++hi) {
      float* kh = kk.data() + hi * hd;
      float* vh = vv.data() + hi * hd;
      hal::rmsnorm(kh, kn, kh, hd, wa.rms_eps, dt, wa.rms_one_plus);
      hal::apply_rope_freqs(kh, hd, rotary_dim, pos, wa.rope_theta);
      float* kdst = Lkv.k.data() + (static_cast<size_t>(hi) * mcache.max_seq() + Lkv.seq) * hd;
      float* vdst = Lkv.v.data() + (static_cast<size_t>(hi) * mcache.max_seq() + Lkv.seq) * hd;
      std::memcpy(kdst, kh, sizeof(float) * hd);
      std::memcpy(vdst, vh, sizeof(float) * hd);
    }
    const int seq_len = Lkv.seq + 1;
    hal::attn_decode_one(qq.data(), Lkv.k.data(), Lkv.v.data(), attn_heads.data(), nh, nkv, hd,
                         seq_len, mcache.max_seq(), scale);
    Lkv.seq += 1;
    for (int i = 0; i < nh * hd; ++i) attn_heads[i] *= sigmoid(gate[i]);
    wa.gemm(wa.ctx, attn_heads.data(), "mtp.layers.0.self_attn.o_proj.weight", attn_out.data(), H,
            nh * hd);
    for (int i = 0; i < H; ++i) x[i] = residual[i] + attn_out[i];

    std::memcpy(residual.data(), x.data(), sizeof(float) * H);
    hal::rmsnorm(x.data(), n_post, normed.data(), H, wa.rms_eps, dt, wa.rms_one_plus);
    wa.gemm(wa.ctx, normed.data(), "mtp.layers.0.mlp.gate_proj.weight", gproj.data(), I, H);
    wa.gemm(wa.ctx, normed.data(), "mtp.layers.0.mlp.up_proj.weight", uproj.data(), I, H);
    hal::silu_and_mul(gproj.data(), uproj.data(), mid.data(), I);
    wa.gemm(wa.ctx, mid.data(), "mtp.layers.0.mlp.down_proj.weight", down.data(), H, I);
    for (int i = 0; i < H; ++i) x[i] = residual[i] + down[i];

    hal::rmsnorm(x.data(), n_out, h.data(), H, wa.rms_eps, dt, wa.rms_one_plus);

    int32_t best = 0;
    if (use_cand) {
      float bestv = -1e30f;
      best = cand[0];
      for (int tok_i : cand) {
        const float s = wa.embed_dot(wa.ctx, h.data(), tok_i);
        if (std::isfinite(s) && s > bestv) {
          bestv = s;
          best = tok_i;
        }
      }
    } else {
      wa.gemm(wa.ctx, h.data(), emb_w, logits.data(), V, H);
      float bestv = -1e30f;
      for (int v = 0; v < V; ++v) {
        if (std::isfinite(logits[v]) && logits[v] > bestv) {
          bestv = logits[v];
          best = v;
        }
      }
    }
    return best;
  };

  if (pin_first >= 0) {
    out.push_back(pin_first);
    tok = pin_first;
    if (draft_k == 1) return true;
    (void)propose_one(S - 1);
    for (int step = 1; step < draft_k; ++step) {
      const int pos = S - 1 + step;
      const int32_t best = propose_one(pos);
      out.push_back(best);
      tok = best;
    }
    return static_cast<int>(out.size()) == draft_k;
  }

  for (int step = 0; step < draft_k; ++step) {
    const int pos = S - 1 + step;
    const int32_t best = propose_one(pos);
    out.push_back(best);
    tok = best;
  }
  return !out.empty();
}

}  // namespace llmoc::model
