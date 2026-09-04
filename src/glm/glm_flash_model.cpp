// llm-on-cpu :: glm/glm_flash_model.cpp
#include "glm/glm_flash_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "common/log.h"
#include "exec/factory.h"
#include "exec/gpu/exec_gpu.h"
#include "exec/nccl_probe.h"
#include "families/family_packs.h"
#include "glm/hal/glm_awq_int4_ops.h"
#include "glm/hal/glm_cuda_ops.h"
#include "glm/hal/glm_nvfp4_ops.h"
#include "glm/ops/glm_ops.h"
#include "hal/cuda_backend.h"
#include "sched/mode_controller.h"
#include "sched/placement_planner.h"

namespace llmoc::glm {
namespace {

std::string Lname(int layer, const char* suffix) {
  return "layers." + std::to_string(layer) + "." + suffix;
}

std::vector<std::string> extract_string_array(const std::string& json, const char* key) {
  std::vector<std::string> out;
  const std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return out;
  pos = json.find('[', pos);
  if (pos == std::string::npos) return out;
  auto end = json.find(']', pos);
  if (end == std::string::npos) return out;
  size_t i = pos + 1;
  while (i < end) {
    while (i < end && (json[i] == ' ' || json[i] == ',' || json[i] == '\n' || json[i] == '\r')) ++i;
    if (i >= end || json[i] != '"') break;
    ++i;
    size_t j = i;
    while (j < end && json[j] != '"') ++j;
    if (j > i) out.emplace_back(json.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

int extract_int_field(const std::string& json, const char* key, int def) {
  const std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return def;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  try {
    return std::stoi(json.substr(pos));
  } catch (...) {
    return def;
  }
}

bool extract_bool_field(const std::string& json, const char* key, bool def) {
  const std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return def;
  const auto rest = json.substr(pos + 1, 16);
  if (rest.find("true") != std::string::npos) return true;
  if (rest.find("false") != std::string::npos) return false;
  return def;
}

float sigmoid(float x) {
  if (x > 20.f) return 1.f;
  if (x < -20.f) return 0.f;
  return 1.f / (1.f + std::exp(-x));
}

}  // namespace

void GlmFlashModel::apply_geometry_from_header() {
  const auto& h = store_.header();
  H_ = static_cast<int>(h.hidden);
  L_ = static_cast<int>(h.layers);
  V_ = static_cast<int>(h.vocab);
  nh_ = static_cast<int>(h.n_heads);
  nkv_ = static_cast<int>(h.n_kv);
  hd_ = static_cast<int>(h.head_dim);
  E_ = static_cast<int>(h.n_experts);
  topk_ = static_cast<int>(h.topk);
  I_ = static_cast<int>(h.moe_inter);
  rms_eps_ = rms_eps_from_hdr(h);
  if (H_ <= 0 || L_ <= 0 || V_ <= 0) throw std::runtime_error("glm: invalid geometry in GLMQ");
  meta_.kind = "glm53_flash";
  meta_.is_moe = true;
  meta_.hidden = H_;
  meta_.layers = L_;
  meta_.vocab = V_;
  meta_.n_kv = nkv_;
  meta_.head_dim = hd_;
  meta_.linear_num_v = nh_;
  meta_.linear_dk = hd_;
  meta_.linear_dv = hd_;
  meta_.conv_dim = nh_ * hd_ * 3;
  meta_.conv_k = 4;
  qk_dim_ = hd_;
  v_dim_ = hd_;
}

void GlmFlashModel::load_meta_sidecar(const std::string& glmq_path) {
  layer_types_.clear();
  first_k_dense_ = 0;
  mhc_ = false;
  hc_mult_ = 1;
  index_topk_ = 0;
  index_dim_ = 0;
  index_kpool_ = 4;
  index_kpool_compress_ = true;
  const std::string meta_path = glmq_path + ".meta.json";
  std::ifstream in(meta_path);
  if (!in) return;
  std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  layer_types_ = extract_string_array(json, "layer_types");
  first_k_dense_ = extract_int_field(json, "first_k_dense_replace", 0);
  mhc_ = extract_bool_field(json, "mhc", false);
  hc_mult_ = extract_int_field(json, "hc_mult", mhc_ ? 4 : 1);
  if (hc_mult_ < 1) hc_mult_ = 1;
  index_topk_ = extract_int_field(json, "index_topk", 0);
  index_dim_ = extract_int_field(json, "index_head_dim", 0);
  index_kpool_ = extract_int_field(json, "index_kpool", 4);
  if (index_kpool_ < 1) index_kpool_ = 1;
  index_kpool_compress_ = extract_bool_field(json, "index_kpool_compress", true);
  const int qk = extract_int_field(json, "qk_nope_head_dim", 0);
  const int vd = extract_int_field(json, "v_head_dim", 0);
  if (qk > 0) qk_dim_ = qk;
  if (vd > 0) v_dim_ = vd;
  // KV cache / GQA path uses a common head_dim; prefer max for buffer sizing
  if (qk_dim_ != hd_ || v_dim_ != hd_) {
    hd_ = std::max(qk_dim_, v_dim_);
    meta_.head_dim = hd_;
    meta_.linear_dk = hd_;
    meta_.linear_dv = hd_;
  }
  LOG_INFO(
      "glm: meta first_k_dense=%d mhc=%d hc=%d index_topk=%d kpool=%d qk=%d v=%d layers_types=%zu",
      first_k_dense_, mhc_ ? 1 : 0, hc_mult_, index_topk_, index_kpool_, qk_dim_, v_dim_,
      layer_types_.size());
}

void GlmFlashModel::warm_gpu_attn_weights() {
  use_gpu_gemm_ = false;
  if (mode_ == ExecMode::kPureCpu || !device_ || !device_->caps().has_cuda) return;
  auto& cuda = hal::GlmCudaContext::instance();
  if (!cuda.available()) return;
  use_gpu_gemm_ = true;
  int n_up = 0, n_q = 0;
  auto try_up_bf16 = [&](const std::string& name) {
    const auto* t = store_.find(name);
    if (!t || t->dtype != static_cast<uint16_t>(GlmqDtype::kBF16)) return;
    if (t->ndim < 2 || t->shape[0] == 0 || t->shape[1] == 0) return;
    const bool expert = name.find(".mlp.experts.") != std::string::npos;
    if (mode_ == ExecMode::kHybridGpu && expert) return;
    const int M = static_cast<int>(t->shape[0]), K = static_cast<int>(t->shape[1]);
    if (llmoc::hal::cuda::prefetch_w16(store_.bf16(name), M, K, false)) ++n_up;
  };
  auto try_up_quant = [&](const std::string& name) {
    if (mode_ != ExecMode::kPureGpu) return;
    hal::AwqView awq;
    if (store_.awq_view(name, awq)) {
      if (llmoc::hal::cuda::prefetch_awq_weight(awq)) ++n_q;
      return;
    }
    hal::Nvfp4View nv;
    if (store_.nvfp4_view(name, nv)) {
      if (llmoc::hal::cuda::prefetch_nvfp4_weight(nv)) ++n_q;
    }
  };
  try_up_bf16("embed_tokens.weight");
  try_up_bf16("lm_head.weight");
  try_up_bf16("norm.weight");
  for (int L = 0; L < L_; ++L) {
    const std::string b = "layers." + std::to_string(L) + ".";
    for (const char* s : {"input_layernorm.weight", "post_attention_layernorm.weight",
                          "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                          "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight",
                          "self_attn.b_proj.weight", "mlp.gate.weight", "mlp.gate_proj.weight",
                          "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
      try_up_bf16(b + s);
      try_up_quant(b + s);
    }
    if (mode_ == ExecMode::kPureGpu && E_ > 0) {
      for (int e = 0; e < E_; ++e) {
        const std::string eb = b + "mlp.experts." + std::to_string(e) + ".";
        try_up_bf16(eb + "gate_proj.weight");
        try_up_bf16(eb + "up_proj.weight");
        try_up_bf16(eb + "down_proj.weight");
        try_up_quant(eb + "gate_proj.weight");
        try_up_quant(eb + "up_proj.weight");
        try_up_quant(eb + "down_proj.weight");
      }
    }
  }
  LOG_INFO("glm: GPU warmed %d BF16 + %d quant tensors (%s)", n_up, n_q,
           GlmEngineConfig::mode_name(mode_));
  hal::log_cuda_status();
}

void GlmFlashModel::bind_expert_hosts() {
  if (!exec_ || !exec_->caps().experts_on_gpu) return;
  auto* eg = dynamic_cast<exec::gpu::ExpertRuntimeGpu*>(exec_->experts());
  if (!eg) return;
  int n_bound = 0;
  for (int L = 0; L < L_; ++L) {
    for (int e = 0; e < E_; ++e) {
      if (!eg->owns({L, e})) continue;
      const std::string base =
          "layers." + std::to_string(L) + ".mlp.experts." + std::to_string(e) + ".gate_proj.weight";
      const auto* t = store_.find(base);
      if (!t) continue;
      // Bind packed host bytes for slot H2D (BF16 raw or quantized blob).
      const void* host = nullptr;
      size_t nbytes = 0;
      if (t->dtype == static_cast<uint16_t>(GlmqDtype::kBF16)) {
        host = store_.bf16(base);
        nbytes = sizeof(uint16_t) * t->shape[0] * t->shape[1];
      } else {
        // Quantized: bind tensor storage if contiguous view available via find payload
        continue;
      }
      if (host && nbytes) {
        eg->bind_host({L, e}, host, nbytes);
        ++n_bound;
      }
    }
  }
  LOG_INFO("glm: bound %d expert host blobs for GPU residency", n_bound);
}

void GlmFlashModel::setup_exec_backend(const GlmEngineConfig& cfg) {
  contracts::ExecMode em = contracts::ExecMode::kPureCpu;
  if (mode_ == ExecMode::kHybridGpu) em = contracts::ExecMode::kHybridGpu;
  else if (mode_ == ExecMode::kPureGpu) em = contracts::ExecMode::kPureGpu;

  contracts::DeviceMeshSpec spec;
  spec.ids = {0};
  spec.strategy = contracts::MeshStrategy::kAuto;
  spec.require_nccl = true;
  spec.has_moe = true;

  contracts::DeviceMesh mesh;
  std::string mesh_err;
  const int ngpu = llmoc::hal::cuda::device_count();
  if (!sched::resolve_mesh_for_mode(em, spec, ngpu > 0 ? ngpu : 0, exec::nccl_available(), true,
                                    &mesh, &mesh_err)) {
    if (em == contracts::ExecMode::kPureCpu) {
      mesh.ids = {0};
      mesh.world_size = mesh.ep_size = mesh.tp_size = 1;
    } else {
      throw std::runtime_error(mesh_err.empty() ? "glm mesh resolve failed" : mesh_err);
    }
  }

  exec::MakeExecOptions opt;
  opt.mesh = mesh;
  opt.n_experts_hint = E_;
  opt.vram_budget_per_rank = static_cast<size_t>(cfg.gpu_vram_gb * (1ull << 30));
  std::string err;
  exec_ = exec::make_exec(em, opt, &err);
  if (!exec_) throw std::runtime_error(err.empty() ? "glm make_exec failed" : err);

  families::Glm53Pack pack;
  uint64_t expert_bytes = 64ull << 20;
  if (E_ > 0 && I_ > 0 && H_ > 0) {
    // gate+up+down BF16 estimate; NVFP4 ~1/4
    expert_bytes = 3ull * static_cast<uint64_t>(I_) * H_ * 2;
    if (quant_ == QuantKind::kNvfp4 || quant_ == QuantKind::kAwqInt4) expert_bytes /= 4;
  }
  pack.set_geometry(L_, E_, topk_, expert_bytes, 128ull << 20);

  sched::PlacementPlanner::Config pcfg;
  pcfg.vram_bytes = opt.vram_budget_per_rank;
  pcfg.dram_bytes = static_cast<uint64_t>(cfg.dram_hot_gb * (1ull << 30));
  pcfg.strict_vram = (em == contracts::ExecMode::kPureGpu);
  auto plan = sched::PlacementPlanner::solve_active(em, pcfg, mesh, pack.active_profile());
  if (!plan.ok && em == contracts::ExecMode::kPureGpu) {
    throw std::runtime_error(plan.error);
  }
  exec_->configure(plan);
  LOG_INFO("glm exec: %s caps experts_gpu=%d attn_gpu=%d | %s", mesh.summary().c_str(),
           exec_->caps().experts_on_gpu ? 1 : 0, exec_->caps().attn_on_gpu ? 1 : 0,
           plan.summary.c_str());
}

void GlmFlashModel::load(const GlmEngineConfig& cfg) {
  cfg_ = cfg;
  quant_ = cfg.quant;
  bool degraded = false;
  device_ = make_device(cfg.mode, &degraded);
  mode_ = degraded ? ExecMode::kPureCpu : cfg.mode;
  device_->configure_memory(cfg.dram_hot_gb, cfg.gpu_vram_gb);

  meta_.kind = "glm53_flash";
  meta_.is_moe = true;

  GlmExpertPrefetch::Config pc;
  pc.slot_bytes = static_cast<size_t>(cfg.prefetch_buf_gb * (1ull << 30) / 4);
  if (pc.slot_bytes < (8ull << 20)) pc.slot_bytes = 64ull << 20;
  prefetch_.configure(pc);

  load_error_.clear();
  try {
    store_.open(cfg.model_path, cfg.quant);
    apply_geometry_from_header();
    load_meta_sidecar(cfg.model_path);
    quant_ = store_.quant();
    weights_ready_ = true;
    setup_exec_backend(cfg);
    warm_gpu_attn_weights();
    bind_expert_hosts();
  } catch (const std::exception& e) {
    weights_ready_ = false;
    load_error_ = e.what();
    meta_.hidden = 4096;
    meta_.layers = 45;
    meta_.vocab = 154880;
    meta_.n_kv = 64;
    meta_.head_dim = 256;
    LOG_WARN("glm: weights not loaded — %s", e.what());
    LOG_WARN("glm: expected path=%s (run download_glm / tools/glm; check file exists)",
             cfg.model_path.c_str());
    try {
      setup_exec_backend(cfg);
    } catch (...) {
    }
  }

  LOG_INFO("arch=glm mode=%s quant=%s device=%s gpu_gemm=%d H=%d L=%d V=%d E=%d weights=%s",
           GlmEngineConfig::mode_name(mode_), GlmEngineConfig::quant_name(store_.is_open()
                                                                             ? store_.quant()
                                                                             : quant_),
           device_->name(), use_gpu_gemm_ ? 1 : 0, meta_.hidden, meta_.layers, meta_.vocab, E_,
           weights_ready_ ? "yes" : "no");
}

void GlmFlashModel::load_strict(const GlmEngineConfig& cfg) {
  load(cfg);
  if (!weights_ready_)
    throw std::runtime_error(std::string("glm: load_strict failed") +
                             (load_error_.empty() ? "" : (": " + load_error_)));
}

void GlmFlashModel::init_cache(model::SessionCache& cache, int max_seq) const {
  cache.init(meta_.layers, max_seq, meta_.n_kv, meta_.head_dim, meta_.linear_num_v,
             meta_.linear_dk, meta_.linear_dv, meta_.conv_dim, meta_.conv_k);
}

void GlmFlashModel::gemm_bf16_named(const float* x, const std::string& wname, float* y, int M,
                                    int K) {
  const uint16_t* W = store_.require_bf16(wname);
  if (use_gpu_gemm_ &&
      hal::GlmCudaContext::instance().gemm_bf16(wname, x, W, y, M, K)) {
    return;
  }
  ops::gemm_bf16(x, W, y, M, K);
}

void GlmFlashModel::gemm_linear(const float* x, const std::string& wname, float* y, int M, int K) {
  const bool expert = wname.find(".mlp.experts.") != std::string::npos;
  const bool allow_quant_gpu = use_gpu_gemm_ && !(mode_ == ExecMode::kHybridGpu && expert);
  hal::AwqView awq;
  if (store_.awq_view(wname, awq)) {
    if (awq.M != M || awq.K != K) throw std::runtime_error("glm: AWQ shape mismatch " + wname);
    if (allow_quant_gpu && llmoc::hal::cuda::try_gemm_awq(x, awq, y)) return;
    hal::gemm_awq_int4(x, awq, y);
    return;
  }
  hal::Nvfp4View nv;
  if (store_.nvfp4_view(wname, nv)) {
    if (nv.M != M || nv.K != K) throw std::runtime_error("glm: NVFP4 shape mismatch " + wname);
    if (allow_quant_gpu && llmoc::hal::cuda::try_gemm_nvfp4(x, nv, y)) return;
    hal::gemm_nvfp4(x, nv, y);
    return;
  }
  gemm_bf16_named(x, wname, y, M, K);
}

void GlmFlashModel::dense_ffn(int layer, float* x) {
  const int H = H_;
  int I = I_;
  if (const auto* t = store_.find(Lname(layer, "mlp.gate_proj.weight"))) {
    if (t->shape[0] > 0) I = static_cast<int>(t->shape[0]);
  }
  std::vector<float> g(I), u(I), mid(I), down(H);
  gemm_linear(x, Lname(layer, "mlp.gate_proj.weight"), g.data(), I, H);
  gemm_linear(x, Lname(layer, "mlp.up_proj.weight"), u.data(), I, H);
  ops::silu_mul(g.data(), u.data(), mid.data(), I);
  gemm_linear(mid.data(), Lname(layer, "mlp.down_proj.weight"), down.data(), H, I);
  std::memcpy(x, down.data(), sizeof(float) * H);
}

void GlmFlashModel::moe_ffn(int layer, float* x) {
  const int H = H_, E = E_, K = topk_, I = I_;
  std::vector<float> logits(E);
  gemm_bf16_named(x, Lname(layer, "mlp.gate.weight"), logits.data(), E, H);
  ops::softmax(logits.data(), E);
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + K, order.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  double wsum = 0.0;
  for (int i = 0; i < K; ++i) wsum += logits[order[i]];
  std::vector<float> acc(H, 0.f);
  std::vector<float> g(I), u(I), mid(I), down(H);
  std::vector<int> ids(K);
  for (int i = 0; i < K; ++i) ids[i] = order[i];
  prefetch_.plan(layer, ids);

  std::vector<contracts::ExpertId> eids(K);
  std::vector<contracts::BlockHandle> handles(K);
  for (int i = 0; i < K; ++i) eids[i] = {layer, order[i]};
  if (exec_ && exec_->experts()) {
    exec_->experts()->prefetch(layer, eids.data(), K);
    exec_->experts()->pin(layer, eids.data(), K, handles.data());
  }

  for (int i = 0; i < K; ++i) {
    const int e = order[i];
    const float ww = static_cast<float>(logits[e] / wsum);
    const std::string base =
        "layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) + ".";
    gemm_linear(x, base + "gate_proj.weight", g.data(), I, H);
    gemm_linear(x, base + "up_proj.weight", u.data(), I, H);
    ops::silu_mul(g.data(), u.data(), mid.data(), I);
    gemm_linear(mid.data(), base + "down_proj.weight", down.data(), H, I);
    for (int d = 0; d < H; ++d) acc[d] += ww * down[d];
  }
  if (store_.find(Lname(layer, "mlp.shared_experts.gate_proj.weight"))) {
    gemm_linear(x, Lname(layer, "mlp.shared_experts.gate_proj.weight"), g.data(), I, H);
    gemm_linear(x, Lname(layer, "mlp.shared_experts.up_proj.weight"), u.data(), I, H);
    ops::silu_mul(g.data(), u.data(), mid.data(), I);
    gemm_linear(mid.data(), Lname(layer, "mlp.shared_experts.down_proj.weight"), down.data(), H, I);
    for (int d = 0; d < H; ++d) acc[d] += down[d];
  }
  if (exec_ && exec_->experts()) exec_->experts()->release(handles.data(), K);
  prefetch_.release(layer);
  std::memcpy(x, acc.data(), sizeof(float) * H);
}

void GlmFlashModel::attn_linear_kda(int layer, const float* normed, float* oproj,
                                    model::SessionCache& cache) {
  const int H = H_, nh = nh_, hd = hd_;
  const int conv_k = std::max(1, meta_.conv_k);
  auto& Lkv = cache.layer(layer);
  if (Lkv.linear.recurrent.size() < static_cast<size_t>(nh) * hd * hd) {
    Lkv.linear.recurrent.assign(static_cast<size_t>(nh) * hd * hd, 0.f);
  }
  const int conv_dim = nh * hd * 3;
  if (Lkv.linear.conv.size() < static_cast<size_t>(conv_dim) * conv_k) {
    Lkv.linear.conv.assign(static_cast<size_t>(conv_dim) * conv_k, 0.f);
  }

  std::vector<float> q(nh * hd), k(nh * hd), v(nh * hd), core(nh * hd);
  std::vector<float> g(nh, -0.1f), beta(nh, 1.f);

  auto project_qk_like = [&](const char* name, float* dst, int want_rows) {
    const auto* W = store_.find(Lname(layer, name));
    if (!W) return false;
    const int rows = static_cast<int>(W->shape[0]);
    if (rows == want_rows) {
      gemm_bf16_named(normed, Lname(layer, name), dst, want_rows, H);
      return true;
    }
    std::vector<float> small(rows);
    gemm_bf16_named(normed, Lname(layer, name), small.data(), rows, H);
    const int n_src = std::max(1, rows / std::max(hd, 1));
    const int rep = std::max(1, want_rows / hd / n_src);
    for (int h = 0; h < want_rows / hd; ++h)
      std::memcpy(dst + h * hd, small.data() + (h / rep) * hd, sizeof(float) * hd);
    return true;
  };

  // Fused path: self_attn.fused_qkvbfg_a_proj.weight → [3*nh*hd + nh + 2*hd, H]
  if (store_.find(Lname(layer, "self_attn.fused_qkvbfg_a_proj.weight"))) {
    const auto* Wf = store_.find(Lname(layer, "self_attn.fused_qkvbfg_a_proj.weight"));
    const int rows = static_cast<int>(Wf->shape[0]);
    std::vector<float> fused(rows);
    gemm_bf16_named(normed, Lname(layer, "self_attn.fused_qkvbfg_a_proj.weight"), fused.data(),
                    rows, H);
    const int qkv = 3 * nh * hd;
    if (rows >= qkv) {
      std::memcpy(q.data(), fused.data(), sizeof(float) * nh * hd);
      std::memcpy(k.data(), fused.data() + nh * hd, sizeof(float) * nh * hd);
      std::memcpy(v.data(), fused.data() + 2 * nh * hd, sizeof(float) * nh * hd);
    }
    if (rows >= qkv + nh) {
      for (int i = 0; i < nh; ++i) beta[i] = sigmoid(fused[qkv + i]);
    }
  } else if (!project_qk_like("self_attn.q_proj.weight", q.data(), nh * hd)) {
    std::fill(oproj, oproj + H, 0.f);
    Lkv.linear.has_state = true;
    return;
  } else {
    project_qk_like("self_attn.k_proj.weight", k.data(), nh * hd);
    project_qk_like("self_attn.v_proj.weight", v.data(), nh * hd);
    if (store_.find(Lname(layer, "self_attn.b_proj.weight"))) {
      std::vector<float> braw(nh);
      gemm_bf16_named(normed, Lname(layer, "self_attn.b_proj.weight"), braw.data(), nh, H);
      for (int i = 0; i < nh; ++i) beta[i] = sigmoid(braw[i]);
    }
  }

  // Short conv on q/k/v (separate or fused qkv_conv1d)
  auto apply_conv = [&](float* xh, const char* wname, int state_off) {
    if (!store_.find(Lname(layer, wname))) return;
    const auto* Wc = store_.find(Lname(layer, wname));
    const int C = static_cast<int>(Wc->shape[0]);
    int K = conv_k;
    if (Wc->ndim >= 2 && Wc->shape[1] > 0) K = static_cast<int>(Wc->shape[1]);
    if (Wc->ndim >= 3 && Wc->shape[2] > 0) K = static_cast<int>(Wc->shape[2]);
    std::vector<float> outc(C);
    ops::short_conv1d_step(xh, store_.require_bf16(Lname(layer, wname)),
                           Lkv.linear.conv.data() + state_off, outc.data(), C, K);
    std::memcpy(xh, outc.data(), sizeof(float) * std::min(C, nh * hd));
  };
  if (store_.find(Lname(layer, "self_attn.qkv_conv1d.weight"))) {
    std::vector<float> qkv(nh * hd * 3);
    std::memcpy(qkv.data(), q.data(), sizeof(float) * nh * hd);
    std::memcpy(qkv.data() + nh * hd, k.data(), sizeof(float) * nh * hd);
    std::memcpy(qkv.data() + 2 * nh * hd, v.data(), sizeof(float) * nh * hd);
    apply_conv(qkv.data(), "self_attn.qkv_conv1d.weight", 0);
    std::memcpy(q.data(), qkv.data(), sizeof(float) * nh * hd);
    std::memcpy(k.data(), qkv.data() + nh * hd, sizeof(float) * nh * hd);
    std::memcpy(v.data(), qkv.data() + 2 * nh * hd, sizeof(float) * nh * hd);
  } else {
    apply_conv(q.data(), "self_attn.q_conv1d.weight", 0);
    apply_conv(k.data(), "self_attn.k_conv1d.weight", nh * hd * conv_k);
    apply_conv(v.data(), "self_attn.v_conv1d.weight", 2 * nh * hd * conv_k);
  }

  // Forget gate → per-head decay g (Kimi-style with optional A_log/dt_bias)
  std::vector<float> forget(nh * hd, 0.f);
  if (store_.find(Lname(layer, "self_attn.f_a_proj.weight")) &&
      store_.find(Lname(layer, "self_attn.f_b_proj.weight"))) {
    const auto* fa = store_.find(Lname(layer, "self_attn.f_a_proj.weight"));
    const int fa_out = static_cast<int>(fa->shape[0]);
    std::vector<float> mid(fa_out);
    gemm_bf16_named(normed, Lname(layer, "self_attn.f_a_proj.weight"), mid.data(), fa_out, H);
    const auto* fb = store_.find(Lname(layer, "self_attn.f_b_proj.weight"));
    const int fb_out = static_cast<int>(fb->shape[0]);
    forget.assign(fb_out, 0.f);
    gemm_bf16_named(mid.data(), Lname(layer, "self_attn.f_b_proj.weight"), forget.data(), fb_out,
                    fa_out);
  }
  const uint16_t* A_log = store_.bf16(Lname(layer, "self_attn.A_log"));
  const uint16_t* dt_bias = store_.bf16(Lname(layer, "self_attn.dt_bias"));
  for (int h = 0; h < nh; ++h) {
    float a = 0.f;
    if (!forget.empty() && static_cast<int>(forget.size()) >= (h + 1) * hd) {
      double s = 0.0;
      for (int d = 0; d < hd; ++d) s += forget[h * hd + d];
      a = static_cast<float>(s / hd);
    } else if (!forget.empty() && static_cast<int>(forget.size()) == nh) {
      a = forget[h];
    }
    if (dt_bias) a += ops::bf16_to_f32(dt_bias[h]);
    float A = 1.f;
    if (A_log) {
      A = std::exp(ops::bf16_to_f32(A_log[h]));
      if (!std::isfinite(A) || A > 1e4f) A = 1e4f;
      if (A < 1e-6f) A = 1e-6f;
    }
    if (A_log || !forget.empty()) g[h] = -A * ops::softplus(a);
  }

  ops::kda_gated_delta_step(q.data(), k.data(), v.data(), g.data(), beta.data(),
                            Lkv.linear.recurrent.data(), core.data(), nh, hd, hd);
  Lkv.linear.has_state = true;

  // o_norm gate from g_a/g_b
  std::vector<float> gated(nh * hd);
  if (store_.find(Lname(layer, "self_attn.g_a_proj.weight")) &&
      store_.find(Lname(layer, "self_attn.g_b_proj.weight"))) {
    const auto* ga = store_.find(Lname(layer, "self_attn.g_a_proj.weight"));
    const int ga_out = static_cast<int>(ga->shape[0]);
    std::vector<float> mid(ga_out);
    gemm_bf16_named(normed, Lname(layer, "self_attn.g_a_proj.weight"), mid.data(), ga_out, H);
    const auto* gb = store_.find(Lname(layer, "self_attn.g_b_proj.weight"));
    const int gb_out = static_cast<int>(gb->shape[0]);
    std::vector<float> gproj(gb_out);
    gemm_bf16_named(mid.data(), Lname(layer, "self_attn.g_b_proj.weight"), gproj.data(), gb_out,
                    ga_out);
    if (gb_out < nh * hd) gproj.resize(nh * hd, 0.f);
    const uint16_t* on = store_.bf16(Lname(layer, "self_attn.o_norm.weight"));
    ops::rms_norm_gated(core.data(), gproj.data(), on, gated.data(), nh, hd, rms_eps_);
  } else {
    gated = core;
  }

  if (store_.find(Lname(layer, "self_attn.o_proj.weight"))) {
    gemm_bf16_named(gated.data(), Lname(layer, "self_attn.o_proj.weight"), oproj, H, nh * hd);
  } else {
    std::fill(oproj, oproj + H, 0.f);
    for (int i = 0; i < H && i < nh * hd; ++i) oproj[i] = gated[i];
  }
}

void GlmFlashModel::attn_sparse_mla_or_gqa(int layer, const float* normed, float* oproj,
                                          model::SessionCache& cache, int pos) {
  const int H = H_, nh = nh_, nkv = nkv_, hd = hd_;
  const int max_seq = cache.max_seq();
  std::vector<float> q(nh * hd), k(nkv * hd), v(nkv * hd), attn_out(nh * hd);

  if (store_.find(Lname(layer, "self_attn.q_a_proj.weight")) &&
      store_.find(Lname(layer, "self_attn.q_b_proj.weight")) &&
      store_.find(Lname(layer, "self_attn.kv_a_proj_with_mqa.weight")) &&
      store_.find(Lname(layer, "self_attn.kv_b_proj.weight"))) {
    const auto* qa = store_.find(Lname(layer, "self_attn.q_a_proj.weight"));
    const auto* kva = store_.find(Lname(layer, "self_attn.kv_a_proj_with_mqa.weight"));
    const int q_lora = static_cast<int>(qa->shape[0]);
    const int kv_lora = static_cast<int>(kva->shape[0]);
    const int qkd = qk_dim_ > 0 ? qk_dim_ : hd;
    const int vd = v_dim_ > 0 ? v_dim_ : hd;
    q.assign(static_cast<size_t>(nh) * qkd, 0.f);
    k.assign(static_cast<size_t>(nkv) * qkd, 0.f);
    v.assign(static_cast<size_t>(nkv) * vd, 0.f);
    ops::mla_absorb_q(normed, store_.require_bf16(Lname(layer, "self_attn.q_a_proj.weight")),
                      store_.require_bf16(Lname(layer, "self_attn.q_b_proj.weight")), q.data(), H,
                      q_lora, nh, qkd);
    ops::mla_absorb_kv(normed, store_.require_bf16(Lname(layer, "self_attn.kv_a_proj_with_mqa.weight")),
                       store_.require_bf16(Lname(layer, "self_attn.kv_b_proj.weight")), k.data(),
                       v.data(), H, kv_lora, nkv, qkd, vd);
    // Pad/truncate into hd_ cache slots when dims differ
    if (qkd != hd || vd != hd) {
      std::vector<float> q2(nh * hd, 0.f), k2(nkv * hd, 0.f), v2(nkv * hd, 0.f);
      for (int h = 0; h < nh; ++h)
        std::memcpy(q2.data() + h * hd, q.data() + h * qkd, sizeof(float) * std::min(qkd, hd));
      for (int h = 0; h < nkv; ++h) {
        std::memcpy(k2.data() + h * hd, k.data() + h * qkd, sizeof(float) * std::min(qkd, hd));
        std::memcpy(v2.data() + h * hd, v.data() + h * vd, sizeof(float) * std::min(vd, hd));
      }
      q.swap(q2);
      k.swap(k2);
      v.swap(v2);
    }
  } else if (store_.find(Lname(layer, "self_attn.q_proj.weight"))) {
    gemm_bf16_named(normed, Lname(layer, "self_attn.q_proj.weight"), q.data(), nh * hd, H);
    gemm_bf16_named(normed, Lname(layer, "self_attn.k_proj.weight"), k.data(), nkv * hd, H);
    gemm_bf16_named(normed, Lname(layer, "self_attn.v_proj.weight"), v.data(), nkv * hd, H);
  } else {
    std::fill(oproj, oproj + H, 0.f);
    cache.layer(layer).seq = pos + 1;
    return;
  }

  // qk_rope_head_dim==0 (mla_use_nope) → skip RoPE when MLA weights present
  const bool use_rope = !store_.find(Lname(layer, "self_attn.q_a_proj.weight"));
  if (use_rope) {
    for (int h = 0; h < nh; ++h) ops::rope_inplace(q.data() + h * hd, hd, pos, rope_theta_);
    for (int h = 0; h < nkv; ++h) ops::rope_inplace(k.data() + h * hd, hd, pos, rope_theta_);
  }

  auto& Lkv = cache.layer(layer);
  for (int h = 0; h < nkv; ++h) {
    float* kdst = Lkv.k.data() + (static_cast<size_t>(h) * max_seq + pos) * hd;
    float* vdst = Lkv.v.data() + (static_cast<size_t>(h) * max_seq + pos) * hd;
    std::memcpy(kdst, k.data() + h * hd, sizeof(float) * hd);
    std::memcpy(vdst, v.data() + h * hd, sizeof(float) * hd);
  }
  Lkv.seq = pos + 1;

  // KPool indexer (compress → top pools → expand). FlashMLA is SM90+; Ada uses this SDPA path.
  const int d_idx = index_dim_ > 0 ? index_dim_ : hd;
  const bool have_idx =
      store_.find(Lname(layer, "self_attn.indexer.weights_proj.weight")) ||
      store_.find(Lname(layer, "self_attn.indexer.wq_b.weight"));
  const int pool = index_kpool_compress_ ? index_kpool_ : 1;
  const int n_pools = (pos + pool) / pool;
  if (have_idx && index_topk_ > 0 && n_pools > index_topk_) {
    std::vector<float> q_idx(d_idx);
    if (store_.find(Lname(layer, "self_attn.indexer.weights_proj.weight"))) {
      gemm_bf16_named(normed, Lname(layer, "self_attn.indexer.weights_proj.weight"), q_idx.data(),
                      d_idx, H);
    } else {
      gemm_bf16_named(normed, Lname(layer, "self_attn.indexer.wq_b.weight"), q_idx.data(), d_idx, H);
    }
    std::vector<float> k_idx_cache(static_cast<size_t>(max_seq) * d_idx, 0.f);
    const int use_d = std::min(d_idx, hd);
    for (int t = 0; t <= pos; ++t) {
      for (int d = 0; d < use_d; ++d) {
        double s = 0.0;
        for (int h = 0; h < nkv; ++h)
          s += Lkv.k[(static_cast<size_t>(h) * max_seq + t) * hd + d];
        k_idx_cache[static_cast<size_t>(t) * d_idx + d] = static_cast<float>(s / nkv);
      }
    }
    std::vector<int> idx(static_cast<size_t>(index_topk_) * static_cast<size_t>(pool) + 1);
    const int nsel = ops::kpool_select(q_idx.data(), k_idx_cache.data(), idx.data(), index_topk_,
                                       pos, max_seq, d_idx, pool);
    ops::gqa_attn_indexed(q.data(), Lkv.k.data(), Lkv.v.data(), attn_out.data(), nh, nkv, hd,
                          max_seq, idx.data(), nsel);
  } else {
    ops::gqa_attn(q.data(), Lkv.k.data(), Lkv.v.data(), attn_out.data(), nh, nkv, hd, max_seq, pos);
  }

  gemm_bf16_named(attn_out.data(), Lname(layer, "self_attn.o_proj.weight"), oproj, H, nh * hd);
}

void GlmFlashModel::layer_forward(int layer, float* x, model::SessionCache& cache, int pos) {
  const int H = H_;
  std::vector<float> residual(H), normed(H), oproj(H);
  std::memcpy(residual.data(), x, sizeof(float) * H);

  ops::rmsnorm(x, store_.require_bf16(Lname(layer, "input_layernorm.weight")), normed.data(), H,
               rms_eps_);

  const bool is_linear =
      (static_cast<size_t>(layer) < layer_types_.size() &&
       layer_types_[layer] == "linear_attention") ||
      (layer_types_.empty() && (layer % 2 == 1));

  if (is_linear) {
    attn_linear_kda(layer, normed.data(), oproj.data(), cache);
    cache.layer(layer).seq = pos + 1;
  } else {
    attn_sparse_mla_or_gqa(layer, normed.data(), oproj.data(), cache, pos);
  }

  if (mhc_ && hc_mult_ > 1) {
    std::vector<float> streams(static_cast<size_t>(hc_mult_) * H);
    for (int c = 0; c < hc_mult_; ++c)
      std::memcpy(streams.data() + c * H, residual.data(), sizeof(float) * H);
    for (int d = 0; d < H; ++d) streams[d] += oproj[d];
    std::vector<float> mixed(static_cast<size_t>(hc_mult_) * H);
    const uint16_t* Hmix = store_.bf16(Lname(layer, "mhc.mix.weight"));
    ops::mhc_mix(streams.data(), mixed.data(), hc_mult_, H, Hmix);
    std::memcpy(x, mixed.data(), sizeof(float) * H);  // stream 0
  } else {
    for (int i = 0; i < H; ++i) x[i] = residual[i] + oproj[i];
  }

  std::memcpy(residual.data(), x, sizeof(float) * H);
  ops::rmsnorm(x, store_.require_bf16(Lname(layer, "post_attention_layernorm.weight")),
               normed.data(), H, rms_eps_);

  const bool use_dense =
      layer < first_k_dense_ || store_.find(Lname(layer, "mlp.gate_proj.weight")) != nullptr;
  if (use_dense && store_.find(Lname(layer, "mlp.gate.weight")) == nullptr) {
    dense_ffn(layer, normed.data());
  } else {
    moe_ffn(layer, normed.data());
  }
  for (int i = 0; i < H; ++i) x[i] = residual[i] + normed[i];
}

void GlmFlashModel::forward(const std::vector<int32_t>& tokens, model::SessionCache& cache,
                            std::vector<float>& logits, bool is_prefill) {
  if (!weights_ready_) {
    throw std::runtime_error(
        std::string("glm: no GLMQ weights") +
        (load_error_.empty() ? "" : (": " + load_error_)) +
        " — check models/*.glmq exists and matches configs/engine_glm_*.yaml model.path. "
        "See docs/MODEL_GLM53_FLASH.md");
  }
  if (tokens.empty()) {
    logits.assign(V_, 0.f);
    return;
  }

  const int H = H_, V = V_;
  const uint16_t* emb = store_.require_bf16("embed_tokens.weight");
  std::vector<float> x(H);

  int pos_start = is_prefill ? 0 : cache.layer(0).seq;
  if (is_prefill) {
    for (int i = 0; i < cache.n_layers(); ++i) cache.layer(i).seq = 0;
  }

  for (size_t ti = 0; ti < tokens.size(); ++ti) {
    const int tok = tokens[ti];
    const int pos = pos_start + static_cast<int>(ti);
    if (tok < 0 || tok >= V) throw std::runtime_error("glm: token id OOB");
    const uint16_t* row = emb + static_cast<size_t>(tok) * H;
    for (int i = 0; i < H; ++i) x[i] = ops::bf16_to_f32(row[i]);
    for (int L = 0; L < L_; ++L) layer_forward(L, x.data(), cache, pos);
  }

  std::vector<float> normed(H);
  ops::rmsnorm(x.data(), store_.require_bf16("norm.weight"), normed.data(), H, rms_eps_);
  logits.assign(V, 0.f);
  const uint16_t* lm = store_.bf16("lm_head.weight");
  if (!lm) lm = emb;
  ops::gemm_bf16(normed.data(), lm, logits.data(), V, H);
}

}  // namespace llmoc::glm
