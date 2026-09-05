#pragma once
// llm-on-cpu :: model/qwen3_5_int4_model.h
// Qwen3.5 INT4 推理路径（读 QLWC，不改动原 Qwen35Model）。

#include <string>
#include <vector>

#include "hal/cpu_ops.h"
#include "model/causal_lm.h"
#include "model/mtp_head.h"
#include "model/tokenizer_hf.h"
#include "model/vision/qwen_vision_encoder.h"
#include "weights/layer_stream.h"
#include "weights/qlwc_store.h"

namespace llmoc::model {

struct Qwen35Int4Config {
  int hidden = 2560;
  int layers = 32;
  int n_heads = 16;
  int n_kv = 4;
  int head_dim = 256;
  int intermediate = 9216;
  int vocab = 248320;
  float rms_eps = 1e-6f;
  float rope_theta = 10000000.f;
  float partial_rotary = 0.25f;
  int linear_num_k = 16;
  int linear_num_v = 32;
  int linear_dk = 128;
  int linear_dv = 128;
  int conv_k = 4;
  bool tie_embeddings = true;
  std::vector<std::string> layer_types;
  int32_t image_token_id = 248056;
  int32_t vision_start_id = 248053;
  int32_t vision_end_id = 248054;
};

class Qwen35Int4Model final : public ICausalLM {
 public:
  void load(qlwc::QlwcStore* store, const std::string& hf_config_json_path);
  // 层流式：store 须 lazy open；loader 负责 pin/prefetch/release
  void enable_layer_stream(wt::ILayerStreamLoader* loader);
  bool layer_stream_enabled() const { return streamer_ != nullptr; }
  // hybrid/pure_gpu：将层投影 INT4 反量化上传 VRAM（跳过 embed/lm_head）
  void warm_gpu_int4_weights();
  // Estimate GDN states + scratch for auto resident-GPU probe (after warm).
  size_t resident_workspace_bytes() const;
  void enable_resident_gpu(bool on);
  bool resident_gpu() const { return resident_gpu_; }
  const CausalLmMeta& meta() const override { return meta_; }
  void init_cache(SessionCache& cache, int max_seq) const override;
  void forward(const std::vector<int32_t>& tokens, SessionCache& cache, std::vector<float>& logits,
               bool is_prefill) override;
  bool forward_decode_greedy(const std::vector<int32_t>& tokens, SessionCache& cache,
                             int32_t& out_token) override;
  void forward_all_logits(const std::vector<int32_t>& tokens, SessionCache& cache,
                          std::vector<float>& logits_all, bool is_prefill) override;
  void commit_prefix_state(int pos) override;
  bool has_mtp() const override;
  bool draft_propose(const std::vector<int32_t>& history, int draft_k,
                     std::vector<int32_t>& out, int32_t pin_first = -1) override;

  bool has_vision() const override { return vision_.ready(); }
  void set_vision_embeds(std::vector<float> embeds, int n_tok) override;
  void clear_vision_embeds() override;
  int32_t image_pad_token_id() const override { return cfg_.image_token_id; }
  int32_t vision_start_token_id() const override { return cfg_.vision_start_id; }
  int32_t vision_end_token_id() const override { return cfg_.vision_end_id; }
  bool encode_message_images(const std::vector<ChatMessage>& messages,
                             std::vector<float>& embeds_out,
                             std::vector<int>& pad_counts_out) override;

  vision::QwenVisionEncoder& vision() { return vision_; }
  const vision::ImagePrepConfig& image_prep_cfg() const { return image_prep_; }
  int32_t vision_start_id() const { return cfg_.vision_start_id; }
  int32_t vision_end_id() const { return cfg_.vision_end_id; }

 private:
  struct OptW {
    bool is_int4 = false;
    qlwc::Int4View i4{};
    const uint16_t* pass = nullptr;
    hal::WDtype dt = hal::WDtype::kBF16;
    int M = 0;
    int K = 0;
  };

  struct LayerPack {
    bool is_full = false;
    const uint16_t* ln1 = nullptr;
    const uint16_t* ln2 = nullptr;
    // full attention
    qlwc::Int4View wq, wk, wv, wo;
    const uint16_t* qn = nullptr;
    const uint16_t* kn = nullptr;
    // linear / GDN（a/b/out 在 AWQ ignore 列表里常为 BF16 透传）
    qlwc::Int4View wqkv, wz;
    OptW wb, wa, wout;
    const uint16_t* nrm = nullptr;
    std::vector<float> A_log_f;
    std::vector<float> dt_bias_f;
    std::vector<float> conv_w_f;  // [conv_dim * conv_k]
    // mlp
    qlwc::Int4View wgate, wup, wdown;
  };

  qlwc::QlwcStore* store_ = nullptr;
  Qwen35Int4Config cfg_;
  CausalLmMeta meta_;
  std::string prefix_ = "language_model.";
  hal::WDtype pass_wd_ = hal::WDtype::kBF16;
  vision::QwenVisionEncoder vision_;
  vision::ImagePrepConfig image_prep_;
  std::vector<float> vision_embeds_;
  int vision_n_tok_ = 0;
  int vision_cursor_ = 0;
  std::vector<int> vision_grid_thw_;  // flat [n_img*3] patch grid (t,h,w)
  int mrope_section_[3] = {11, 11, 10};
  bool mrope_interleaved_ = true;
  int mrope_next_ = 0;
  std::vector<int> cur_pos_t_, cur_pos_h_, cur_pos_w_;
  std::vector<LayerPack> layers_;
  qlwc::Int4View emb_int4_{};
  bool emb_is_int4_ = false;
  const uint16_t* emb_pass_ = nullptr;
  qlwc::Int4View lm_int4_{};
  bool lm_is_int4_ = false;
  const uint16_t* lm_pass_ = nullptr;
  const uint16_t* final_norm_ = nullptr;

  bool is_int4(const std::string& name) const;
  const uint16_t* pass(const std::string& name);
  OptW load_opt_w(const std::string& name, int M, int K);
  void gemm_w(const float* x, const std::string& wname, float* y, int M, int K);
  void gemm_view(const float* x, const qlwc::Int4View& W, float* y);
  void gemm_view_batch(const float* X, int n, const qlwc::Int4View& W, float* Y);
  void gemm_opt(const float* x, const OptW& W, float* y);
  void gemm_opt_batch(const float* X, int n, const OptW& W, float* Y);
  void embed(int32_t token, float* out);
  void layer_forward(int layer, float* x, SessionCache& cache, int pos_start, int n_tok,
                     bool is_prefill);
  void forward_to_hidden(const std::vector<int32_t>& tokens, SessionCache& cache, bool is_prefill,
                         float* h_out, double* ms_lin = nullptr, double* ms_full = nullptr);
  void prepare_mrope_positions(const std::vector<int32_t>& tokens, bool is_prefill);
  void build_layer_packs();
  void fill_layer_pack(int L);
  void build_global_packs();
  MtpWeightAccess mtp_access();

  static bool mtp_has_cb(void* ctx, const std::string& name);
  static void mtp_gemm_cb(void* ctx, const float* x, const std::string& wname, float* y, int M,
                          int K);
  static const uint16_t* mtp_pass_cb(void* ctx, const std::string& name);
  static void mtp_embed_cb(void* ctx, int32_t token, float* out);
  static float mtp_embed_dot_cb(void* ctx, const float* h, int32_t token);

  wt::ILayerStreamLoader* streamer_ = nullptr;
  int stream_window_ = 2;
  bool resident_gpu_ = false;

  std::vector<float> last_hidden_;
  std::vector<float> last_logits_;
  std::vector<float> prefix_hiddens_;
  std::vector<float> prefix_logits_;
};

}  // namespace llmoc::model
