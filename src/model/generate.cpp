// llm-on-cpu :: model/generate.cpp
#include "model/generate.h"

#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "common/log.h"

namespace llmoc::model {
namespace {

int32_t sample_token(const std::vector<float>& logits, float temperature) {
  if (logits.empty()) throw std::runtime_error("empty logits");
  const int n = static_cast<int>(logits.size());

  if (temperature <= 1e-5f) {
    // 并行分段求 max（大词表 greedy）
    int32_t best = 0;
    float v = -std::numeric_limits<float>::infinity();
#if defined(_OPENMP)
    if (n >= 4096) {
      int32_t best_t = 0;
      float v_t = -std::numeric_limits<float>::infinity();
#pragma omp parallel
      {
        int32_t lb = 0;
        float lv = -std::numeric_limits<float>::infinity();
#pragma omp for nowait schedule(static)
        for (int i = 0; i < n; ++i) {
          const float x = logits[i];
          if (!std::isfinite(x)) continue;
          if (x > lv) {
            lv = x;
            lb = i;
          }
        }
#pragma omp critical
        {
          if (lv > v_t) {
            v_t = lv;
            best_t = lb;
          }
        }
      }
      return best_t;
    }
#endif
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(logits[i])) continue;
      if (logits[i] > v) {
        v = logits[i];
        best = i;
      }
    }
    return best;
  }

  bool any_finite = false;
  for (float x : logits) {
    if (std::isfinite(x)) {
      any_finite = true;
      break;
    }
  }
  if (!any_finite) throw std::runtime_error("logits are all NaN/Inf");

  const float inv_t = 1.f / temperature;
  float m = logits[0] * inv_t;
  for (float x : logits) m = std::max(m, x * inv_t);
  double sum = 0.0;
  std::vector<double> p(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp(static_cast<double>(logits[i] * inv_t - m));
    sum += p[i];
  }
  static thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<double> dist(0.0, sum);
  double r = dist(rng);
  double c = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    c += p[i];
    if (r <= c) return static_cast<int32_t>(i);
  }
  return static_cast<int32_t>(p.size() - 1);
}

size_t utf8_valid_prefix_len(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t n = 0;
    if (c < 0x80) n = 1;
    else if ((c >> 5) == 0x6) n = 2;
    else if ((c >> 4) == 0xE) n = 3;
    else if ((c >> 3) == 0x1E) n = 4;
    else break;
    if (i + n > s.size()) break;
    bool ok = true;
    for (size_t k = 1; k < n; ++k)
      if ((static_cast<unsigned char>(s[i + k]) >> 6) != 0x2) {
        ok = false;
        break;
      }
    if (!ok) break;
    i += n;
  }
  return i;
}

std::string sanitize_utf8(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t good = utf8_valid_prefix_len(s.substr(i));
    out += s.substr(i, good);
    i += good;
    if (i < s.size()) {
      out += "\xEF\xBF\xBD";
      ++i;
    }
  }
  return out;
}

bool only_ws_piece(const std::string& s) {
  if (s.empty()) return true;
  for (unsigned char c : s)
    if (c > 32) return false;
  return true;
}

bool mtp_wanted(const std::string& mode) {
  return mode == "true" || mode == "auto" || mode == "1";
}

bool messages_have_images(const std::vector<ChatMessage>& messages) {
  for (const auto& m : messages)
    if (!m.images.empty()) return true;
  return false;
}

std::vector<int32_t> build_prompt_ids(const GenerateRequest& req, HfTokenizer* tok,
                                      const std::vector<int>& image_pad_counts,
                                      int32_t vision_start, int32_t image_pad, int32_t vision_end) {
  std::vector<int32_t> ids;
  auto append_text = [&](const std::string& s) {
    auto e = tok->encode(s);
    ids.insert(ids.end(), e.begin(), e.end());
  };
  int img_i = 0;
  for (const auto& m : req.messages) {
    append_text("<|im_start|>" + m.role + "\n");
    if (!m.images.empty()) {
      for (size_t k = 0; k < m.images.size(); ++k) {
        if (img_i >= static_cast<int>(image_pad_counts.size()))
          throw std::runtime_error("image_pad_counts mismatch");
        const int n_pad = image_pad_counts[img_i++];
        std::ostringstream pic;
        pic << "Picture " << (k + 1) << ": ";
        append_text(pic.str());
        ids.push_back(vision_start);
        for (int p = 0; p < n_pad; ++p) ids.push_back(image_pad);
        ids.push_back(vision_end);
      }
    }
    append_text(m.content);
    append_text("<|im_end|>\n");
  }
  append_text("<|im_start|>assistant\n");
  if (req.enable_thinking)
    append_text("<think>\n");
  else
    append_text("<think>\n\n</think>\n\n");
  return ids;
}

}  // namespace

void Generator::init(ICausalLM* model, HfTokenizer* tok, int max_seq) {
  model_ = model;
  tok_ = tok;
  max_seq_ = max_seq;
}

GenerateResult Generator::generate(const GenerateRequest& req,
                                   const std::function<void(const std::string& delta)>& on_token) {
  if (!model_ || !tok_) throw std::runtime_error("Generator not initialized");

  std::vector<int32_t> ids;
  model_->clear_vision_embeds();

  if (messages_have_images(req.messages)) {
    if (!model_->has_vision())
      throw std::runtime_error("images provided but vision encoder is unavailable");
    std::vector<float> all_embeds;
    std::vector<int> pad_counts;
    if (!model_->encode_message_images(req.messages, all_embeds, pad_counts))
      throw std::runtime_error("failed to encode images");
    int total_pads = 0;
    for (int n : pad_counts) total_pads += n;
    model_->set_vision_embeds(std::move(all_embeds), total_pads);
    ids = build_prompt_ids(req, tok_, pad_counts, model_->vision_start_token_id(),
                           model_->image_pad_token_id(), model_->vision_end_token_id());
  } else {
    const std::string prompt = apply_qwen_chat_template(req.messages, true, req.enable_thinking);
    ids = tok_->encode(prompt);
  }

  if (static_cast<int>(ids.size()) >= max_seq_ - 8)
    throw std::runtime_error("prompt too long");

  SessionCache cache;
  model_->init_cache(cache, max_seq_);
  (void)radix_.longest_prefix(ids);

  std::vector<float> logits;
  model_->forward(ids, cache, logits, true);
  cache.tokens = ids;

  GenerateResult out;
  out.prompt_tokens = static_cast<int>(ids.size());
  const int32_t eos = tok_->im_end_id() >= 0 ? tok_->im_end_id() : tok_->eos_id();

  const bool want_mtp = mtp_wanted(req.mtp) && req.spec_k > 0;
  const bool use_mtp = want_mtp && model_->has_mtp();
  if (want_mtp && !model_->has_mtp()) {
    LOG_INFO("mtp disabled: model has no MTP weights (mode=%s)", req.mtp.c_str());
  }

  std::string utf8_carry;
  int ws_run = 0;
  auto emit_one = [&](int32_t next) -> bool {
    if (next == eos || (tok_->eos_id() >= 0 && next == tok_->eos_id())) return false;
    out.token_ids.push_back(next);
    const std::string piece = tok_->decode({next}, true);
    out.text += piece;
    if (only_ws_piece(piece)) {
      if (++ws_run >= 8) return false;
    } else {
      ws_run = 0;
    }
    if (on_token) {
      utf8_carry += piece;
      size_t good = utf8_valid_prefix_len(utf8_carry);
      if (good) {
        on_token(utf8_carry.substr(0, good));
        utf8_carry.erase(0, good);
      }
      while (utf8_carry.size() > 3) {
        on_token("\xEF\xBF\xBD");
        utf8_carry.erase(0, 1);
        good = utf8_valid_prefix_len(utf8_carry);
        if (good) {
          on_token(utf8_carry.substr(0, good));
          utf8_carry.erase(0, good);
        }
      }
    }
    return static_cast<int>(out.token_ids.size()) < req.max_new_tokens;
  };

  while (static_cast<int>(out.token_ids.size()) < req.max_new_tokens) {
    if (use_mtp) {
      std::vector<int32_t> drafts;
      if (!model_->draft_propose(cache.tokens, req.spec_k, drafts) || drafts.empty()) {
        const int32_t next = sample_token(logits, req.temperature);
        if (!emit_one(next)) break;
        model_->forward({next}, cache, logits, false);
        cache.tokens.push_back(next);
        continue;
      }

      ++out.mtp_verify_steps;
      // 先用当前 logits 核对 drafts[0]；再一次性 forward 全部草稿做批量 verify
      if (sample_token(logits, /*temperature=*/0.f) != drafts[0]) {
        const int32_t next = sample_token(logits, req.temperature);
        if (!emit_one(next)) break;
        model_->forward({next}, cache, logits, false);
        cache.tokens.push_back(next);
        continue;
      }

      const auto snap = cache.snapshot();
      std::vector<float> all_logits;
      model_->forward_all_logits(drafts, cache, all_logits, false);
      const int V = model_->meta().vocab;
      const int k = static_cast<int>(drafts.size());

      int accepted = 1;
      for (int i = 1; i < k; ++i) {
        const float* slice = all_logits.data() + static_cast<size_t>(i - 1) * V;
        std::vector<float> tmp(slice, slice + V);
        if (sample_token(tmp, /*temperature=*/0.f) != drafts[static_cast<size_t>(i)]) break;
        ++accepted;
      }

      if (accepted < k) {
        // 截断 KV 到采纳前缀，复用 all_logits / 位置 hidden（避免二次 full forward）
        cache.restore(snap);
        for (int li = 0; li < cache.n_layers(); ++li)
          cache.layer(li).seq = snap.seq[static_cast<size_t>(li)] + accepted;
        logits.assign(all_logits.begin() + static_cast<size_t>(accepted - 1) * V,
                      all_logits.begin() + static_cast<size_t>(accepted) * V);
        model_->commit_prefix_state(accepted - 1);  // last_hidden/logits from forward_all
      } else {
        logits.assign(all_logits.begin() + static_cast<size_t>(k - 1) * V, all_logits.end());
      }

      bool stop = false;
      for (int i = 0; i < accepted; ++i) {
        cache.tokens.push_back(drafts[static_cast<size_t>(i)]);
        ++out.mtp_draft_accepted;
        if (!emit_one(drafts[static_cast<size_t>(i)])) {
          stop = true;
          break;
        }
      }
      if (stop) break;

      // 免费 next：来自最后一个采纳位置的 logits
      const int32_t next = sample_token(logits, req.temperature);
      if (!emit_one(next)) break;
      model_->forward({next}, cache, logits, false);
      cache.tokens.push_back(next);
      continue;
    }

    // Greedy / 普通采样逐步路径
    const int32_t next = sample_token(logits, req.temperature);
    if (!emit_one(next)) break;
    model_->forward({next}, cache, logits, false);
    cache.tokens.push_back(next);
  }

  while (!out.token_ids.empty()) {
    const std::string p = tok_->decode({out.token_ids.back()}, true);
    bool ws = p.empty();
    for (unsigned char c : p)
      if (c > 32) {
        ws = false;
        break;
      } else
        ws = true;
    if (!ws) break;
    out.token_ids.pop_back();
  }
  out.text.clear();
  for (int32_t id : out.token_ids) out.text += tok_->decode({id}, true);
  if (on_token && !utf8_carry.empty()) on_token(sanitize_utf8(utf8_carry));
  out.text = sanitize_utf8(out.text);
  if (!req.enable_thinking) out.text = strip_qwen_think(out.text);
  out.completion_tokens = static_cast<int>(out.token_ids.size());
  radix_.insert(cache.tokens, static_cast<int>(cache.tokens.size()));
  model_->clear_vision_embeds();
  return out;
}

}  // namespace llmoc::model
