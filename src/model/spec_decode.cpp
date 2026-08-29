// llm-on-cpu :: model/spec_decode.cpp
#include "model/spec_decode.h"

#include <stdexcept>

namespace llmoc::spec {

SpecDecodeRunner::SpecDecodeRunner(IDraft& draft, IVerify& verify, Config cfg)
    : draft_(draft), verify_(verify), cfg_(cfg) {}

void SpecDecodeRunner::run(const std::vector<TokenId>& input, size_t total,
                           std::vector<TokenId>* output) {
    if (cfg_.draft_k < 1) throw std::runtime_error("draft_k must be >= 1");
    std::vector<TokenId> hist(input.begin(), input.end());
    std::vector<TokenId> drafts;

    while (st_.emitted < total) {
        drafts.clear();
        draft_.propose(hist, cfg_.draft_k, drafts);
        if (static_cast<int>(drafts.size()) > cfg_.draft_k)
            throw std::runtime_error("draft overflow");

        const VerifyOutcome v = verify_.verify(hist, drafts);
        if (v.accepted < 0 ||
            v.accepted > static_cast<int>(drafts.size()))
            throw std::runtime_error("verify: accepted out of range");

        // 采纳前缀
        hist.insert(hist.end(), drafts.begin(), drafts.begin() + v.accepted);
        // 免费的真值 token
        hist.push_back(v.next);

        ++st_.steps;
        st_.accepted_total += static_cast<uint64_t>(v.accepted);
        st_.emitted += static_cast<uint64_t>(v.accepted) + 1;
    }

    // 越步截断: 高采纳率下单步可能超产(最多 k 个), 用户流按请求长度截齐
    // (真实引擎中多余 token 仍留在 KV/prefix 供后续复用, 不丢弃计算价值)
    const size_t want = input.size() + total;
    if (hist.size() > want) hist.resize(want);

    if (output) output->assign(hist.begin() + static_cast<long>(input.size()),
                               hist.end());
}

VerifyOutcome MockVerify::verify(const std::vector<TokenId>& history,
                                 const std::vector<TokenId>& drafts) {
    VerifyOutcome v;
    const size_t base = history.size();
    while (v.accepted < static_cast<int>(drafts.size()) &&
           drafts[static_cast<size_t>(v.accepted)] ==
               oracle_.true_next(base + static_cast<size_t>(v.accepted)))
        ++v.accepted;
    v.next = oracle_.true_next(base + static_cast<size_t>(v.accepted));
    return v;
}

MockDraft::MockDraft(IOracle& oracle, double alpha_hit, uint64_t seed)
    : oracle_(oracle), alpha_(alpha_hit), rng_(seed ? seed : 0x9E3779B97F4A7C15ull) {}

void MockDraft::propose(const std::vector<TokenId>& history,
                        int draft_k, std::vector<TokenId>& out) {
    out.clear();
    const size_t base = history.size();
    for (int i = 0; i < draft_k; ++i) {
        rng_ = rng_ * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t r = static_cast<uint32_t>(rng_ >> 33);  // [0, 2^31)
        const bool hit = (r % 1000u) <
                         static_cast<uint32_t>(alpha_ * 1000.0);
        const size_t pos = base + static_cast<size_t>(i);
        out.push_back(hit ? oracle_.true_next(pos)
                          : static_cast<TokenId>(oracle_.true_next(pos) + 1 + (r % 7)));
    }
}

}  // namespace llmoc::spec
