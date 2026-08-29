#pragma once
// llm-on-cpu :: model/spec_decode.h
// M3: MTP(nextn) 自投机解码 verify 循环 —— 状态机骨架(D3)。
//
// 真实引擎中的映射:
//   IDraft::propose   <- MTP(nextn) 头逐位草拟 k 个 token(接口 M4 接权重)
//   IVerify::verify   <- 主模型一次前向的 logits 对草稿序列做贪心前缀比对
//   本轮交付: 可离线单测/可数字仿真的完整状态机 + 模拟实现(oracle 驱动)
//
// 状态转移(对应 ARCHITECTURE D3 时序):
//   [主模型前向=verify] --accept(0..k)+next--> [补草稿] --> 下一 verify
// 每 verify 至少产出 1 个 token(next), 至多 k+1 个 —— 带宽墙乘法器的来源。

#include <cstdint>
#include <vector>

namespace llmoc::spec {

using TokenId = int32_t;

// 验证结果: accepted 为草稿前缀被采纳的数量; next 是验证步必然正确的"免费"token
struct VerifyOutcome {
    int accepted = 0;
    TokenId next = 0;
};

// 草稿生成器(MTP 头替身)
class IDraft {
   public:
    virtual ~IDraft() = default;
    // 基于(初始输入+已产出)历史, 草拟至多 draft_k 个后续 token
    virtual void propose(const std::vector<TokenId>& history,
                         int draft_k, std::vector<TokenId>& out) = 0;
};

// 验证器(主模型前向替身)
class IVerify {
   public:
    virtual ~IVerify() = default;
    virtual VerifyOutcome verify(const std::vector<TokenId>& history,
                                 const std::vector<TokenId>& drafts) = 0;
};

class SpecDecodeRunner {
   public:
    struct Config {
        int draft_k = 3;
    };

    struct Stats {
        uint64_t steps = 0;           // verify 次数 = 主模型前向次数(权重遍历数)
        uint64_t emitted = 0;         // 实际产出 token 数
        uint64_t accepted_total = 0;  // 草稿被采纳总数

        double alpha() const {              // 平均单步采纳率 accepted/steps
            return steps ? static_cast<double>(accepted_total) /
                               static_cast<double>(steps)
                         : 0.0;
        }
        double tokens_per_step() const {    // 带宽乘法器实测值
            return steps ? static_cast<double>(emitted) /
                               static_cast<double>(steps)
                         : 0.0;
        }
    };

    SpecDecodeRunner(IDraft& draft, IVerify& verify, Config cfg);

    // 驱动循环至产出 total 个 token; output 为完整序列(初始输入之后的部分)。
    void run(const std::vector<TokenId>& input, size_t total,
             std::vector<TokenId>* output);

    const Stats& stats() const { return st_; }
    int draft_k() const { return cfg_.draft_k; }

   private:
    IDraft& draft_;
    IVerify& verify_;
    Config cfg_;
    Stats st_{};
};

// ---- 模拟实现(单测/仿真用) ----

// Oracle: 真实分布的替身 —— "正确续写"由外部给定
class IOracle {
   public:
    virtual ~IOracle() = default;
    virtual TokenId true_next(size_t pos) = 0;
};

class MockVerify final : public IVerify {
   public:
    explicit MockVerify(IOracle& oracle) : oracle_(oracle) {}
    VerifyOutcome verify(const std::vector<TokenId>& history,
                         const std::vector<TokenId>& drafts) override;

   private:
    IOracle& oracle_;
};

// 按目标命中率 alpha_hit 独立猜测每个草稿位(确定性种子, 可复现)
class MockDraft final : public IDraft {
   public:
    MockDraft(IOracle& oracle, double alpha_hit, uint64_t seed);
    void propose(const std::vector<TokenId>& history,
                 int draft_k, std::vector<TokenId>& out) override;

   private:
    IOracle& oracle_;
    double alpha_;
    uint64_t rng_;
};

}  // namespace llmoc::spec
