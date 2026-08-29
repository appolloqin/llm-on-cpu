// llm-on-cpu :: tests/unit/test_spec_decode.cpp
// M3 verify 循环状态机测试: 输出恒等 oracle / 采纳率与乘法器统计 / 边界
#include "test_main.h"

#include <stdexcept>
#include <vector>

#include "model/spec_decode.h"

namespace {

using namespace llmoc::spec;

// oracle: 交错序列, 可预测
class SeqOracle final : public IOracle {
   public:
    explicit SeqOracle(int mod) : mod_(mod) {}
    TokenId true_next(size_t pos) override {
        return static_cast<TokenId>(pos % static_cast<size_t>(mod_));
    }

   private:
    int mod_;
};

}  // namespace

TINY_TEST(Spec, OutputAlwaysMatchesOracle) {
    for (double alpha = 0.0; alpha <= 1.0001; alpha += 0.25) {
        SeqOracle oracle(97);
        MockDraft draft(oracle, alpha, 42);
        MockVerify verify(oracle);
        SpecDecodeRunner runner(draft, verify, {/*draft_k=*/4});

        std::vector<TokenId> out;
        runner.run({1, 2}, 200, &out);
        EXPECT_EQ(out.size(), 200ull);
        for (size_t i = 0; i < out.size(); ++i)
            EXPECT_EQ(out[i], oracle.true_next(i + 2));  // 输入{1,2}后接 pos=2 起
    }
}

TINY_TEST(Spec, AlphaZeroIsPlainDecode) {
    SeqOracle oracle(53);
    MockDraft draft(oracle, 0.0, 7);
    MockVerify verify(oracle);
    SpecDecodeRunner runner(draft, verify, {/*draft_k=*/3});

    std::vector<TokenId> out;
    runner.run({}, 100, &out);
    // 全拒 => 每步 1 token => steps==emitted==100, 乘法器==1
    EXPECT_EQ(runner.stats().steps, 100ull);
    EXPECT_EQ(runner.stats().emitted, 100ull);
    EXPECT_TRUE(runner.stats().tokens_per_step() > 0.99 &&
                runner.stats().tokens_per_step() < 1.01);
    EXPECT_EQ(out.size(), 100ull);
}

TINY_TEST(Spec, AlphaOneHitsPerfectSpeedup) {
    SeqOracle oracle(53);
    MockDraft draft(oracle, 1.0, 7);
    MockVerify verify(oracle);
    SpecDecodeRunner runner(draft, verify, {/*draft_k=*/4});

    std::vector<TokenId> out;
    runner.run({}, 200, &out);
    // 全收 => 每步 k+1=5 token
    EXPECT_EQ(runner.stats().tokens_per_step(), 5.0);
    EXPECT_EQ(runner.stats().alpha(), 4.0);
    EXPECT_EQ(out.size(), 200ull);
}

TINY_TEST(Spec, DraftKMustBePositive) {
    SeqOracle oracle(10);
    MockDraft draft(oracle, 0.5, 1);
    MockVerify verify(oracle);
    SpecDecodeRunner runner(draft, verify, {/*draft_k=*/0});
    bool threw = false;
    try {
        runner.run({}, 5, nullptr);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}
