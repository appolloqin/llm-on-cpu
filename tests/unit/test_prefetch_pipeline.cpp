// llm-on-cpu :: tests/unit/test_prefetch_pipeline.cpp
// M2 流水线(D2)行为测试: 正确性 / 槽复用完整性 / pin 热专家 / 状态机保护
#include "test_main.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "weights/prefetch_pipeline.h"

namespace fs = std::filesystem;

namespace {

fs::path tmp_dir() {
    auto d = fs::temp_directory_path() / "llmoc_tests";
    fs::create_directories(d);
    return d;
}

uint64_t seed_of(const std::string& s) {
    return llmoc::lwc::fnv1a64(s.data(), s.size());
}

// fixture: layers=3, experts=4, 每 part 8KiB
struct Fixture {
    llmoc::lwc::Header header;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> payloads;
};

Fixture make_fixture() {
    using namespace llmoc::lwc;
    Fixture fx;
    fx.header.dtype = Dtype::BF16;
    fx.header.block_align = 4096;
    constexpr int kLayers = 3, kExperts = 4, kPart = 8 << 10;

    for (int l = 0; l < kLayers; ++l) {
        llmoc::lwc::GroupMeta g;
        g.layer = static_cast<uint32_t>(l);
        for (int e = 0; e < kExperts; ++e) {
            for (const char* part : {"gate", "up", "down"}) {
                const std::string name =
                    llmoc::wt::ExpertPrefetcher::part_name(l, e, part);
                const uint64_t seed = seed_of(name);
                std::vector<uint8_t> buf(kPart);
                for (size_t i = 0; i < buf.size(); ++i)
                    buf[i] = static_cast<uint8_t>(seed + i * 131u);
                fx.payloads.emplace_back(name, std::move(buf));
                TensorMeta t;
                t.name = name;
                t.shape = {kPart / 2};
                fx.header.tensors.push_back(std::move(t));
                g.tensor_names.push_back(name);
            }
        }
        fx.header.groups.push_back(g);
    }
    return fx;
}

bool block_eq(const llmoc::wt::BlockRef& b, const std::vector<uint8_t>& expect) {
    if (b.bytes != expect.size() || b.data == nullptr) return false;
    for (size_t i = 0; i < expect.size(); ++i)
        if (b.data[i] != expect[i]) return false;
    return true;
}

std::vector<uint8_t> expect_of(const std::string& name, size_t n) {
    std::vector<uint8_t> buf(n);
    const uint64_t seed = seed_of(name);
    for (size_t i = 0; i < n; ++i) buf[i] = static_cast<uint8_t>(seed + i * 131u);
    return buf;
}

std::vector<llmoc::wt::ExpertData> acquire_checked(
    llmoc::wt::ExpertPrefetcher& pf, int layer, const std::vector<int>& ids) {
    std::vector<llmoc::wt::ExpertData> out;
    pf.acquire(layer, out);
    EXPECT_EQ(out.size(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int e = ids[i];
        EXPECT_TRUE(block_eq(out[i].gate,
                             expect_of(llmoc::wt::ExpertPrefetcher::part_name(layer, e, "gate"), 8 << 10)));
        EXPECT_TRUE(block_eq(out[i].up,
                             expect_of(llmoc::wt::ExpertPrefetcher::part_name(layer, e, "up"), 8 << 10)));
        EXPECT_TRUE(block_eq(out[i].down,
                             expect_of(llmoc::wt::ExpertPrefetcher::part_name(layer, e, "down"), 8 << 10)));
    }
    return out;
}

}  // namespace

TINY_TEST(Pipe, CorrectnessAcrossSlotReuse) {
    using namespace llmoc::wt;
    const auto file = tmp_dir() / "pipe_reuse.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    ExpertPrefetcher::Config cfg;
    cfg.slot_bytes = 96u << 10;   // 一层 topk=3×3×8KiB=72KiB 放得下; 槽必须跨层复用
    cfg.slots = 2;

    ExpertPrefetcher pf;
    pf.open(file, cfg);

    const std::vector<int> r0 = {0, 1, 2};
    const std::vector<int> r1 = {1, 2, 3};
    const std::vector<int> r2 = {0, 3};

    pf.plan_next_layer(0, r0);
    auto v0 = acquire_checked(pf, 0, r0);
    EXPECT_EQ(v0.size(), r0.size());
    pf.release(0);

    pf.plan_next_layer(1, r1);
    auto v1 = acquire_checked(pf, 1, r1);
    EXPECT_EQ(v1.size(), r1.size());
    pf.release(1);

    pf.plan_next_layer(2, r2);
    auto v2 = acquire_checked(pf, 2, r2);
    EXPECT_EQ(v2.size(), r2.size());
    pf.release(2);

    auto st = pf.stats();
    EXPECT_EQ(st.layer_plans, 3ull);
    // 顶层 topk=2 无 pin: 全部走磁盘
    EXPECT_EQ(st.block_reads_disk, (3 + 3 + 2) * 3ull);

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Pipe, PinNeverRereadsDisk) {
    using namespace llmoc::wt;
    const auto file = tmp_dir() / "pipe_pin.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    ExpertPrefetcher::Config cfg;
    cfg.slot_bytes = 96u << 10;
    cfg.pinned = {{0, 0}};

    ExpertPrefetcher pf;
    pf.open(file, cfg);

    pf.plan_next_layer(0, {0, 1});
    (void)acquire_checked(pf, 0, {0, 1});
    const auto disk_after_first = pf.stats().block_reads_disk;
    pf.release(0);

    // 第二轮: 热专家0 不再产生磁盘读; 冷专家1 正常重读(+3 块)
    pf.plan_next_layer(0, {0, 1});
    (void)acquire_checked(pf, 0, {0, 1});
    EXPECT_EQ(pf.stats().block_reads_disk, disk_after_first + 3);
    EXPECT_TRUE(pf.stats().block_reads_pin >= 3);

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Pipe, PinBudgetDemoteIsAtomic) {
    using namespace llmoc::wt;
    // 回归: pin 预算不足时必须整专家降级, 不允许半 pin 状态(旧实现只擦 gate 键,
    // acquire 阶段 up/down 从 pin_store_.at() 取值直接越界)
    const auto file = tmp_dir() / "pipe_pinbudget.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    ExpertPrefetcher::Config cfg;
    cfg.slot_bytes = 96u << 10;
    cfg.pinned = {{0, 0}};
    cfg.pin_budget_bytes = 4096;  // 远小于单块 8KiB -> 必然整体降级

    ExpertPrefetcher pf;
    pf.open(file, cfg);
    pf.plan_next_layer(0, {0});
    auto out = acquire_checked(pf, 0, {0});   // 不得抛异常
    EXPECT_EQ(out.size(), 1ull);
    EXPECT_TRUE(out[0].from_pin == false);    // 已降级 -> 走磁盘槽
    EXPECT_EQ(pf.stats().block_reads_disk, 3ull);
    pf.release(0);

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Pipe, DoublePlanThrows) {
    using namespace llmoc::wt;
    const auto file = tmp_dir() / "pipe_dup.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    {
        ExpertPrefetcher pf;
        pf.open(file, {/*slot=*/96u << 10, /*slots=*/2});
        pf.plan_next_layer(1, {0});
        bool threw = false;
        try {
            pf.plan_next_layer(1, {1});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        EXPECT_TRUE(threw);
        // Drain/stop IO before unlinking the file (destructor also closes).
        pf.close();
    }

    std::error_code ec;
    fs::remove(file, ec);
}
