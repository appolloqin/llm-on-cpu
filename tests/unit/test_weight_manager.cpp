// llm-on-cpu :: tests/unit/test_weight_manager.cpp
// 三级驻留(D1 骨架)行为测试: 常驻命中 / LRU 命中 / 冷拉取 / 预算淘汰
#include "test_main.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "weights/weight_manager.h"

namespace fs = std::filesystem;

namespace {

fs::path tmp_dir() {
    auto d = fs::temp_directory_path() / "llmoc_tests";
    fs::create_directories(d);
    return d;
}

// 构造 mini-MoE 固定内容文件: payload[i] = (fnv(name)+i) 低字节, 可独立校验正确性
struct Fixture {
    llmoc::lwc::Header header;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> payloads;
};

uint64_t seed_of(const std::string& s) {
    return llmoc::lwc::fnv1a64(s.data(), s.size());
}

Fixture make_fixture() {
    using namespace llmoc::lwc;
    Fixture f;
    Header& partial = f.header;
    partial.dtype = Dtype::BF16;
    partial.block_align = 4096;

    auto push = [&](const std::string& name, size_t nbytes) {
        std::vector<uint8_t> buf(nbytes);
        const uint64_t seed = seed_of(name);
        for (size_t i = 0; i < nbytes; ++i)
            buf[i] = static_cast<uint8_t>((seed + i * 131u));
        f.payloads.emplace_back(name, std::move(buf));

        TensorMeta t;
        t.name = name;
        t.shape = {nbytes / 2};
        partial.tensors.push_back(std::move(t));
    };

    push("embedding.weight", 128 << 10);            // 稠密 => 启发式常驻
    push("layers.0.attn.qkv.weight", 96 << 10);     // 稠密 => 常驻

    for (int l = 0; l < 2; ++l) {
        llmoc::lwc::GroupMeta g;
        g.layer = static_cast<uint32_t>(l);
        for (int e = 0; e < 3; ++e) {
            const std::string base =
                "layers." + std::to_string(l) + ".experts." + std::to_string(e);
            push(base + ".gate", 64 << 10);
            push(base + ".up", 64 << 10);
            push(base + ".down", 64 << 10);
            g.tensor_names.insert(g.tensor_names.end(),
                                  {base + ".gate", base + ".up", base + ".down"});
        }
        partial.groups.push_back(g);
    }
    return f;
}

std::vector<uint8_t> expect_payload(const std::string& name, size_t n) {
    std::vector<uint8_t> buf(n);
    const uint64_t seed = seed_of(name);
    for (size_t i = 0; i < n; ++i) buf[i] = static_cast<uint8_t>((seed + i * 131u));
    return buf;
}

}  // namespace

TINY_TEST(Wm, ResidentHeuristicAndHotPath) {
    using namespace llmoc::wt;
    const auto file = tmp_dir() / "wm_basic.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    WeightManager wm;
    wm.open(file, {/*force_resident=*/{}, /*budget=*/1u << 30, /*workers=*/2});

    // 1) 常驻稠密: 直接收放, 不产生冷拉取
    const auto& emb = wm.get("embedding.weight");
    EXPECT_TRUE(emb == fx.payloads[0].second);

    // 2) 冷专家首次访问 miss -> 拉取进 LRU -> 第二次 LRU hit
    const auto& g00 = wm.get("layers.0.experts.0.gate");
    EXPECT_TRUE(g00 == fx.payloads[2].second);
    const auto& g00b = wm.get("layers.0.experts.0.gate");

    auto st = wm.stats();
    EXPECT_EQ(st.resident_hits, 1ull);
    EXPECT_EQ(st.cold_misses, 1ull);
    EXPECT_EQ(st.lru_hits, 1ull);
    EXPECT_EQ(g00b.size(), (64 << 10));

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Wm, BudgetEvictionAndReloadIntegrity) {
    using namespace llmoc::wt;
    const auto file = tmp_dir() / "wm_evict.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    WeightManager wm;
    // 预算只够 ~3 个 64KiB 专家块(192KiB < 512KiB 总专家量), 必然触发淘汰
    wm.open(file, {{}, /*budget=*/192u << 10, /*workers=*/2});

    const char* seq[] = {"layers.0.experts.0.gate",
                         "layers.0.experts.1.gate",
                         "layers.0.experts.2.gate",
                         "layers.1.experts.0.gate"};
    for (const char* n : seq) {
        (void)wm.get(n);
    }
    // 拉第 5 个不同块后, 已超预算 => 必有淘汰发生
    (void)wm.get("layers.1.experts.1.gate");
    auto st = wm.stats();
    EXPECT_TRUE(st.evictions >= 1);

    // 被淘汰的最早条目重读: 内容必须与写入时一致(cold 重拉完整性)
    // LRU 语义下最旧的是 layers.0.experts.0.gate —— 已被挤出则应再次冷拉
    const auto& back = wm.get("layers.0.experts.0.gate");
    const auto expected = expect_payload(seq[0], 64 << 10);
    EXPECT_TRUE(back == expected);

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Wm, UnknownTensorThrows) {
    using namespace llmoc::wt;
    const auto file = tmp_dir() / "wm_unknown.lwc";
    Fixture fx = make_fixture();
    Write(file, fx.header, fx.payloads);

    WeightManager wm;
    wm.open(file, {});
    bool threw = false;
    try {
        (void)wm.get("layers.9.experts.9.gate");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);

    std::error_code ec;
    fs::remove(file, ec);
}
