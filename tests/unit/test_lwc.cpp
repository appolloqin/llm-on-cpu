// llm-on-cpu :: tests/unit/test_lwc.cpp
// LWC 容器格式回环 / 对齐 / 完整性 测试（main 在 test_main.cpp）
#include "test_main.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "weights/lwc_format.h"

namespace fs = std::filesystem;

namespace {

fs::path tmp_dir() {
    auto d = fs::temp_directory_path() / "llmoc_tests";
    fs::create_directories(d);
    return d;
}

std::pair<llmoc::lwc::Header, std::vector<std::pair<std::string, std::vector<uint8_t>>>>
make_fixture(uint32_t align) {
    using namespace llmoc::lwc;
    std::mt19937_64 rng(20260827u);
    std::vector<std::pair<std::string, std::vector<uint8_t>>> payloads;

    // 模拟一个 mini-MoE: 2 层 × 4 专家 × {gate,up,down}, 外加 embedding/attn 稠密块
    constexpr int kLayers = 2, kExperts = 4, kHiddenDim = 128;
    Header partial;
    partial.dtype = Dtype::BF16;
    partial.block_align = align;

    auto push = [&](const std::string& name, size_t nbytes) {
        std::vector<uint8_t> buf(nbytes);
        std::generate(buf.begin(), buf.end(), [&] { return static_cast<uint8_t>(rng()); });
        payloads.emplace_back(name, std::move(buf));
        TensorMeta t;
        t.name = name;
        t.shape = {nbytes / 2};
        partial.tensors.push_back(std::move(t));
    };

    push("embedding.weight", kHiddenDim * 1024 * 2);
    push("layers.0.attn.qkv.weight", 512 * 1024);
    for (int l = 0; l < kLayers; ++l) {
        GroupMeta g;
        g.layer = static_cast<uint32_t>(l);
        for (int e = 0; e < kExperts; ++e) {
            const std::string base =
                "layers." + std::to_string(l) + ".experts." + std::to_string(e);
            for (const char* part : {"gate", "up", "down"})
                push(base + "." + part, (e == 0 && l == 0) ? 30000 : 65536);
            g.tensor_names.push_back(base + ".gate");
            g.tensor_names.push_back(base + ".up");
            g.tensor_names.push_back(base + ".down");
        }
        partial.groups.push_back(g);
    }
    return {partial, payloads};
}

}  // namespace

TINY_TEST(Lwc, RoundTrip) {
    using namespace llmoc::lwc;
    const auto dir = tmp_dir();
    const auto file = dir / "roundtrip.lwc";
    auto [partial, payloads] = make_fixture(4096);

    Header written = Write(file, partial, payloads);
    Header back = ReadHeader(file);

    EXPECT_EQ(static_cast<uint32_t>(back.dtype), static_cast<uint32_t>(Dtype::BF16));
    EXPECT_EQ(back.block_align, 4096u);
    EXPECT_EQ(back.tensors.size(), written.tensors.size());
    EXPECT_EQ(back.groups.size(), written.groups.size());

    for (size_t i = 0; i < payloads.size(); ++i) {
        const auto& meta = back.tensors[i];
        const auto& pl = payloads[i].second;
        EXPECT_TRUE(meta.offset % 4096 == 0);
        EXPECT_EQ(meta.nbytes, pl.size());
        EXPECT_EQ(meta.checksum, fnv1a64(pl.data(), pl.size()));
        auto loaded = ReadTensor(file, back, meta.name);
        EXPECT_TRUE(loaded == pl);
    }

    // 专家组内张量区间严格升序且互不交叠(允许块间对齐空洞)
    for (const auto& g : back.groups) {
        const uint32_t align = back.block_align;
        for (size_t i = 1; i < g.tensor_names.size(); ++i) {
            const auto* a = back.find(g.tensor_names[i - 1]);
            const auto* b = back.find(g.tensor_names[i]);
            EXPECT_TRUE(a && b);
            EXPECT_TRUE(a->offset < b->offset);
            const uint64_t padded_end =
                (a->offset + a->nbytes + align - 1) / align * align;
            EXPECT_TRUE(padded_end <= b->offset + align);
        }
    }
    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Lwc, LastBlockAlignmentPaddingAtEof) {
    using namespace llmoc::lwc;
    const auto dir = tmp_dir();
    const auto file = dir / "tailpad.lwc";
    auto [partial, payloads] = make_fixture(4096);
    // 只保留前三个张量(最后一个长度非对齐), 验证 EOF 处对齐空洞不影响读回
    // 注: tensors 与 payloads 必须同序同长截断
    partial.tensors.resize(3);
    payloads.resize(3);

    Header h = Write(file, partial, payloads);
    Header back = ReadHeader(file);
    const auto& last = back.tensors.back();
    EXPECT_EQ(last.nbytes, payloads.back().second.size());
    auto loaded = ReadTensor(file, back, last.name);
    EXPECT_TRUE(loaded == payloads.back().second);

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Lwc, ChecksumDetectsCorruption) {
    using namespace llmoc::lwc;
    const auto dir = tmp_dir();
    const auto file = dir / "corrupt.lwc";
    auto [partial, payloads] = make_fixture(4096);
    Write(file, partial, payloads);
    Header h = ReadHeader(file);

    const auto& victim = h.tensors.front();
    std::fstream io(file, std::ios::binary | std::ios::in | std::ios::out);
    io.seekp(static_cast<std::streamoff>(victim.offset + 11));
    char flipped = 'X';
    io.read(&flipped, 1);
    flipped ^= 0xFF;
    io.seekp(static_cast<std::streamoff>(victim.offset + 11));
    io.write(&flipped, 1);
    io.close();

    bool threw = false;
    try {
        (void)ReadTensor(file, h, victim.name);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);

    std::error_code ec;
    fs::remove(file, ec);
}

TINY_TEST(Lwc, FindMissingReturnsNull) {
    using namespace llmoc::lwc;
    auto [partial, payloads] = make_fixture(256);
    EXPECT_TRUE(partial.find("nonexistent.tensor") == nullptr);
    EXPECT_TRUE(partial.find(payloads.front().first) != nullptr);
}
