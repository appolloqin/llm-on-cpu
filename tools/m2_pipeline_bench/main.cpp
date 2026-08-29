// llm-on-cpu :: tools/m2_pipeline_bench/main.cpp
// M2 基线: 单流 decode 的「同步 vs 双缓冲预取流水线」对比(D2 收益实证)。
// 真实 IO(生成的 .lwc 文件) + 模拟计算(sleep 代言 AMX GEMM 时隙)。
// dev-machine 数字不代表 SPR, 但"IO 藏进计算"的结构性收益可复现。
//
// 用法:
//   m2_pipeline_bench [--layers 8] [--experts 32] [--part-mib 2]
//                     [--topk 8] [--compute-us 5000]
//                     [--pin-experts N] [--dir bench] [--reuse-file] [--json FILE]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/log.h"
#include "common/platform.h"
#include "weights/prefetch_pipeline.h"

namespace fs = std::filesystem;
using llmoc::wt::ExpertPrefetcher;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    int layers = 8;
    int experts = 32;
    int part_mib = 2;
    int topk = 8;
    int compute_us = 5000;
    int pin_experts = 0;
    std::string dir = "bench";
    bool reuse_file = false;
    std::string file;             // --file: 直接使用真实 .lwc(结构由 header 推导)
    const char* json = nullptr;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto val = [&]() -> const char* { return argv[++i]; };
        if (!std::strcmp(argv[i], "--layers")) o.layers = atoi(val());
        else if (!std::strcmp(argv[i], "--experts")) o.experts = atoi(val());
        else if (!std::strcmp(argv[i], "--part-mib")) o.part_mib = atoi(val());
        else if (!std::strcmp(argv[i], "--topk")) o.topk = atoi(val());
        else if (!std::strcmp(argv[i], "--compute-us")) o.compute_us = atoi(val());
        else if (!std::strcmp(argv[i], "--pin-experts")) o.pin_experts = atoi(val());
        else if (!std::strcmp(argv[i], "--dir")) o.dir = val();
        else if (!std::strcmp(argv[i], "--reuse-file")) o.reuse_file = true;
        else if (!std::strcmp(argv[i], "--file")) o.file = val();
        else if (!std::strcmp(argv[i], "--json")) o.json = val();
    }
    if (o.layers < 1) o.layers = 1;
    if (o.experts < 1) o.experts = 1;
    if (o.topk < 1) o.topk = 1;
    if (o.topk > o.experts) o.topk = o.experts;
    if (o.part_mib < 1) o.part_mib = 1;
    return o;
}

uint64_t seed_like(const std::string& s) {
    return llmoc::lwc::fnv1a64(s.data(), s.size());
}

void filler_fill(std::vector<uint8_t>& v, uint64_t seed) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(seed + i * 131u);
}

// 确定 top-k 路由: 每层选 (L*7+3) 起步等差 5 序列, 可复现
std::vector<int> route_for(const Options& o, int layer) {
    std::vector<int> ids;
    const int start = (layer * 7 + 3) % o.experts;
    for (int k = 0; k < o.topk; ++k) ids.push_back((start + k * 5) % o.experts);
    return ids;
}

// 生成 .lwc: 每 expert {gate,up,down} 各 part_bytes; 内容按名字种子填充
llmoc::lwc::Header generate(const fs::path& file, const Options& o, size_t part_bytes) {
    using namespace llmoc::lwc;
    Header partial;
    partial.dtype = Dtype::BF16;
    partial.block_align = 4096;

    std::vector<std::pair<std::string, std::vector<uint8_t>>> payloads;
    std::vector<uint8_t> pattern(65536);

    for (int l = 0; l < o.layers; ++l) {
        llmoc::lwc::GroupMeta g;
        g.layer = static_cast<uint32_t>(l);
        for (int e = 0; e < o.experts; ++e) {
            for (const char* part : {"gate", "up", "down"}) {
                const std::string name =
                    ExpertPrefetcher::part_name(l, e, part);
                filler_fill(pattern, seed_like(name));
                std::vector<uint8_t> buf(part_bytes);
                for (size_t i = 0; i < buf.size(); ++i)
                    buf[i] = pattern[i & (pattern.size() - 1)];
                payloads.emplace_back(name, std::move(buf));
                TensorMeta t;
                t.name = name;
                t.shape = {part_bytes / 2};
                partial.tensors.push_back(std::move(t));
                g.tensor_names.push_back(name);
            }
        }
        partial.groups.push_back(g);
    }
    LOG_INFO("generating model file: %d tensors, %.1f MiB under %s ...",
             static_cast<int>(payloads.size()),
             static_cast<double>(part_bytes) * static_cast<double>(payloads.size()) /
                 (1024.0 * 1024.0),
             file.string().c_str());
    return Write(file, partial, payloads);
}

struct PassResult {
    double wall_s = 0;
    double io_wait_s = 0;
    llmoc::wt::ExpertPrefetcher::Stats stats;
};

PassResult run_pass(const Options& o, const fs::path& file,
                    size_t part_bytes, bool pipelined) {
    ExpertPrefetcher::Config cfg;
    // 单槽容量: 一层 top-k 专家 × 3 块, 留 2MiB 余量
    cfg.slot_bytes = static_cast<size_t>(o.topk) * 3 * part_bytes + (2u << 20);
    cfg.slots = 2;
    cfg.io_workers = 2;
    std::vector<llmoc::wt::ExpertKey> pins;
    for (int e = 0; e < o.pin_experts; ++e) pins.push_back({0, e});  // 第0层前N个为热
    cfg.pinned = pins;

    ExpertPrefetcher pf;
    pf.open(file, cfg);

    const auto compute = std::chrono::microseconds(o.compute_us);
    PassResult res;
    std::vector<llmoc::wt::ExpertData> data;

    const auto acquire_timed = [&](int layer) {
        auto t0 = Clock::now();
        pf.acquire(layer, data);
        res.io_wait_s += std::chrono::duration<double>(Clock::now() - t0).count();
    };

    auto t_all = Clock::now();

    // 第 0 层: 两模式都必须先读后算(冷启动不可避免)
    pf.plan_next_layer(0, route_for(o, 0));
    acquire_timed(0);
    if (o.compute_us > 0) std::this_thread::sleep_for(compute);
    pf.release(0);

    for (int l = 1; l < o.layers; ++l) {
        if (pipelined) {
            // gate(L-1) 已知 => 立即发起层 l 预取
            pf.plan_next_layer(l, route_for(o, l));
            if (o.compute_us > 0) std::this_thread::sleep_for(compute);  // 重叠窗口
            acquire_timed(l);                                            // 只等残余
        } else {
            // 同步基线: 读完全部阻塞在关键路径, 再计算
            pf.plan_next_layer(l, route_for(o, l));
            acquire_timed(l);
            if (o.compute_us > 0) std::this_thread::sleep_for(compute);
        }
        pf.release(l);
    }
    res.wall_s = std::chrono::duration<double>(Clock::now() - t_all).count();
    res.stats = pf.stats();
    pf.close();
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    Options o = parse(argc, argv);

    fs::create_directories(o.dir);

    // ---- 模型来源二选一: --file 真实 LWC(结构由 header 推导), 否则合成 ----
    size_t part_bytes = static_cast<size_t>(o.part_mib) << 20;
    fs::path file;
    if (!o.file.empty()) {
        file = o.file;
        const auto hdr = llmoc::lwc::ReadHeader(file);
        std::set<int> layer_set;
        int max_eid = 0;
        for (const auto& g : hdr.groups) {
            layer_set.insert(static_cast<int>(g.layer));
            if (static_cast<int>(g.expert_id) + 1 > max_eid)
                max_eid = static_cast<int>(g.expert_id) + 1;
        }
        o.layers = static_cast<int>(layer_set.size());
        o.experts = max_eid;
        const auto* first = hdr.find(ExpertPrefetcher::part_name(0, 0, "gate"));
        if (!first)
            throw std::runtime_error("--file: header lacks layers.0.experts.0.gate");
        part_bytes = first->nbytes;
        LOG_INFO("using real model file: %s (layers=%d experts=%d part=%.1f KiB)",
                 file.string().c_str(), o.layers, o.experts,
                 static_cast<double>(part_bytes) / 1024.0);
    } else {
        file = fs::path(o.dir) / "m2_model.lwc";
        if (!(o.reuse_file && fs::exists(file))) {
            auto hdr = generate(file, o, part_bytes);
            (void)hdr;
        } else {
            LOG_INFO("reusing existing model file: %s", file.string().c_str());
        }
    }
    if (o.topk > o.experts) o.topk = o.experts;
    if (o.pin_experts > o.experts) o.pin_experts = o.experts;

    LOG_INFO("== M2 pipeline bench (%s): layers=%d topk=%d part=%.1f KiB compute=%dus ==",
             llmoc::sys::os_name(), o.layers, o.topk,
             static_cast<double>(part_bytes) / 1024.0, o.compute_us);

    const PassResult sync_r = run_pass(o, file, part_bytes, /*pipelined=*/false);
    const PassResult pipe_r = run_pass(o, file, part_bytes, /*pipelined=*/true);

    const double sync_tps = o.layers / sync_r.wall_s;
    const double pipe_tps = o.layers / pipe_r.wall_s;

    std::printf("\n%-12s %10s %12s %12s %14s\n", "mode", "wall_s", "tok/s",
                "io_wait_s", "disk_MiB");
    auto row = [&](const char* name, const PassResult& r) {
        std::printf("%-12s %10.3f %12.2f %12.3f %14.1f\n", name, r.wall_s,
                    static_cast<double>(o.layers) / r.wall_s, r.io_wait_s,
                    static_cast<double>(r.stats.bytes_read_disk) / (1024.0 * 1024.0));
    };
    row("sync", sync_r);
    row("pipeline", pipe_r);
    std::printf("speedup: %.2fx  (io_wait %.1fms -> %.1fms)\n",
                pipe_tps / sync_tps, sync_r.io_wait_s * 1e3, pipe_r.io_wait_s * 1e3);

    if (o.json) {
        std::ofstream jf(o.json, std::ios::binary | std::ios::trunc);
        std::ostringstream js;
        js << "{\n"
           << "  \"layers\": " << o.layers << ", \"topk\": " << o.topk
           << ", \"part_mib\": " << o.part_mib << ", \"compute_us\": " << o.compute_us
           << ",\n  \"sync\":  {\"wall_s\": " << sync_r.wall_s << ", \"tps\": "
           << sync_tps << ", \"io_wait_s\": " << sync_r.io_wait_s << "},\n"
           << "  \"pipeline\": {\"wall_s\": " << pipe_r.wall_s << ", \"tps\": "
           << pipe_tps << ", \"io_wait_s\": " << pipe_r.io_wait_s << "},\n"
           << "  \"speedup\": " << (pipe_tps / sync_tps) << "\n}\n";
        jf.write(js.str().data(), static_cast<std::streamsize>(js.str().size()));
        LOG_INFO("json written to %s", o.json);
    }
    return 0;
}
