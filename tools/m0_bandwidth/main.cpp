// llm-on-cpu :: tools/m0_bandwidth/main.cpp
// M0 基线: DRAM 流式带宽(读/写/拷贝) + NVMe 顺序读(绕过页缓存)。
// 输出 JSON(hw_profile 片段) 回填 ARCHITECTURE §2 公式评审表。
//
// 用法:
//   m0_bandwidth [--gb 4] [--threads N] [--passes 3]
//                [--dir PATH] [--file-gb 1] [--skip-file] [--keep-file] [--out FILE]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

#include "common/alloc.h"
#include "common/cpuid.h"
#include "common/log.h"

namespace fs = std::filesystem;
using llmoc::mem::alloc_aligned;
using llmoc::mem::free_aligned;
using Clock = std::chrono::steady_clock;

namespace {

inline unsigned default_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n != 0 ? n : 2u;
}

struct Options {
    size_t ram_gb = 4;
    unsigned threads = default_threads();
    int passes = 3;
    std::string dir = ".";
    size_t file_gb = 1;
    bool skip_file = false;
    bool keep_file = false;
    const char* out = nullptr;
};

const char* shift(int& i, char** argv) { return argv[++i]; }

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gb")) o.ram_gb = std::strtoull(shift(i, argv), nullptr, 10);
        else if (!std::strcmp(argv[i], "--threads")) o.threads = std::strtoul(shift(i, argv), nullptr, 10);
        else if (!std::strcmp(argv[i], "--passes")) o.passes = atoi(shift(i, argv));
        else if (!std::strcmp(argv[i], "--dir")) o.dir = shift(i, argv);
        else if (!std::strcmp(argv[i], "--file-gb")) o.file_gb = std::strtoull(shift(i, argv), nullptr, 10);
        else if (!std::strcmp(argv[i], "--skip-file")) o.skip_file = true;
        else if (!std::strcmp(argv[i], "--keep-file")) o.keep_file = true;
        else if (!std::strcmp(argv[i], "--out")) o.out = shift(i, argv);
    }
    if (o.ram_gb < 1) o.ram_gb = 1;
    if (o.passes < 1) o.passes = 1;
    return o;
}

struct RamBw {
    double read_gbps = 0, write_gbps = 0, copy_gbps = 0;  // 有效搬运字节口径
};

template <typename F>
double best_of(int passes, F&& f) {
    double best = 0;
    for (int p = 0; p < passes; ++p) {
        const double v = f();
        if (v > best) best = v;
    }
    return best;
}

double gbps(size_t bytes, Clock::duration d) {
    const double s = std::chrono::duration<double>(d).count();
    return static_cast<double>(bytes) / 1e9 / (s > 0 ? s : 1e-9);
}

RamBw bench_ram(const Options& opt) {
    const size_t total = opt.ram_gb << 30;
    uint8_t* a = static_cast<uint8_t*>(alloc_aligned(total));
    uint8_t* b = static_cast<uint8_t*>(alloc_aligned(total));
    RamBw r{};

    const unsigned th = opt.threads;
    std::vector<std::thread> pool;
    pool.reserve(th);

    // ---- read (xor-reduce 防消除) ----
    for (int pass = 0; pass < opt.passes; ++pass) {
        auto t0 = Clock::now();
        for (unsigned i = 0; i < th; ++i) {
            pool.emplace_back([&, i] {
                const size_t chunk = total / th;
                const uint64_t* p = reinterpret_cast<const uint64_t*>(a + chunk * i);
                const size_t n = chunk / sizeof(uint64_t);
                uint64_t sink = 0x9E3779B97F4A7C15ull;
                for (size_t j = 0; j < n; ++j) sink ^= p[j];
                volatile uint64_t keep = sink;
                (void)keep;
            });
        }
        for (auto& t : pool) t.join();
        auto dt = Clock::now() - t0;
        const double v = gbps(total, dt);
        if (v > r.read_gbps) r.read_gbps = v;
        pool.clear();
    }

    // ---- write (fill) ----
    for (int pass = 0; pass < opt.passes; ++pass) {
        auto t0 = Clock::now();
        for (unsigned i = 0; i < th; ++i) {
            pool.emplace_back([&, i] {
                const size_t chunk = total / th;
                std::memset(b + chunk * i, 0xA5, chunk);
            });
        }
        for (auto& t : pool) t.join();
        auto dt = Clock::now() - t0;
        const double v = gbps(total, dt);
        if (v > r.write_gbps) r.write_gbps = v;
        pool.clear();
    }

    // ---- copy (a->b, 口径=2×bytes: 读+写) ----
    for (int pass = 0; pass < opt.passes; ++pass) {
        auto t0 = Clock::now();
        for (unsigned i = 0; i < th; ++i) {
            pool.emplace_back([&, i] {
                const size_t chunk = total / th;
                std::memcpy(b + chunk * i, a + chunk * i, chunk);
            });
        }
        for (auto& t : pool) t.join();
        auto dt = Clock::now() - t0;
        const double v = gbps(total * 2, dt);
        if (v > r.copy_gbps) r.copy_gbps = v;
        pool.clear();
    }

    free_aligned(a);
    free_aligned(b);
    return r;
}

// NVMe 顺序读(绕过页缓存)。返回有效 MB/s, 失败返回负值并带原因。
struct FileBw { double mbps = -1; std::string err; };

FileBw bench_file_seq_read(const fs::path& file, size_t file_bytes,
                           const char** which_api) {
    constexpr size_t kChunk = 1u << 20;  // 与扇区/对齐要求相容
    void* bufv = alloc_aligned(kChunk, 4096);
    auto* buf = static_cast<uint8_t*>(bufv);
    FileBw out;

#if defined(_WIN32)
    *which_api = "ReadFile(NO_BUFFERING)";
    HANDLE h = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        out.err = "CreateFileW failed";
        free_aligned(bufv);
        return out;
    }
    uint64_t done = 0;
    DWORD got = 0;
    auto t0 = Clock::now();
    while (done < file_bytes &&
           ReadFile(h, buf, kChunk, &got, nullptr) && got > 0) {
        const auto* u = reinterpret_cast<const uint64_t*>(buf);
        volatile uint64_t sink = u[0];
        (void)sink;
        done += got;
    }
    auto dt = Clock::now() - t0;
    CloseHandle(h);
#else
    *which_api = "read(O_DIRECT)";
    int fd = ::open(file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        out.err = "open(O_DIRECT) failed";
        free_aligned(bufv);
        return out;
    }
    uint64_t done = 0;
    auto t0 = Clock::now();
    while (done < file_bytes) {
        ssize_t got = ::read(fd, buf, kChunk);
        if (got <= 0) break;
        done += static_cast<uint64_t>(got);
    }
    auto dt = Clock::now() - t0;
    ::close(fd);
#endif

    if (done == 0) {
        out.err = "no data read";
    } else {
        out.mbps = static_cast<double>(done) / (1024.0 * 1024.0) /
                   std::chrono::duration<double>(dt).count();
    }
    free_aligned(bufv);
    return out;
}

std::string hostname() {
#if defined(_WIN32)
    char buf[MAX_COMPUTERNAME_LENGTH + 1]{};
    DWORD len = MAX_COMPUTERNAME_LENGTH;
    if (GetComputerNameA(buf, &len)) return buf;
#else
    char buf[256]{};
    if (::gethostname(buf, sizeof(buf)) == 0) return buf;
#endif
    return "unknown-host";
}

constexpr size_t kSeqChunk = 1u << 20;  // 1MiB, 与扇区对齐要求相容

void run_file_bench(const Options& opt, std::ostringstream& js) {
    const fs::path dir(opt.dir);
    fs::create_directories(dir);
    const fs::path file = dir / "llmoc_m0_seq.tmp";
    const size_t file_bytes = opt.file_gb << 30;

    // 生成临时文件(非计时段), 整块写零以满足 NO_BUFFERING 的整块读约束
    {
        LOG_INFO("generating %zu GiB temp file under %s ...", opt.file_gb,
                 opt.dir.c_str());
        std::ofstream w(file, std::ios::binary | std::ios::trunc);
        if (!w) throw std::runtime_error("cannot create temp benchmark file");
        std::vector<char> block(kSeqChunk, '\0');
        for (size_t left = file_bytes; left > 0;) {
            const size_t n = std::min(left, block.size());
            w.write(block.data(), static_cast<std::streamsize>(n));
            left -= n;
        }
    }

    const char* api = "?";
    FileBw fb = bench_file_seq_read(file, file_bytes, &api);

    if (!opt.keep_file) {
        std::error_code ec;
        fs::remove(file, ec);
    }

    js << ",\n  \"nvme_seq_read\": {\"api\": \"" << api << "\", \"mbps\": "
       << (fb.mbps > 0 ? fb.mbps : -1) << ", \"error\": \"" << fb.err << "\"}";
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);

    LOG_INFO("M0 DRAM bench: %.1f GiB buffer x %u threads x %d passes ...",
             static_cast<double>(opt.ram_gb), opt.threads, opt.passes);
    const RamBw ram = bench_ram(opt);

    std::ostringstream js;
    js << "{\n  \"host\": \"" << hostname() << "\",\n"
       << "  \"cpu\": \"" << llmoc::cpu::brand_string() << "\",\n"
       << "  \"logical_cores\": " << opt.threads << ",\n"
       << "  \"ram_buffer_gb\": " << opt.ram_gb << ",\n"
       << "  \"dram\": {\"read_gbps\": " << ram.read_gbps
       << ", \"write_gbps\": " << ram.write_gbps
       << ", \"copy_eff_gbps\": " << ram.copy_gbps << "}";
    if (!opt.skip_file) run_file_bench(opt, js);
    js << "\n}\n";

    const std::string text = js.str();
    std::fputs(text.c_str(), stdout);
    if (opt.out) {
        std::ofstream of(opt.out, std::ios::binary | std::ios::trunc);
        of.write(text.data(), static_cast<std::streamsize>(text.size()));
        LOG_INFO("profile written to %s", opt.out);
    }
    return 0;
}
