#pragma once
// llm-on-cpu :: weights/prefetch_pipeline.h
// M2 核心: decode 路径专家预取流水线 (ARCHITECTURE D2)。
//
// 时序契约(单流 decode):
//   plan_next_layer(L+1, ids)   <- 层 L 的 gate 决策点, 立即发起 L+1 冷块异步读
//   simulate_compute(L)         <- 计算与 IO 重叠窗口
//   acquire(L+1)                <- 只等待残余 IO (内部 drain 栅栏)
//   release(L+1)                <- 归还槽位供 L+2 复用 (双缓冲轮转)
//
// 热专家(D1): Config.pinned 列表进入 RAM 常驻(受 pin_budget 限制),
// 之后永不再走磁盘 —— 对应 FreeToken 观察到的路由偏斜收益。
//
// 并发契约: 单线程 owner(前向线程), 与 WeightManager 相同。

#include <cstdint>
#include <deque>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/alloc.h"
#include "weights/io_engine.h"
#include "weights/lwc_format.h"

namespace llmoc::wt {

struct ExpertKey {
    int layer = 0;
    int expert = 0;
    bool operator==(const ExpertKey& o) const {
        return layer == o.layer && expert == o.expert;
    }
};

struct BlockRef {
    const uint8_t* data = nullptr;
    size_t bytes = 0;
};

struct ExpertData {
    BlockRef gate, up, down;
    bool from_pin = false;
};

class ExpertPrefetcher {
   public:
    struct Config {
        size_t slot_bytes = 64u << 20;   // 单槽容量(容纳一层 top-k 专家三块)
        uint32_t slots = 2;              // 双缓冲(D2)
        unsigned io_workers = 2;
        std::vector<ExpertKey> pinned;   // 热专家 pin 列表(D1)
        size_t pin_budget_bytes = 0;     // 0=不限(测试/小模型); 生产由 D6 调拨
    };

    struct Stats {
        uint64_t layer_plans = 0;
        uint64_t block_reads_disk = 0;
        uint64_t block_reads_pin = 0;
        uint64_t bytes_read_disk = 0;
        uint64_t pin_bytes = 0;
    };

    void open(const io::Path& file, const Config& cfg);
    void close();
    ~ExpertPrefetcher() { close(); }

    ExpertPrefetcher() = default;
    ExpertPrefetcher(const ExpertPrefetcher&) = delete;
    ExpertPrefetcher& operator=(const ExpertPrefetcher&) = delete;

    // gate 决策点: 记录下一层所需专家并立即发起磁盘预取(热专家除外)。
    // 同层重复 plan 抛错(状态机保护)。
    void plan_next_layer(int layer, const std::vector<int>& expert_ids);

    // 层边界: 阻塞至该层所有在途读完成, 填充 out(每专家三块引用)。
    void acquire(int layer, std::vector<ExpertData>& out);

    // 层消费完毕: 归还槽位。pinned 数据不受影响。
    void release(int layer);

    const lwc::Header& header() const { return hdr_; }
    Stats stats() const { return st_; }

    static std::string part_name(int layer, int expert, const char* part);

   private:
    struct AlignedBuf {
        uint8_t* p = nullptr;
        size_t cap = 0;
        explicit AlignedBuf(size_t bytes)
            : p(static_cast<uint8_t*>(mem::alloc_aligned(bytes))), cap(bytes) {}
        ~AlignedBuf() {
            if (p) mem::free_aligned(p);
        }
        AlignedBuf(const AlignedBuf&) = delete;
        AlignedBuf& operator=(const AlignedBuf&) = delete;
    };

    struct Slot {
        std::unique_ptr<AlignedBuf> buf;
        size_t used = 0;  // 已打包字节
        std::error_code err;  // 任一在途读失败时置位, acquire 时抛出
        // name -> {slot 内偏移, 字节数}
        std::unordered_map<std::string, std::pair<size_t, size_t>> span;
    };

    struct LayerPlan {
        int layer = -1;
        int slot = -1;                       // -1: 全部命中 pin, 不占槽
        bool acquired = false;
        std::vector<ExpertKey> ids;
    };

    const lwc::TensorMeta& meta_of(const std::string& name) const;
    size_t pack_into_slot(Slot& s, const lwc::TensorMeta& m);   // 返回偏移
    void load_pin_if_needed(const ExpertKey& k);
    bool is_pinned(const ExpertKey& k) const;

    lwc::Header hdr_;
    Config cfg_;
    io::Path file_;
    std::unique_ptr<io::IoEngine> engine_;

    std::vector<Slot> slots_;
    std::deque<size_t> free_slots_;
    std::unordered_map<int, LayerPlan> plans_;

    std::unordered_set<std::string> pin_keys_;             // part 名集合
    std::unordered_map<std::string, std::vector<uint8_t>> pin_store_;

    Stats st_{};
};

}  // namespace llmoc::wt
