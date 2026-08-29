#pragma once
// llm-on-cpu :: weights/weight_manager.h
// 三级驻留(D1)最小可跑骨架: 常驻区(mlock) / LRU 热专家区(预算内) / 冷区(按需拉取)。
// 模式无关 —— tier 结构对 pure-cpu/hybrid/pure-gpu 透明, 遵循 ARCHITECTURE §3.1 统一原则。
//
// 并发契约: 单线程持有(前向线程池 owner)。多请求并发由 scheduler 上游串行化(M4 引入)。

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "weights/io_engine.h"
#include "weights/lwc_format.h"

namespace llmoc::wt {

enum class Tier : uint8_t { kResident = 0, kHotLru = 1, kCold = 2 };

struct WmStats {
    uint64_t resident_hits = 0;
    uint64_t lru_hits = 0;
    uint64_t cold_misses = 0;   // 冷专家被动拉取次数
    uint64_t evictions = 0;     // LRU 淘汰次数
    uint64_t locked_bytes = 0;  // mlock 成功字节数
};

class WeightManager {
   public:
    struct Config {
        // 强制常驻的张量名(优先级高于启发式); 未列出的稠密张量按
        // "不隶属任何专家组 => 常驻" 启发式归入常驻区。
        std::vector<std::string> force_resident;
        uint64_t lru_budget_bytes = 256u << 20;  // 热专家区预算(D6 调拨器后接管)
        unsigned io_workers = 2;
    };

    void open(const io::Path& file, const Config& cfg);
    void close();

    // 返回张量数据指针; 冷块自动拉取并进入 LRU。
    // 未知名字抛 std::runtime_error。
    const std::vector<uint8_t>& get(const std::string& tensor_name);

    const lwc::Header& header() const { return hdr_; }
    WmStats stats() const { return st_; }
    uint64_t lru_used_bytes() const { return lru_used_; }

   private:
    struct Entry {
        lwc::TensorMeta meta;
        Tier tier = Tier::kCold;
        std::vector<uint8_t> data;
        bool resident_loaded = false;
        uint64_t last_use = 0;
        bool pages_locked = false;
    };

    void load_now(Entry& e);
    void touch(Entry& e);
    void evict_until_budget();

    lwc::Header hdr_;
    Config cfg_;
    io::Path file_;
    std::unique_ptr<io::IoEngine> engine_;

    std::unordered_map<std::string, Entry> entries_;
    std::vector<std::string> order_;           // 插入序(用于确定性遍历/预热)
    std::unordered_set<std::string> expert_names_;
    uint64_t clock_ = 1;
    uint64_t lru_used_ = 0;

    WmStats st_{};
};

}  // namespace llmoc::wt
