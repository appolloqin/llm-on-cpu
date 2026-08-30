// llm-on-cpu :: weights/weight_manager.cpp
#include "weights/weight_manager.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "common/log.h"
#include "common/platform.h"

namespace llmoc::wt {

void WeightManager::open(const io::Path& file, const Config& cfg) {
    close();
    file_ = file;
    cfg_ = cfg;
    hdr_ = lwc::ReadHeader(file_);
    engine_ = io::make_engine(cfg_.io_workers);

    // 专家成员表(头部 groups) => 其余全是稠密权重 => 常驻启发式
    for (const auto& g : hdr_.groups)
        for (const auto& n : g.tensor_names) expert_names_.insert(n);

    std::vector<std::string> force(cfg_.force_resident.begin(),
                                   cfg_.force_resident.end());
    const auto is_force_resident = [&](const std::string& n) {
        return std::find(force.begin(), force.end(), n) != force.end();
    };

    for (const auto& t : hdr_.tensors) {
        Entry e;
        e.meta = t;
        e.tier = (!expert_names_.count(t.name) || is_force_resident(t.name))
                     ? Tier::kResident
                     : Tier::kCold;
        entries_.emplace(t.name, std::move(e));
        order_.push_back(t.name);
    }

    // 常驻区就绪: 逐个加载并尽力锁页。冷启动 IO 是唯一大头, 引擎异步性在
    // M2 预热流水线(D2 prefill 路径)时启用, 当前为顺序拉齐保证语义先正确。
    uint64_t resident_bytes = 0;
    for (auto& [name, e] : entries_) {
        if (e.tier == Tier::kResident && !e.resident_loaded) {
            load_now(e);
            e.resident_loaded = true;
            resident_bytes += e.data.size();
            if (sys::lock_pages(e.data.data(), e.data.size()))
                st_.locked_bytes += e.data.size();
        }
    }
    const double resident_gib = static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double locked_mib = static_cast<double>(st_.locked_bytes) / (1024.0 * 1024.0);
    LOG_INFO("weight manager ready: %zu tensors (%zu resident, %zu experts), "
             "resident=%.1f GiB locked=%.1f MiB",
             entries_.size(), entries_.size() - expert_names_.size(),
             expert_names_.size(), resident_gib, locked_mib);
    if (resident_bytes > (1ull << 30) && st_.locked_bytes * 20 < resident_bytes) {
        LOG_WARN("page lock nearly failed (locked << resident). Ensure RAM >= model+OS "
                 "(~64G for 27B BF16) and check Task Manager disk thrashing; enable "
                 "\"Lock pages in memory\" if you need VirtualLock. Prefer INT4 for chat.");
    }
}

void WeightManager::close() {
    if (engine_) engine_->drain();
    entries_.clear();
    order_.clear();
    expert_names_.clear();
    lru_used_ = 0;
    st_ = WmStats{};
    file_.clear();
}

void WeightManager::load_now(Entry& e) {
    if (e.data.empty()) {
        // io_engine::submit 需要“回调前保持有效”的缓冲, 同步路径直接预估容量后 drain。
        std::error_code ec_final{};
        e.data.resize(e.meta.nbytes);
        engine_->submit(file_, e.meta.offset, e.data.data(), e.meta.nbytes,
                        [&](std::error_code ec) { ec_final = ec; });
        engine_->drain();
        if (ec_final) throw std::runtime_error("load failed: " + e.meta.name);
    }
}

void WeightManager::touch(Entry& e) { e.last_use = ++clock_; }

void WeightManager::evict_until_budget() {
    while (lru_used_ > cfg_.lru_budget_bytes) {
        Entry* victim = nullptr;
        uint64_t best = ~0ull;
        for (const auto& name : order_) {
            auto& e = entries_[name];
            if (e.tier != Tier::kHotLru) continue;
            if (e.last_use < best) {
                best = e.last_use;
                victim = &e;
            }
        }
        if (!victim) break;  // 无 LRU 可逐(预算本身过小), 留待 D6 处理
        LOG_DEBUG("evict %s (%zu bytes)", victim->meta.name.c_str(),
                  victim->data.size());
        lru_used_ -= victim->data.size();
        victim->data.clear();
        victim->data.shrink_to_fit();
        victim->tier = Tier::kCold;
        ++st_.evictions;
    }
}

const std::vector<uint8_t>& WeightManager::get(const std::string& tensor_name) {
    auto it = entries_.find(tensor_name);
    if (it == entries_.end())
        throw std::runtime_error("unknown tensor: " + tensor_name);
    Entry& e = it->second;

    switch (e.tier) {
        case Tier::kResident:
            ++st_.resident_hits;
            return e.data;

        case Tier::kHotLru:
            ++st_.lru_hits;
            touch(e);
            return e.data;

        case Tier::kCold:
        default:
            ++st_.cold_misses;
            load_now(e);                       // 冷拉取(M3 后改为 gate 前预取)
            e.tier = Tier::kHotLru;            // 进入热区
            touch(e);
            lru_used_ += e.data.size();
            evict_until_budget();
            return e.data;
    }
}

}  // namespace llmoc::wt
