// llm-on-cpu :: weights/prefetch_pipeline.cpp
#include "weights/prefetch_pipeline.h"

#include <stdexcept>
#include <utility>

#include "common/log.h"
#include "common/platform.h"

namespace llmoc::wt {

namespace {
constexpr uint64_t kSlotAlign = 4096;  // 为未来 io_uring/O_DIRECT 预留对齐约束

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
}  // namespace

std::string ExpertPrefetcher::part_name(int layer, int expert, const char* part) {
    return "layers." + std::to_string(layer) + ".experts." +
           std::to_string(expert) + "." + part;
}

void ExpertPrefetcher::open(const io::Path& file, const Config& cfg) {
    close();
    file_ = file;
    cfg_ = cfg;
    hdr_ = lwc::ReadHeader(file_);
    engine_ = io::make_engine(cfg_.io_workers);

    slots_.resize(cfg_.slots);
    for (auto& s : slots_) {
        s.buf = std::make_unique<AlignedBuf>(cfg_.slot_bytes);
        s.used = 0;
    }
    for (uint32_t i = 0; i < cfg_.slots; ++i) free_slots_.push_back(i);

    for (const auto& k : cfg_.pinned) {
        for (const char* part : {"gate", "up", "down"})
            pin_keys_.insert(part_name(k.layer, k.expert, part));
    }
    LOG_INFO("prefetch pipeline ready: %u slots x %.1f MiB, %zu pinned experts (%s)",
             cfg_.slots, static_cast<double>(cfg_.slot_bytes) / (1024.0 * 1024.0),
             cfg_.pinned.size(), sys::os_name());
}

void ExpertPrefetcher::close() {
    // Drain then stop workers BEFORE freeing slot buffers — otherwise async
    // reads can write into freed AlignedBuf (glibc: corrupted double-linked list).
    if (engine_) {
        engine_->drain();
        engine_.reset();
    }
    plans_.clear();
    slots_.clear();
    free_slots_.clear();
    pin_store_.clear();
    pin_keys_.clear();
    st_ = Stats{};
}

bool ExpertPrefetcher::is_pinned(const ExpertKey& k) const {
    return pin_keys_.count(part_name(k.layer, k.expert, "gate")) != 0;
}

const lwc::TensorMeta& ExpertPrefetcher::meta_of(const std::string& name) const {
    const auto* m = hdr_.find(name);
    if (!m) throw std::runtime_error("prefetch: tensor not in header: " + name);
    return *m;
}

void ExpertPrefetcher::load_pin_if_needed(const ExpertKey& k) {
    // 预检: 三块未加载部分总体积超预算 -> 整专家降级为磁盘路径。
    // 绝不允许"半 pin"状态(gate 常驻而 up/down 走盘), 否则 acquire 取数路径分裂。
    size_t need = 0;
    for (const char* part : {"gate", "up", "down"}) {
        const std::string name = part_name(k.layer, k.expert, part);
        if (!pin_store_.count(name)) need += meta_of(name).nbytes;
    }
    if (cfg_.pin_budget_bytes != 0 && st_.pin_bytes + need > cfg_.pin_budget_bytes) {
        LOG_WARN("pin budget exceeded at L%dE%d (need %zu B) — 整专家降级为磁盘路径",
                 k.layer, k.expert, need);
        for (const char* part : {"gate", "up", "down"})
            pin_keys_.erase(part_name(k.layer, k.expert, part));
        return;
    }
    for (const char* part : {"gate", "up", "down"}) {
        const std::string name = part_name(k.layer, k.expert, part);
        if (pin_store_.count(name)) continue;
        const auto& m = meta_of(name);
        std::vector<uint8_t> buf(m.nbytes);
        std::error_code ec{};
        engine_->submit(file_, m.offset, buf.data(), buf.size(),
                        [&](std::error_code e) { ec = e; });
        engine_->drain();
        if (ec) throw std::runtime_error("pin load failed: " + name);
        st_.pin_bytes += m.nbytes;
        pin_store_.emplace(name, std::move(buf));
    }
}

size_t ExpertPrefetcher::pack_into_slot(Slot& s, const lwc::TensorMeta& m) {
    const uint64_t off = align_up(s.used, kSlotAlign);
    if (off + m.nbytes > cfg_.slot_bytes)
        throw std::runtime_error("slot overflow — increase slot_bytes");
    s.used = static_cast<size_t>(off + m.nbytes);
    return static_cast<size_t>(off);
}

void ExpertPrefetcher::plan_next_layer(int layer,
                                       const std::vector<int>& expert_ids) {
    if (plans_.count(layer))
        throw std::runtime_error("plan_next_layer: layer already planned");

    LayerPlan plan;
    plan.layer = layer;

    // 去重保序
    std::vector<ExpertKey> uniq;
    for (int id : expert_ids) {
        ExpertKey k{layer, id};
        bool dup = false;
        for (const auto& u : uniq) dup = dup || (u == k);
        if (!dup) uniq.push_back(k);
    }
    plan.ids = std::move(uniq);

    // 1) pin 部分: 首次触达时同步加载(常驻之后零成本)
    for (const auto& k : plan.ids)
        if (is_pinned(k)) load_pin_if_needed(k);

    // 2) 磁盘部分: 打包进一个空槽并异步提交
    bool need_slot = false;
    for (const auto& k : plan.ids)
        if (!is_pinned(k)) need_slot = true;

    if (need_slot) {
        if (free_slots_.empty())
            throw std::runtime_error(
                "no free slot — release() previous layer first (slots="
                + std::to_string(cfg_.slots) + ")");
        const size_t si = free_slots_.front();
        free_slots_.pop_front();
        Slot& s = slots_[si];
        s.used = 0;
        s.span.clear();
        s.err = std::error_code{};
        Slot* ps = &s;  // slots_ 只在 open() resize, 地址稳定

        for (const auto& k : plan.ids) {
            if (is_pinned(k)) continue;
            for (const char* part : {"gate", "up", "down"}) {
                const std::string name = part_name(k.layer, k.expert, part);
                const auto& m = meta_of(name);
                const size_t off = pack_into_slot(s, m);
                s.span.emplace(name, std::make_pair(off, m.nbytes));
                engine_->submit(file_, m.offset, s.buf->p + off, m.nbytes,
                                [ps](std::error_code e) { ps->err = e; });
                ++st_.block_reads_disk;
                st_.bytes_read_disk += m.nbytes;
            }
        }
        plan.slot = static_cast<int>(si);
    }

    plans_.emplace(layer, std::move(plan));
    ++st_.layer_plans;
}

void ExpertPrefetcher::acquire(int layer, std::vector<ExpertData>& out) {
    auto it = plans_.find(layer);
    if (it == plans_.end())
        throw std::runtime_error("acquire: layer not planned");
    LayerPlan& plan = it->second;

    engine_->drain();  // 栅栏: 残余 IO 等待(流水线设计下应已基本完成)

    if (plan.slot >= 0) {
        const Slot& s = slots_[static_cast<size_t>(plan.slot)];
        if (s.err)
            throw std::runtime_error("acquire: io failed for layer " +
                                     std::to_string(layer) + ": " + s.err.message());
    }

    out.clear();
    out.reserve(plan.ids.size());
    Slot* s = plan.slot >= 0 ? &slots_[static_cast<size_t>(plan.slot)] : nullptr;

    for (const auto& k : plan.ids) {
        ExpertData d;
        if (is_pinned(k)) {
            d.from_pin = true;
            d.gate = {pin_store_.at(part_name(k.layer, k.expert, "gate")).data(),
                      meta_of(part_name(k.layer, k.expert, "gate")).nbytes};
            d.up = {pin_store_.at(part_name(k.layer, k.expert, "up")).data(),
                    meta_of(part_name(k.layer, k.expert, "up")).nbytes};
            d.down = {pin_store_.at(part_name(k.layer, k.expert, "down")).data(),
                      meta_of(part_name(k.layer, k.expert, "down")).nbytes};
            st_.block_reads_pin += 3;
        } else {
            d.gate = {s->buf->p + s->span.at(part_name(k.layer, k.expert, "gate")).first,
                      s->span.at(part_name(k.layer, k.expert, "gate")).second};
            d.up = {s->buf->p + s->span.at(part_name(k.layer, k.expert, "up")).first,
                    s->span.at(part_name(k.layer, k.expert, "up")).second};
            d.down = {s->buf->p + s->span.at(part_name(k.layer, k.expert, "down")).first,
                      s->span.at(part_name(k.layer, k.expert, "down")).second};
        }
        out.push_back(d);
    }
    plan.acquired = true;
}

void ExpertPrefetcher::release(int layer) {
    auto it = plans_.find(layer);
    if (it == plans_.end()) return;
    LayerPlan& plan = it->second;
    if (plan.slot >= 0) {
        Slot& s = slots_[static_cast<size_t>(plan.slot)];
        s.used = 0;
        s.span.clear();
        free_slots_.push_back(static_cast<size_t>(plan.slot));
    }
    plans_.erase(it);
}

}  // namespace llmoc::wt
