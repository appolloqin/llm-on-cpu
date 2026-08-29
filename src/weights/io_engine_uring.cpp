// llm-on-cpu :: weights/io_engine_uring.cpp
// Linux 专属: io_uring + O_DIRECT 异步定位读后端(M1 收尾项)。
// 编译条件: LLMOC_OS_LINUX && LLMOC_HAVE_LIBURING (CMake 探测, 见根 CMakeLists)。
//
// 对齐约定(LWC 格式天然满足 offset 4K 对齐):
//   * O_DIRECT 要求 offset/size/内存三重 4K 对齐
//   * size 非对齐(bf16 张量字节数任意)时走内部 bounce 对齐读 + memcpy —— 正确性优先,
//     LWC v2 计划把张量块 pad 到 4K 倍数后全量零拷贝(TODO)
// 选择: 环境变量 LLMOC_IO_ENGINE = uring | thread (默认 uring, 缺库自动回退 thread)

#if defined(LLMOC_OS_LINUX) && defined(LLMOC_HAVE_LIBURING)

#include <liburing.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/alloc.h"
#include "common/log.h"
#include "common/platform.h"
#include "weights/io_engine.h"

namespace llmoc::io {

namespace {

constexpr uint32_t kRingDepth = 64;
constexpr size_t kAlign = 4096;

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
uint64_t align_dn64(uint64_t v, uint64_t a) { return v / a * a; }

void* alloc_4k(size_t n) { return mem::alloc_aligned(n, kAlign); }

class DirectFile {
   public:
    ~DirectFile() {
        if (fd_ >= 0) ::close(fd_);
    }
    bool open(const Path& p) {
        fd_ = ::open(p.c_str(), O_RDONLY | O_DIRECT);
        return fd_ >= 0;
    }
    int fd() const { return fd_; }

   private:
    int fd_ = -1;
};

struct Job {
    uint64_t offset = 0;
    void* dst = nullptr;
    size_t nbytes = 0;
    ReadCallback cb;
    std::shared_ptr<DirectFile> file;
    // O_DIRECT 非对齐回退
    std::unique_ptr<void, void (*)(void*)> bounce{nullptr, [](void* p) {
                                                      if (p) mem::free_aligned(p);
                                                  }};
    size_t bounce_len = 0;
    size_t copy_off = 0;  // bounce 内有效数据起点
};

class UringIoEngine final : public IoEngine {
   public:
    explicit UringIoEngine(unsigned /*workers*/) {
        if (io_uring_queue_init(kRingDepth, &ring_, 0) != 0)
            throw std::runtime_error("io_uring_queue_init failed");
        worker_ = std::thread([this] { loop(); });
    }
    ~UringIoEngine() override {
        {
            std::lock_guard g(m_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        io_uring_queue_exit(&ring_);
    }

    void submit(const Path& file, uint64_t offset, void* dst, size_t nbytes,
                ReadCallback cb) override {
        auto job = std::make_shared<Job>();
        job->offset = offset;
        job->dst = dst;
        job->nbytes = nbytes;
        job->cb = std::move(cb);
        job->file = file_for(file);

        const bool aligned =
            (offset % kAlign == 0) && (nbytes % kAlign == 0) &&
            (reinterpret_cast<uintptr_t>(dst) % kAlign == 0);
        if (!aligned) {
            const uint64_t start = align_dn64(offset, kAlign);
            job->copy_off = static_cast<size_t>(offset - start);
            job->bounce_len = align_up(offset - start + nbytes, kAlign);
            job->bounce.reset(alloc_4k(job->bounce_len));
            job->offset = start;
        }

        {
            std::lock_guard g(m_);
            q_.push_back(std::move(job));
        }
        cv_.notify_one();
    }

    void drain() override {
        std::unique_lock lk(m_);
        done_cv_.wait(lk, [this] { return inflight_ == 0 && q_.empty(); });
    }

    const char* backend_name() const override { return "io_uring+O_DIRECT"; }

   private:
    std::shared_ptr<DirectFile> file_for(const Path& p) {
        std::lock_guard g(fcm_);
        auto& e = cache_[p.string()];
        if (!e) {
            e = std::make_shared<DirectFile>();
            if (!e->open(p)) {
                cache_.erase(p.string());
                throw std::runtime_error("cannot open (O_DIRECT): " + p.string());
            }
        }
        return e;
    }

    bool try_submit_one() {
        std::shared_ptr<Job> job;
        {
            std::lock_guard g(m_);
            if (q_.empty()) return false;
            job = std::move(q_.front());
            q_.pop_front();
            ++inflight_;
        }
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {  // 满了, 放回
            std::lock_guard g(m_);
            q_.push_front(std::move(job));
            --inflight_;
            return false;
        }
        void* buf = job->bounce ? job->bounce.get() : job->dst;
        size_t len = job->bounce ? job->bounce_len : job->nbytes;
        io_uring_prep_read(sqe, job->file->fd(), buf, len, job->offset);
        io_uring_sqe_set_data(sqe, new std::shared_ptr<Job>(std::move(job)));
        return true;
    }

    void reap_one(bool wait) {
        io_uring_cqe* cqe = nullptr;
        if (wait)
            io_uring_wait_cqe(&ring_, &cqe);
        else if (io_uring_peek_cqe(&ring_, &cqe) != 0 || !cqe)
            return;
        if (!cqe) return;

        auto* holder = static_cast<std::shared_ptr<Job>*>(io_uring_cqe_get_data(cqe));
        std::shared_ptr<Job> job = *holder;
        delete holder;
        const int res = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);

        std::error_code ec;
        if (res < 0) {
            ec = std::error_code(-res, std::generic_category());
        } else {
            const size_t got = static_cast<size_t>(res);
            if (job->bounce) {
                if (got < job->copy_off + job->nbytes)
                    ec = std::error_code(EIO, std::generic_category());
                else
                    std::memcpy(job->dst,
                                static_cast<uint8_t*>(job->bounce.get()) +
                                    job->copy_off,
                                job->nbytes);
            } else if (got != job->nbytes) {
                ec = std::error_code(EIO, std::generic_category());  // 短读
            }
        }
        if (ec)
            LOG_ERROR("uring read failed: fd=%d @+%llu bytes=%zu (%s)",
                      job->file->fd(), static_cast<unsigned long long>(job->offset),
                      job->nbytes, ec.message().c_str());

        bool idle = false;
        {
            std::lock_guard g(m_);
            --inflight_;
            idle = inflight_ == 0 && q_.empty();
        }
        if (job->cb) job->cb(ec);
        if (idle) done_cv_.notify_all();
    }

    void loop() {
        for (;;) {
            {
                std::lock_guard g(m_);
                if (stop_ && q_.empty() && inflight_ == 0) return;
            }
            bool did = false;
            while (try_submit_one()) did = true;
            if (did || inflight_ > 0) io_uring_submit(&ring_);
            reap_one(!did);
        }
    }

    io_uring ring_{};
    std::thread worker_;
    std::mutex m_, fcm_;
    std::condition_variable cv_, done_cv_;
    std::deque<std::shared_ptr<Job>> q_;
    int inflight_ = 0;
    bool stop_ = false;
    std::unordered_map<std::string, std::shared_ptr<DirectFile>> cache_;
};

}  // namespace

std::unique_ptr<IoEngine> make_uring_engine_if_available() {
    return std::make_unique<UringIoEngine>(0);
}

}  // namespace llmoc::io

#endif  // LLMOC_OS_LINUX && LLMOC_HAVE_LIBURING
