// llm-on-cpu :: weights/io_engine.cpp
// ThreadIoEngine: 三系统通用的线程池异步定位读。
// Linux io_uring 后端(M1 收尾)实现同一接口后, 在 make_engine() 处替换返回值即可。

#include "weights/io_engine.h"

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "common/platform.h"

#if defined(LLMOC_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace llmoc::io {

namespace {

// ---- 平台定位读原语(集中在此文件内, 供业务侧零 #ifdef) ----

#if defined(LLMOC_OS_WINDOWS)

// 定位读语义(pread 等价物):
// Windows 上 ReadFile 只有在 FILE_FLAG_OVERLAPPED 打开的句柄上才认 OVERLAPPED.Offset;
// 普通句柄会忽略偏移并使用共享文件指针 —— 多 worker 并发读会互相串位。
// 因此: 句柄带 OVERLAPPED 标志打开, 每次读挂自己的事件做同步等待。
class NativeFile {
   public:
    ~NativeFile() { close(); }
    bool open(const Path& p) {
        h_ = ::CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING,
                           FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED,
                           nullptr);
        return h_ != INVALID_HANDLE_VALUE;
    }
    void close() {
        if (h_ != nullptr && h_ != INVALID_HANDLE_VALUE) ::CloseHandle(h_);
        h_ = nullptr;
    }
    std::error_code read_at(uint64_t off, void* dst, size_t n) {
        uint8_t* p = static_cast<uint8_t*>(dst);
        while (n > 0) {
            const DWORD chunk = static_cast<DWORD>(n > (1u << 30) ? (1u << 30) : n);
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(off & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>(off >> 32);
            ov.hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!ov.hEvent)
                return std::error_code(static_cast<int>(::GetLastError()),
                                       std::system_category());
            DWORD got = 0;
            const BOOL ok = ::ReadFile(h_, p, chunk, &got, &ov);
            if (!ok) {
                const DWORD e = ::GetLastError();
                if (e != ERROR_IO_PENDING) {
                    ::CloseHandle(ov.hEvent);
                    return std::error_code(static_cast<int>(e),
                                           std::system_category());
                }
                if (!::GetOverlappedResult(h_, &ov, &got, TRUE)) {
                    const DWORD e2 = ::GetLastError();
                    ::CloseHandle(ov.hEvent);
                    return std::error_code(static_cast<int>(e2),
                                           std::system_category());
                }
            }
            ::CloseHandle(ov.hEvent);
            if (got == 0)
                return std::error_code(1, std::generic_category());  // EOF
            p += got;
            n -= got;
            off += got;
        }
        return {};
    }

   private:
    void* h_ = nullptr;
};

#elif defined(LLMOC_OS_LINUX) || defined(LLMOC_OS_MACOS)

class NativeFile {
   public:
    ~NativeFile() { close(); }
    bool open(const Path& p) {
        fd_ = ::open(p.c_str(), O_RDONLY);
        return fd_ >= 0;
    }
    void close() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }
    std::error_code read_at(uint64_t off, void* dst, size_t n) {
        uint8_t* p = static_cast<uint8_t*>(dst);
        while (n > 0) {
            const ssize_t got = ::pread(fd_, p, n, static_cast<off_t>(off));
            if (got <= 0)
                return std::error_code(errno ? errno : EIO, std::generic_category());
            p += got;
            n -= static_cast<size_t>(got);
            off += static_cast<uint64_t>(got);
        }
        return {};
    }

   private:
    int fd_ = -1;
};

#endif

}  // namespace

namespace {

class ThreadIoEngine final : public IoEngine {
   public:
    explicit ThreadIoEngine(unsigned workers) {
        if (workers == 0)
            workers = std::max(2u, std::thread::hardware_concurrency() / 4);
        for (unsigned i = 0; i < workers; ++i)
            ws_.emplace_back([this] { worker_loop(); });
    }
    ~ThreadIoEngine() override {
        {
            std::lock_guard g(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : ws_) t.join();
    }

    void submit(const Path& file, uint64_t offset, void* dst, size_t nbytes,
                ReadCallback cb) override {
        {
            std::lock_guard g(m_);
            q_.push_back(Job{file, offset, dst, nbytes, std::move(cb)});
            ++inflight_;
        }
        cv_.notify_one();
    }

    void drain() override {
        std::unique_lock lk(m_);
        done_cv_.wait(lk, [this] { return inflight_ == 0; });
    }

    const char* backend_name() const override { return "thread-pool"; }

   private:
    struct Job {
        Path file;
        uint64_t offset;
        void* dst;
        size_t nbytes;
        ReadCallback cb;
    };

    std::shared_ptr<NativeFile> file_for(const Path& p) {
        std::lock_guard g(fcm_);
        auto& e = cache_[p.string()];
        if (!e) {
            e = std::make_shared<NativeFile>();
            if (!e->open(p)) {
                cache_.erase(p.string());
                throw std::runtime_error("cannot open: " + p.string());
            }
        }
        return e;  // shared: map 锁外安全使用
    }

    void worker_loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                job = std::move(q_.front());
                q_.pop_front();
            }
            std::error_code ec;
            try {
                auto f = file_for(job.file);
                ec = f->read_at(job.offset, job.dst, job.nbytes);
            } catch (...) {
                ec = std::error_code(EIO, std::generic_category());
            }
            if (ec)
                LOG_ERROR("io read failed: %s @+%llu bytes=%zu (%s)",
                          job.file.string().c_str(),
                          static_cast<unsigned long long>(job.offset), job.nbytes,
                          ec.message().c_str());

            bool idle = false;
            {
                std::lock_guard g(m_);
                --inflight_;
                idle = inflight_ == 0;
            }
            if (job.cb) job.cb(ec);
            if (idle) done_cv_.notify_all();
        }
    }

    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> q_;
    int inflight_ = 0;
    bool stop_ = false;
    std::vector<std::thread> ws_;

    // 文件句柄缓存: mutex 保护 + shared_ptr 出锁使用。
    // (Windows 定位读靠 OVERLAPPED 句柄, POSIX 用 pread, 两者天然并发安全)
    std::mutex fcm_;
    std::unordered_map<std::string, std::shared_ptr<NativeFile>> cache_;
};

}  // namespace

std::unique_ptr<IoEngine> make_engine(unsigned workers) {
    const char* choice = std::getenv("LLMOC_IO_ENGINE");

#if defined(LLMOC_OS_LINUX) && defined(LLMOC_HAVE_LIBURING)
    if (!choice || std::strcmp(choice, "uring") == 0) {
        try {
            auto e = make_uring_engine_if_available();
            LOG_INFO("io engine: %s (%s)", e->backend_name(), sys::os_name());
            return e;
        } catch (const std::exception& ex) {
            LOG_WARN("uring engine init failed (%s) — falling back to thread-pool",
                     ex.what());
        }
    }
#else
    if (choice && std::strcmp(choice, "uring") == 0)
        LOG_WARN("LLMOC_IO_ENGINE=uring 但本构建未编入 io_uring 后端 — 使用 thread-pool");
#endif
    (void)choice;
    LOG_INFO("io engine: thread-pool (%s)", sys::os_name());
    return std::make_unique<ThreadIoEngine>(workers);
}

}  // namespace llmoc::io
