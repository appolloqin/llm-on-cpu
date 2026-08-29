#pragma once
// llm-on-cpu :: common/log.h
// Header-only structured logger. Dev-machine friendly (no deps).
// Level via env LLMOC_LOG=debug|info|warn|error (default info)
//
// 格式化约定: LOG_*(fmt, args...) 直接走 vfprintf 语义,
// 字符串参数必须传 c_str()/字面量(内核热路径禁止构造 std::string, 天然满足)。

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace llmoc::log {

enum class Level : int { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

inline std::atomic<Level>& threshold() {
    static std::atomic<Level> t{[] {
        const char* e = std::getenv("LLMOC_LOG");
        if (!e) return Level::kInfo;
        if (std::strcmp(e, "debug") == 0) return Level::kDebug;
        if (std::strcmp(e, "warn") == 0) return Level::kWarn;
        if (std::strcmp(e, "error") == 0) return Level::kError;
        return Level::kInfo;
    }()};
    return t;
}

inline const char* tag(Level lv) {
    switch (lv) {
        case Level::kDebug: return "D";
        case Level::kInfo:  return "I";
        case Level::kWarn:  return "W";
        default:            return "E";
    }
}

inline bool enabled(Level lv) { return lv >= threshold().load(std::memory_order_relaxed); }

inline void prefix(Level lv) {
    using namespace std::chrono;
    const auto now = system_clock::now().time_since_epoch();
    const auto ms = duration_cast<milliseconds>(now).count() % 1000;
    const auto sec = duration_cast<seconds>(now).count();
    std::fprintf(stderr, "%s [%lld.%03lld] ", tag(lv),
                 static_cast<long long>(sec), static_cast<long long>(ms));
}

template <typename... Args>
void emit(Level lv, const char* fmt, Args&&... args) {
    if (!enabled(lv)) return;
#if defined(_MSC_VER)
    _lock_file(stderr);
#else
    std::flockfile(stderr);
#endif
    prefix(lv);
    if constexpr (sizeof...(Args) == 0) {
        std::fwrite(fmt, 1, std::strlen(fmt), stderr);
    } else {
        std::fprintf(stderr, fmt, std::forward<Args>(args)...);
    }
    std::fputc('\n', stderr);
#if defined(_MSC_VER)
    _unlock_file(stderr);
#else
    std::funlockfile(stderr);
#endif
}

}  // namespace llmoc::log

#define LOG_DEBUG(...) ::llmoc::log::emit(::llmoc::log::Level::kDebug, __VA_ARGS__)
#define LOG_INFO(...)  ::llmoc::log::emit(::llmoc::log::Level::kInfo,  __VA_ARGS__)
#define LOG_WARN(...)  ::llmoc::log::emit(::llmoc::log::Level::kWarn,  __VA_ARGS__)
#define LOG_ERROR(...) ::llmoc::log::emit(::llmoc::log::Level::kError, __VA_ARGS__)
