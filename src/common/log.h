#pragma once
// llm-on-cpu :: common/log.h
// 控制台 + 按日滚动文件日志。
//   LLMOC_LOG=debug|info|warn|error   (默认 info)
//   LLMOC_LOG_DIR=logs                (默认 logs；空字符串则只打 stderr)
//   LLMOC_PROFILE=1                   (生成/层耗时明细)
//
// 文件名: {LLMOC_LOG_DIR}/llmoc-YYYY-MM-DD.log

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>

namespace llmoc::log {

enum class Level : int { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

// 进程启动时调用一次（可重复，幂等）。dir=nullptr 用环境变量 / 默认 "logs"。
void init(const char* dir = nullptr);

std::atomic<Level>& threshold();
bool profile_enabled();

inline const char* tag(Level lv) {
  switch (lv) {
    case Level::kDebug: return "D";
    case Level::kInfo: return "I";
    case Level::kWarn: return "W";
    default: return "E";
  }
}

inline bool enabled(Level lv) {
  return lv >= threshold().load(std::memory_order_relaxed);
}

// 内部实现（log.cpp）：带本地时间戳，写 stderr + 当日日志文件
void write_line(Level lv, const char* msg);

inline void vemit(Level lv, const char* fmt, ...) {
  if (!enabled(lv)) return;
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  write_line(lv, buf);
}

template <typename... Args>
void emit(Level lv, const char* fmt, Args&&... args) {
  if (!enabled(lv)) return;
  if constexpr (sizeof...(Args) == 0) {
    write_line(lv, fmt);
  } else {
    char buf[4096];
    std::snprintf(buf, sizeof(buf), fmt, std::forward<Args>(args)...);
    write_line(lv, buf);
  }
}

/** RAII 计时：析构时打一条 INFO（或 PROFILE 级别走同一通道） */
struct ScopeTimer {
  const char* name;
  std::chrono::steady_clock::time_point t0;
  explicit ScopeTimer(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
  ~ScopeTimer() {
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    emit(Level::kInfo, "[timer] %s %.1f ms", name, ms);
  }
  ScopeTimer(const ScopeTimer&) = delete;
  ScopeTimer& operator=(const ScopeTimer&) = delete;
};

}  // namespace llmoc::log

#define LOG_DEBUG(...) ::llmoc::log::emit(::llmoc::log::Level::kDebug, __VA_ARGS__)
#define LOG_INFO(...) ::llmoc::log::emit(::llmoc::log::Level::kInfo, __VA_ARGS__)
#define LOG_WARN(...) ::llmoc::log::emit(::llmoc::log::Level::kWarn, __VA_ARGS__)
#define LOG_ERROR(...) ::llmoc::log::emit(::llmoc::log::Level::kError, __VA_ARGS__)
#define LOG_TIMER(name) ::llmoc::log::ScopeTimer _llmoc_timer_##__LINE__(name)
