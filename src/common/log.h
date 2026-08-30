#pragma once
// llm-on-cpu :: common/log.h
// 控制台 + 按日滚动文件日志。
//   LLMOC_LOG=debug|info|warn|error   (默认 info)
//   LLMOC_LOG_DIR=logs                (默认 logs；设为空则只打 stderr)
//   LLMOC_PROFILE=1                   (生成/层耗时明细)
//
// 文件名: {LLMOC_LOG_DIR}/llmoc-YYYY-MM-DD.log

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace llmoc::log {

enum class Level : int { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

void init(const char* dir = nullptr);
std::atomic<Level>& threshold();
bool profile_enabled();
void write_line(Level lv, const char* msg);
// printf 风格（实现在 log.cpp，避免头文件里 stack snprintf / FILE 锁问题）
void emit_fmt(Level lv, const char* fmt, ...);

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

inline void emit(Level lv, const char* msg) {
  if (!enabled(lv)) return;
  write_line(lv, msg);
}

struct ScopeTimer {
  const char* name;
  std::chrono::steady_clock::time_point t0;
  explicit ScopeTimer(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
  ~ScopeTimer() {
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (enabled(Level::kInfo)) emit_fmt(Level::kInfo, "[timer] %s %.1f ms", name, ms);
  }
  ScopeTimer(const ScopeTimer&) = delete;
  ScopeTimer& operator=(const ScopeTimer&) = delete;
};

}  // namespace llmoc::log

// 单参数走 write_line；多参数走 emit_fmt（C 变参，MSVC 安全）
#define LOG_DEBUG(...) ::llmoc::log::emit_fmt(::llmoc::log::Level::kDebug, __VA_ARGS__)
#define LOG_INFO(...) ::llmoc::log::emit_fmt(::llmoc::log::Level::kInfo, __VA_ARGS__)
#define LOG_WARN(...) ::llmoc::log::emit_fmt(::llmoc::log::Level::kWarn, __VA_ARGS__)
#define LOG_ERROR(...) ::llmoc::log::emit_fmt(::llmoc::log::Level::kError, __VA_ARGS__)
#define LOG_TIMER(name) ::llmoc::log::ScopeTimer _llmoc_timer_##__LINE__(name)
