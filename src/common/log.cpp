// llm-on-cpu :: common/log.cpp
#include "common/log.h"

#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>

namespace llmoc::log {
namespace {

// Intentionally never destroyed: workers / atexit must not race CRT teardown.
std::mutex& log_mu() {
  static std::mutex* m = new std::mutex;
  return *m;
}

FILE*& log_fp() {
  static FILE* fp = nullptr;
  return fp;
}

std::string& log_dir() {
  static std::string* d = new std::string;  // empty = stderr only until init()
  return *d;
}

int& log_day() {
  static int day = -1;
  return day;
}

bool& file_logging() {
  static bool on = false;  // only true after init()
  return on;
}

int today_yyyymmdd() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  return (tm.tm_year + 1900) * 10000 + (tm.tm_mon + 1) * 100 + tm.tm_mday;
}

void format_stamp(char* out, size_t n) {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const auto tt = system_clock::to_time_t(now);
  const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  std::snprintf(out, n, "%04d-%02d-%02d %02d:%02d:%02d.%03d", tm.tm_year + 1900, tm.tm_mon + 1,
                tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms));
}

void open_today_locked() {
  if (!file_logging() || log_dir().empty()) {
    if (log_fp()) {
      std::fclose(log_fp());
      log_fp() = nullptr;
    }
    log_day() = today_yyyymmdd();
    return;
  }
  const int day = today_yyyymmdd();
  if (log_fp() && day == log_day()) return;
  if (log_fp()) {
    std::fclose(log_fp());
    log_fp() = nullptr;
  }
  std::error_code ec;
  std::filesystem::create_directories(log_dir(), ec);
  const int y = day / 10000;
  const int mo = (day / 100) % 100;
  const int d = day % 100;
  char name[64];
  std::snprintf(name, sizeof(name), "llmoc-%04d-%02d-%02d.log", y, mo, d);
  const std::string path = (std::filesystem::path(log_dir()) / name).string();
#if defined(_WIN32)
  FILE* fp = nullptr;
  if (fopen_s(&fp, path.c_str(), "a") != 0) fp = nullptr;
  log_fp() = fp;
#else
  log_fp() = std::fopen(path.c_str(), "a");
#endif
  log_day() = day;
  if (log_fp()) std::setvbuf(log_fp(), nullptr, _IOLBF, 0);
}

}  // namespace

std::atomic<Level>& threshold() {
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

bool profile_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("LLMOC_PROFILE");
    return e && e[0] == '1';
  }();
  return on;
}

void init(const char* dir) {
  std::lock_guard<std::mutex> lock(log_mu());
  if (dir) {
    log_dir() = dir;
  } else if (const char* e = std::getenv("LLMOC_LOG_DIR")) {
    log_dir() = e;
  } else {
    log_dir() = "logs";
  }
  file_logging() = !log_dir().empty();
  open_today_locked();
  char stamp[64];
  format_stamp(stamp, sizeof(stamp));
  char line[512];
  if (!file_logging()) {
    std::snprintf(line, sizeof(line), "%s I [log] file logging disabled", stamp);
  } else {
    std::snprintf(line, sizeof(line), "%s I [log] dir=%s file=llmoc-YYYY-MM-DD.log", stamp,
                  log_dir().c_str());
  }
  std::fputs(line, stderr);
  std::fputc('\n', stderr);
  if (log_fp()) {
    std::fputs(line, log_fp());
    std::fputc('\n', log_fp());
    std::fflush(log_fp());
  }
}

void write_line(Level lv, const char* msg) {
  char stamp[64];
  format_stamp(stamp, sizeof(stamp));
  std::string line;
  line.reserve(128 + (msg ? std::strlen(msg) : 0));
  line.append(stamp);
  line.push_back(' ');
  line.append(tag(lv));
  line.push_back(' ');
  line.append(msg ? msg : "");
  line.push_back('\n');

  std::lock_guard<std::mutex> lock(log_mu());
  if (file_logging()) open_today_locked();

  std::fwrite(line.data(), 1, line.size(), stderr);
  if (log_fp()) {
    std::fwrite(line.data(), 1, line.size(), log_fp());
    if (lv >= Level::kWarn) std::fflush(log_fp());
  }
}

void emit_fmt(Level lv, const char* fmt, ...) {
  if (!enabled(lv) || !fmt) return;
  // Heap buffer: avoid large stack frame + /GS issues under deep call stacks.
  std::string buf(2048, '\0');
  va_list ap;
  va_start(ap, fmt);
  const int n = std::vsnprintf(buf.data(), buf.size(), fmt, ap);
  va_end(ap);
  if (n < 0) {
    write_line(lv, "(log format error)");
    return;
  }
  if (static_cast<size_t>(n) >= buf.size()) {
    buf.resize(static_cast<size_t>(n) + 1);
    va_list ap2;
    va_start(ap2, fmt);
    std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
  }
  buf.resize(static_cast<size_t>(n));
  write_line(lv, buf.c_str());
}

}  // namespace llmoc::log
