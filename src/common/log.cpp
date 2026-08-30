// llm-on-cpu :: common/log.cpp
#include "common/log.h"

#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace llmoc::log {
namespace {

std::mutex g_mu;
FILE* g_fp = nullptr;
std::string g_dir = "logs";
int g_day = -1;  // YYYYMMDD
bool g_inited = false;

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
  if (g_dir.empty()) {
    if (g_fp) {
      std::fclose(g_fp);
      g_fp = nullptr;
    }
    g_day = today_yyyymmdd();
    return;
  }
  const int day = today_yyyymmdd();
  if (g_fp && day == g_day) return;
  if (g_fp) {
    std::fclose(g_fp);
    g_fp = nullptr;
  }
  std::error_code ec;
  std::filesystem::create_directories(g_dir, ec);
  char path[512];
  std::snprintf(path, sizeof(path), "%s/llmoc-%04d-%02d-%02d.log", g_dir.c_str(), day / 10000,
                (day / 100) % 100, day % 100);
#if defined(_WIN32)
  fopen_s(&g_fp, path, "a");
#else
  g_fp = std::fopen(path, "a");
#endif
  g_day = day;
  if (g_fp) {
    std::setvbuf(g_fp, nullptr, _IOLBF, 0);  // line buffered
  }
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
  std::lock_guard<std::mutex> lock(g_mu);
  if (dir) {
    g_dir = dir;
  } else if (const char* e = std::getenv("LLMOC_LOG_DIR")) {
    g_dir = e;
  } else {
    g_dir = "logs";
  }
  open_today_locked();
  g_inited = true;
  char stamp[64];
  format_stamp(stamp, sizeof(stamp));
  char line[256];
  if (g_dir.empty()) {
    std::snprintf(line, sizeof(line), "%s I [log] file logging disabled (LLMOC_LOG_DIR empty)",
                  stamp);
  } else {
    std::snprintf(line, sizeof(line), "%s I [log] writing to %s/llmoc-YYYY-MM-DD.log", stamp,
                  g_dir.c_str());
  }
  std::fputs(line, stderr);
  std::fputc('\n', stderr);
  if (g_fp) {
    std::fputs(line, g_fp);
    std::fputc('\n', g_fp);
    std::fflush(g_fp);
  }
}

void write_line(Level lv, const char* msg) {
  char stamp[64];
  format_stamp(stamp, sizeof(stamp));
  char line[4200];
  std::snprintf(line, sizeof(line), "%s %s %s", stamp, tag(lv), msg ? msg : "");

  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_inited) {
    // lazy init so early LOG_* still works
    if (const char* e = std::getenv("LLMOC_LOG_DIR")) g_dir = e;
    open_today_locked();
    g_inited = true;
  } else {
    open_today_locked();
  }

#if defined(_MSC_VER)
  _lock_file(stderr);
#else
  ::flockfile(stderr);
#endif
  std::fputs(line, stderr);
  std::fputc('\n', stderr);
#if defined(_MSC_VER)
  _unlock_file(stderr);
#else
  ::funlockfile(stderr);
#endif

  if (g_fp) {
    std::fputs(line, g_fp);
    std::fputc('\n', g_fp);
    if (lv >= Level::kWarn) std::fflush(g_fp);
  }
}

}  // namespace llmoc::log
