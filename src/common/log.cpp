// llm-on-cpu :: common/log.cpp
#include "common/log.h"

#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace llmoc::log {
namespace {

// Leaked on purpose: avoid CRT teardown races with worker threads.
struct State {
  std::mutex mu;
  FILE* fp = nullptr;
  std::string dir;       // empty => stderr only
  int day = -1;          // YYYYMMDD
  bool allow_file = false;
};

State& state() {
  static State* s = new State;
  return *s;
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

void ensure_dir(const std::string& dir) {
  if (dir.empty()) return;
#if defined(_WIN32)
  // nested mkdir for "a/b/c"
  std::string cur;
  for (size_t i = 0; i <= dir.size(); ++i) {
    if (i == dir.size() || dir[i] == '/' || dir[i] == '\\') {
      if (!cur.empty()) _mkdir(cur.c_str());
      if (i < dir.size()) cur.push_back('\\');
    } else {
      cur.push_back(dir[i]);
    }
  }
#else
  std::string cur;
  for (size_t i = 0; i <= dir.size(); ++i) {
    if (i == dir.size() || dir[i] == '/') {
      if (!cur.empty()) mkdir(cur.c_str(), 0755);
      if (i < dir.size()) cur.push_back('/');
    } else {
      cur.push_back(dir[i]);
    }
  }
#endif
}

void open_today_locked(State& st) {
  if (!st.allow_file || st.dir.empty()) {
    if (st.fp) {
      std::fclose(st.fp);
      st.fp = nullptr;
    }
    st.day = today_yyyymmdd();
    return;
  }
  const int day = today_yyyymmdd();
  if (st.fp && day == st.day) return;
  if (st.fp) {
    std::fclose(st.fp);
    st.fp = nullptr;
  }
  ensure_dir(st.dir);
  const int y = day / 10000;
  const int mo = (day / 100) % 100;
  const int d = day % 100;
  char path[512];
  std::snprintf(path, sizeof(path), "%s/llmoc-%04d-%02d-%02d.log", st.dir.c_str(), y, mo, d);
#if defined(_WIN32)
  FILE* fp = nullptr;
  if (fopen_s(&fp, path, "a") != 0) fp = nullptr;
  st.fp = fp;
#else
  st.fp = std::fopen(path, "a");
#endif
  st.day = day;
}

void emit_raw(State& st, const char* line, size_t len, Level lv) {
  if (len == 0) return;
  (void)lv;
  std::fwrite(line, 1, len, stderr);
  std::fflush(stderr);
  if (st.fp) {
    std::fwrite(line, 1, len, st.fp);
    std::fflush(st.fp);
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
  State& st = state();
  std::lock_guard<std::mutex> lock(st.mu);
  if (dir) {
    st.dir = dir;
  } else if (const char* e = std::getenv("LLMOC_LOG_DIR")) {
    st.dir = e;
  } else {
    st.dir = "logs";
  }
  st.allow_file = !st.dir.empty();
  open_today_locked(st);

  char stamp[64];
  format_stamp(stamp, sizeof(stamp));
  char line[640];
  int n;
  if (!st.allow_file) {
    n = std::snprintf(line, sizeof(line), "%s I [log] file logging disabled\n", stamp);
  } else {
    n = std::snprintf(line, sizeof(line), "%s I [log] dir=%s file=llmoc-YYYY-MM-DD.log\n", stamp,
                      st.dir.c_str());
  }
  if (n > 0) emit_raw(st, line, static_cast<size_t>(n), Level::kInfo);
}

void write_line(Level lv, const char* msg) {
  char stamp[64];
  format_stamp(stamp, sizeof(stamp));

  const char* body = msg ? msg : "";
  const size_t body_len = std::strlen(body);
  std::vector<char> line;
  line.resize(64 + 4 + body_len + 2);
  const int n =
      std::snprintf(line.data(), line.size(), "%s %s %s\n", stamp, tag(lv), body);
  if (n <= 0) return;

  State& st = state();
  std::lock_guard<std::mutex> lock(st.mu);
  if (st.allow_file) open_today_locked(st);
  emit_raw(st, line.data(), static_cast<size_t>(n), lv);
}

void emit_fmt(Level lv, const char* fmt, ...) {
  if (!enabled(lv) || !fmt) return;

  std::vector<char> buf(1024);
  va_list ap;
  va_start(ap, fmt);
  int n = std::vsnprintf(buf.data(), buf.size(), fmt, ap);
  va_end(ap);
  if (n < 0) {
    write_line(lv, "(log format error)");
    return;
  }
  if (static_cast<size_t>(n) >= buf.size()) {
    buf.resize(static_cast<size_t>(n) + 1);
    va_list ap2;
    va_start(ap2, fmt);
    n = std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
    if (n < 0) {
      write_line(lv, "(log format error)");
      return;
    }
  }
  write_line(lv, buf.data());
}

}  // namespace llmoc::log
