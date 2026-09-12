// llm-on-cpu :: model/gen_stop.h
// Decode early-stop heuristics: whitespace runs + token n-gram cycles (greedy loops).
#pragma once

#include <cstdint>
#include <vector>

namespace llmoc::model {

// Returns cycle period (>0) if the completion suffix is a repeated token n-gram, else -1.
// Long cycles (period 16..128): need >=3 consecutive repeats (typical paragraph loops).
// Short cycles (period 2..15): need >=8 consecutive repeats (e.g. "哈哈" / "---\n").
inline int find_token_cycle_period(const std::vector<int32_t>& ids) {
  const int n = static_cast<int>(ids.size());
  auto match = [&](int period, int cycles) -> bool {
    if (period <= 0 || cycles < 2 || n < period * cycles) return false;
    for (int c = 1; c < cycles; ++c) {
      for (int i = 0; i < period; ++i) {
        if (ids[static_cast<size_t>(n - 1 - i)] !=
            ids[static_cast<size_t>(n - 1 - c * period - i)])
          return false;
      }
    }
    return true;
  };
  for (int p = 16; p <= 128; ++p)
    if (match(p, 3)) return p;
  for (int p = 2; p <= 15; ++p)
    if (match(p, 8)) return p;
  return -1;
}

// Drop trailing exact repeats of `period`, leaving a single copy.
inline void trim_trailing_token_cycles(std::vector<int32_t>& ids, int period) {
  if (period <= 0) return;
  while (static_cast<int>(ids.size()) >= 2 * period) {
    const int n = static_cast<int>(ids.size());
    bool eq = true;
    for (int i = 0; i < period; ++i) {
      if (ids[static_cast<size_t>(n - 1 - i)] !=
          ids[static_cast<size_t>(n - 1 - period - i)]) {
        eq = false;
        break;
      }
    }
    if (!eq) break;
    ids.resize(static_cast<size_t>(n - period));
  }
}

}  // namespace llmoc::model
