// llm-on-cpu :: common/omp_tune.cpp
#include "common/omp_tune.h"

#include <cstdlib>

#include "common/log.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef _WIN32
#include <stdlib.h>
#endif

namespace llmoc {
namespace {

void set_omp_env_if_unset(const char* key, const char* val) {
  if (std::getenv(key) != nullptr) return;
#ifdef _WIN32
  _putenv_s(key, val);
#else
  setenv(key, val, 0);
#endif
}

// INT4 decode is DRAM-bandwidth bound. On HX (8P+8E, 24T) we measured:
//   OMP=8/16 → ~10 tok/s   OMP=12/24 → ~5 tok/s  (2026-08-31, i7-14650HX)
int pick_thread_budget(int logical) {
  if (logical <= 8) return logical;
  if (logical <= 16) return 8;
  return 8;
}

int pick_thread_budget_from_user(int user_threads, int hw_logical) {
  (void)hw_logical;
  if (user_threads <= 16) return user_threads;
  return 8;
}

}  // namespace

int tune_openmp_for_decode() {
#if defined(_OPENMP)
  // Pin threads before the first parallel region (ignored if unsupported).
  set_omp_env_if_unset("OMP_PROC_BIND", "close");
  set_omp_env_if_unset("OMP_PLACES", "cores");

  if (std::getenv("LLMOC_OMP_FULL") != nullptr) {
    const int n = omp_get_max_threads();
    LOG_INFO("OpenMP max_threads=%d (LLMOC_OMP_FULL set)", n);
    return n;
  }

  if (std::getenv("OMP_NUM_THREADS") != nullptr) {
    const int user = omp_get_max_threads();
    const int hw = omp_get_num_procs();
    const int prefer = pick_thread_budget_from_user(user, hw);
    if (prefer < user) {
      omp_set_num_threads(prefer);
      LOG_WARN(
          "OpenMP capped OMP_NUM_THREADS %d → %d for bandwidth-bound INT4 decode "
          "(use 8 on HX; unset or LLMOC_OMP_FULL=1 to keep %d)",
          user, prefer, user);
    } else {
      LOG_INFO("OpenMP max_threads=%d (OMP_NUM_THREADS set by user)", user);
    }
    return omp_get_max_threads();
  }
  const int logical = omp_get_max_threads();
  const int prefer = pick_thread_budget(logical);
  if (prefer < logical) {
    omp_set_num_threads(prefer);
    LOG_WARN(
        "OpenMP capped %d → %d for bandwidth-bound INT4 decode (override with OMP_NUM_THREADS; "
        "8 or 16 on HX laptops)",
        logical, prefer);
  } else {
    LOG_INFO("OpenMP max_threads=%d", prefer);
  }
  return omp_get_max_threads();
#else
  LOG_INFO("OpenMP: not compiled in");
  return 1;
#endif
}

}  // namespace llmoc
