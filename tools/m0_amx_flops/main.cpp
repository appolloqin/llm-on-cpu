// llm-on-cpu :: tools/m0_amx_flops/main.cpp
// M0 基线: ISA 探测 + 标量/AVX2-FMA 单线程可达 flops(依赖链下限口径)。
// AMX tile 微内核随 M5/Linux SPR CI 交付(docs/PLATFORM.md 矩阵)。

#include <chrono>
#include <cstdio>
#include <cstring>

#ifdef LLMOC_ENABLE_AVX2
#  include <immintrin.h>
#  if defined(_MSC_VER)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>  // IsProcessorFeaturePresent / PF_AVX2_*
#  endif
#endif

#include "common/alloc.h"
#include "common/cpuid.h"
#include "common/platform.h"
#include "hal/amx_gemm.h"

using llmoc::mem::alloc_aligned;
using llmoc::mem::free_aligned;
using Clock = std::chrono::steady_clock;

namespace {

constexpr size_t kFloats = 16u << 16;   // 64K floats = 256KB 流式工作集
constexpr long long kIters = 20000;

void print_flags(const llmoc::cpu::IsaFlags& f) {
    auto y = [](bool b) { return b ? "true" : "false"; };
    std::printf("  sse42       : %s\n", y(f.sse42));
    std::printf("  avx2        : %s\n", y(f.avx2));
    std::printf("  avx512f     : %s\n", y(f.avx512f));
    std::printf("  avx512_bf16 : %s\n", y(f.avx512_bf16));
    std::printf("  amx_tile    : %s\n", y(f.amx_tile));
    std::printf("  amx_int8    : %s\n", y(f.amx_int8));
    std::printf("  amx_bf16    : %s\n", y(f.amx_bf16));
    std::printf("  os_amx_ready: %s   (需要 XCR0[AMX]=1, Linux 动态启用)\n",
               y(f.os_supports_amx));
}

#ifdef LLMOC_ENABLE_AVX2
bool cpu_has_avx2_runtime() {
#  if defined(_MSC_VER)
    return IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0;
#  elif defined(__x86_64__)
    return true;  // 宏侧已由编译开关保证
#  else
    return false;
#  endif
}
#endif

}  // namespace

int main() {
    const auto flags = llmoc::cpu::detect_isa();

    std::printf("== LLM-on-CPU M0 host profile (%s) ==\n", llmoc::sys::os_name());
    std::printf("cpu brand : %s\n", llmoc::cpu::brand_string().c_str());
    print_flags(flags);

    float* x = static_cast<float*>(alloc_aligned(kFloats * sizeof(float)));
    float* y = static_cast<float*>(alloc_aligned(kFloats * sizeof(float)));
    std::memset(x, 0, kFloats * sizeof(float));
    for (size_t i = 0; i < kFloats; ++i)
        y[i] = 1e-4f + static_cast<float>(i % 7) * 1e-3f;
    (void)x;

    // ---- 标量(单条依赖链 => 吞吐下限口径) ----
    const float c = 0.9999997f;
    auto t0 = Clock::now();
    float acc = 1.f;
    for (long long it = 0; it < kIters; ++it)
        for (size_t j = 0; j < kFloats; ++j) acc = acc * c + y[j];
    const double s_dt = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("scalar fp32 dep-chain      : %10.2f GFLOPS   "
                "[dt=%.3fs, sink=%g]\n",
                static_cast<double>(kIters) * static_cast<double>(kFloats) * 2.0 /
                    1e9 / s_dt,
                s_dt, static_cast<double>(acc));

#ifdef LLMOC_ENABLE_AVX2
    if (cpu_has_avx2_runtime()) {
        const __m256 vc = _mm256_set1_ps(c);
        auto t2 = Clock::now();
        __m256 accv = _mm256_setzero_ps();
        for (long long it = 0; it < kIters; ++it)
            for (size_t j = 0; j < kFloats; j += 8) {
                const __m256 vy = _mm256_load_ps(y + j);
                accv = _mm256_fmadd_ps(accv, vc, vy);
            }
        const double a_dt = std::chrono::duration<double>(Clock::now() - t2).count();
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, accv);
        std::printf("avx2+fma fp32 dep-chain    : %10.2f GFLOPS   "
                    "[dt=%.3fs, sink=%g]\n",
                    static_cast<double>(kIters) * static_cast<double>(kFloats) *
                        2.0 / 1e9 / a_dt,
                    a_dt, static_cast<double>(tmp[0]));
    } else {
        std::printf("avx2 runtime NOT available on this machine\n");
    }
#else
    std::printf("(AVX2 kernel compiled out — enable /arch:AVX2 target flag)\n");
#endif

    std::printf("amx build status          : %s\n",
                llmoc::hal::amx_bf16_build_status());
    if (flags.os_supports_amx && flags.amx_bf16) {
        const int r = llmoc::hal::amx_bf16_selftest();
        if (r == 0)
            std::printf("amx bf16 selftest         : PASS (geometry verified)\n");
        else if (r == 1)
            std::printf("amx bf16 selftest         : SKIP (hw not ready)\n");
        else
            std::printf("amx bf16 selftest         : FAIL(%d) — geometry bug, CI gate!\n", r);
    } else {
        std::printf("amx bf16 hw               : not present/ready on this machine\n");
    }

    free_aligned(x);
    free_aligned(y);
    return 0;
}
