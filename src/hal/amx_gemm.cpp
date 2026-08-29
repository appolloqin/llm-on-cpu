// llm-on-cpu :: hal/amx_gemm.cpp
#include "hal/amx_gemm.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "common/cpuid.h"

#if defined(LLMOC_AMX_ENABLED)
#  include <immintrin.h>
#  define LLMOC_AMX_IMPL 1
#endif

namespace llmoc::hal {

bool amx_bf16_hw_ready() {
    const auto f = cpu::detect_isa();
    return f.amx_tile && f.amx_bf16 && f.os_supports_amx;
}

#if !defined(LLMOC_AMX_IMPL)

const char* amx_bf16_build_status() {
    return "compiled-out (build toolchain lacks AMX flags)";
}

bool amx_bf16_gemm_accum(const uint16_t*, const uint16_t*, float*, int, int, int) {
    return false;
}

int amx_bf16_selftest() { return 1; }  // skip

#else  // ---- 真实实现(SPR Linux CI 验证路径) ----

namespace {

// tile 寄存器编号分配: 0=A, 1=Bt, 2=C
constexpr int kTileA = 0, kTileB = 1, kTileC = 2;
constexpr int kRowsMax = 16;          // tile 最大行数
constexpr int kColsBytes = 64;        // tile 每行最大字节
constexpr int kMStep = 16;            // M 方向步长
constexpr int kNStep = kColsBytes / static_cast<int>(sizeof(float));   // 16
constexpr int kKStep = kColsBytes / static_cast<int>(sizeof(uint16_t));  // 32

struct alignas(64) TileCfg {
    uint8_t palette_id = 1;
    uint8_t start_row = 0;
    uint8_t _rsv[14] = {};
    uint16_t colsb[8] = {};
    uint16_t rows[8] = {};
};

void load_config() {
    TileCfg cfg;
    cfg.palette_id = 1;
    cfg.rows[kTileA] = kMStep;  cfg.colsb[kTileA] = kColsBytes;  // A 16x32 bf16
    cfg.rows[kTileB] = kNStep;  cfg.colsb[kTileB] = kColsBytes;  // Bt 16x32 bf16
    cfg.rows[kTileC] = kMStep;  cfg.colsb[kTileC] = kColsBytes;  // C  16x16 f32
    _tile_loadconfig(&cfg);
}

}  // namespace

const char* amx_bf16_build_status() { return "enabled"; }

bool amx_bf16_gemm_accum(const uint16_t* A, const uint16_t* Bt, float* C,
                         int M, int K, int N) {
    if (!amx_bf16_hw_ready()) return false;
    if (M % kMStep || K % kKStep || N % kNStep) return false;

    load_config();
    for (int m0 = 0; m0 < M; m0 += kMStep) {
        for (int n0 = 0; n0 < N; n0 += kNStep) {
            _tile_zero(kTileC);
            for (int k0 = 0; k0 < K; k0 += kKStep) {
                // A 块: 16 行 × 32 bf16
                _tile_loadd(kTileA,
                            A + static_cast<size_t>(m0) * K + k0,
                            static_cast<long long>(K) * 2);
                // Bt 块: N×K 转置布局, 取 n0 行起 16 行 × 32 列(k 方向)
                _tile_loadd(kTileB,
                            Bt + static_cast<size_t>(n0) * K + k0,
                            static_cast<long long>(K) * 2);
                _tile_dpbf16_ps(kTileC, kTileA, kTileB);
            }
            _tile_stored(kTileC,
                         C + static_cast<size_t>(m0) * N + n0,
                         static_cast<long long>(N) * 4);
        }
    }
    return true;
}

int amx_bf16_selftest() {
    if (!amx_bf16_hw_ready()) return 1;  // skip: 非 SPR 环境

    constexpr int M = 16, K = 64, N = 32;  // 满足 M%16,K%32,N%16
    std::vector<uint16_t> A(static_cast<size_t>(M) * K);
    std::vector<uint16_t> B(static_cast<size_t>(K) * N);
    std::vector<uint16_t> Bt(static_cast<size_t>(N) * K);
    std::vector<float> C(static_cast<size_t>(M) * N, 0.f);
    std::vector<float> ref(static_cast<size_t>(M) * N, 0.f);

    auto b2f = [](uint16_t h) {  // bf16 -> f32
        uint32_t u = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<uint16_t>(0x3F80u + (i % 128));  // ~[1,2)
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = static_cast<uint16_t>(0x3F80u + (i % 64));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            Bt[static_cast<size_t>(n) * K + k] = B[static_cast<size_t>(k) * N + n];

    // fp32 参考
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            const float a = b2f(A[static_cast<size_t>(m) * K + k]);
            for (int n = 0; n < N; ++n)
                ref[static_cast<size_t>(m) * N + n] +=
                    a * b2f(B[static_cast<size_t>(k) * N + n]);
        }

    if (!amx_bf16_gemm_accum(A.data(), Bt.data(), C.data(), M, K, N)) return 2;

    for (int i = 0; i < M * N; ++i) {
        const float diff = std::fabs(C[static_cast<size_t>(i)] -
                                     ref[static_cast<size_t>(i)]);
        if (diff > 1e-2f * (1.f + std::fabs(ref[static_cast<size_t>(i)])))
            return 3;  // 数值超差
    }
    return 0;
}

#endif  // LLMOC_AMX_IMPL

}  // namespace llmoc::hal
