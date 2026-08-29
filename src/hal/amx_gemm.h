#pragma once
// llm-on-cpu :: hal/amx_gemm.h
// M0/M5: Intel AMX BF16 tile 微内核(docs/PLATFORM.md: 仅 SPR Linux 为目标环境)。
//
// 编译: CMake 探测 -mamx-tile -mamx-bf16(或 MSVC /arch:AMX)通过后定义
//        LLMOC_AMX_ENABLED 才编入实现; 否则桩实现(ready() 恒 false)。
// 验证: amx_bf16_selftest() 与 fp32 标量参考对拍 —— 在 SPR CI 上必须返回 0,
//        这是 AMX 路径的验收门禁(几何常量若有误在此暴露)。
//
// 几何(BF16, palette 1, 单 tile 16 行 × 64B):
//   A: M×K bf16, K 方向步长 32 元素
//   B: 以转置布局装入 tile —— N 行 × K 步长 32 列
//   C: M×N f32, N 方向步长 16 元素
// 入口要求: M%16==0, K%32==0, N%16==0 (边缘填充留给 M5 调度层)

#include <cstdint>

namespace llmoc::hal {

// ISA + OS(XCR0) 是否就绪(与编译开关无关, 供运行时探测)
bool amx_bf16_hw_ready();
const char* amx_bf16_build_status();  // "enabled" / "compiled-out(reason)"

// C[M,N] += A[M,K] · B[N,K]^T(注意: 入参 B 采用转置布局 N×K row-major)
// 仅在 hw_ready 且 build enabled 时调用, 否则行为为空操作并返回 false。
bool amx_bf16_gemm_accum(const uint16_t* A, const uint16_t* Bt, float* C,
                         int M, int K, int N);

// 自测: 小尺寸对拍 fp32 参考。返回 0=PASS, 非 0=FAIL, 1=环境不支持(跳过)
int amx_bf16_selftest();

}  // namespace llmoc::hal
