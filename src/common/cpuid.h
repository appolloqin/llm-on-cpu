#pragma once
// llm-on-cpu :: common/cpuid.h
// Cross-compiler CPUID wrappers + ISA feature detection (M0 host profile).

#include <cstdint>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#  include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#  include <cpuid.h>
#endif

namespace llmoc::cpu {

struct Regs { uint32_t eax, ebx, ecx, edx; };

inline Regs cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    Regs r{};
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int out[4];
    __cpuidex(out, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<uint32_t>(out[0]);
    r.ebx = static_cast<uint32_t>(out[1]);
    r.ecx = static_cast<uint32_t>(out[2]);
    r.edx = static_cast<uint32_t>(out[3]);
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int a = leaf, b = 0, c = subleaf, d = 0;
    __get_cpuid_count(leaf, subleaf, &a, &b, &c, &d);
    r.eax = a; r.ebx = b; r.ecx = c; r.edx = d;
#else
    (void)leaf;
    (void)subleaf;
#endif
    return r;
}

inline uint64_t xcr0() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return _xgetbv(0);
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#else
    return 0;
#endif
}

struct IsaFlags {
    bool sse42 = false;
    bool avx2 = false;
    bool avx512f = false;
    bool avx512_bf16 = false;
    bool amx_tile = false;
    bool amx_int8 = false;
    bool amx_bf16 = false;
    bool os_supports_amx = false;  // XCR0 has AMX state bits enabled
};

inline IsaFlags detect_isa() {
    IsaFlags f;
#if !(defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
    return f;
#else
    const Regs l0 = cpuid(0);
    if (l0.eax < 7) return f;  // legacy cpu, leave defaults
    const Regs l1 = cpuid(1);
    f.sse42 = (l1.ecx >> 20) & 1u;
    const bool os_avx = ((l1.ecx >> 27) & 1u) != 0 &&
                        ((xcr0() & 0x6ULL) == 0x6ULL);
    const Regs l7 = cpuid(7, 0);
    f.avx2 = os_avx && ((l7.ebx >> 5) & 1u);
    f.avx512f = os_avx && ((l7.ebx >> 16) & 1u) &&
                ((xcr0() & 0xE0ULL) == 0xE0ULL);
    const Regs l71 = cpuid(7, 1);
    f.avx512_bf16 = f.avx512f && ((l71.eax >> 5) & 1u);

    f.amx_tile = (l7.edx >> 24) & 1u;
    f.amx_int8 = (l7.edx >> 25) & 1u;
    f.amx_bf16 = (l7.edx >> 22) & 1u;
    const uint64_t xc = xcr0();
    f.os_supports_amx = f.amx_tile && ((xc >> 17) & 1ULL) && ((xc >> 18) & 1ULL);
    return f;
#endif
}

inline std::string brand_string() {
    char b[49]{};
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[12];
    __cpuid(reinterpret_cast<int*>(regs), 0x80000002);
    __cpuid(reinterpret_cast<int*>(regs) + 4, 0x80000003);
    __cpuid(reinterpret_cast<int*>(regs) + 8, 0x80000004);
    std::memcpy(b, regs, sizeof(regs));
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int regs[12];
    __get_cpuid(0x80000002, &regs[0], &regs[1], &regs[2], &regs[3]);
    __get_cpuid(0x80000003, &regs[4], &regs[5], &regs[6], &regs[7]);
    __get_cpuid(0x80000004, &regs[8], &regs[9], &regs[10], &regs[11]);
    std::memcpy(b, regs, sizeof(regs));
#endif
    return std::string(b);
}

}  // namespace llmoc::cpu
