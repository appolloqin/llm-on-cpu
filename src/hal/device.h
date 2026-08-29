#pragma once
// llm-on-cpu :: hal/device.h
// DeviceHAL 算子抽象 (ARCHITECTURE D5)。M1 起逐步充实; 本轮仅立骨架 + Mock。

#include <cstddef>
#include <cstdint>

namespace llmoc::hal {

enum class Backend : uint8_t { kMock = 0, kCpuAvx2 = 1, kCuda = 2 };

// 算子门面。具体内核见 cpu_ops.h (BF16/F32 参考实现)。
class Ops {
   public:
    virtual ~Ops() = default;
    virtual Backend backend() const = 0;
};

// 空实现: 开发机(R4)与单测默认后端。
class MockOps final : public Ops {
   public:
    Backend backend() const override { return Backend::kMock; }
};

}  // namespace llmoc::hal
