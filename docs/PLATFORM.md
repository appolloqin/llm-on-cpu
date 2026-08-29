# 平台支持矩阵：Windows / Linux / macOS

> 对应: ARCHITECTURE v0.4 风险 R4 缓解措施的正式化
> 原则: **核心逻辑三端同构**；性能关键的平台特性用编译期开关选路，缺失时自动降级并打日志。

## 能力矩阵

| 能力 | Windows | Linux | macOS |
|---|---|---|---|
| 角色 | 开发机(R4) + 模式②③客户端可选目标 | **生产主目标**(SPR 至强) + CI 环境 | 逻辑层验证机(可选) |
| 构建脚本 | `scripts/configure_dev.cmd` / `build_dev.cmd` (MSVC+Ninja) | `scripts/build_linux.sh` 或 `scripts/Dockerfile.ci` | 同 Linux(clang)，脚本后补 |
| 执行模式① pure-cpu | ⚠️ 可跑但达不到 30t/s（无 AMX、带宽低） | ✅ 目标形态 | ❌ 不做性能主张 |
| 执行模式② hybrid-gpu | 可选(M5) | 主推(M5) | 不支持 |
| 执行模式③ pure-gpu | 可选(M5) | 支持(M5) | 不支持 |
| AMX tile 微内核 | ❌ 无此 ISA(M0 实测) | ✅ M0-CI 交付 | ❌ 无此 ISA |
| 异步文件引擎 | 线程池通用版(本轮)；IOCP 直读版后补 | 本轮线程池版 → **io_uring+O_DIRECT**(M1 收尾) | 线程池通用版 |
| 锁页 | `VirtualLock`(需提升配额) | `mlock` ✅ 生产使用 | `mlock` 可用 |
| NUMA 绑定 | 无意义(消费级单节点) | `numa_alloc_onnode` 等(D5 核心) | 无 |
| 单测覆盖 | ✅ 全量(离线跑) | ✅ 全量(CI) | ✅ 全量 |

## 三端共同约定

1. 所有平台相关分支集中在两处：`src/common/platform.h` 与各后端目录(`hal/cpu`, `hal/cuda`)的文件内部——**调度/KV/权重管理代码零 `#ifdef`**
2. 引擎工厂 `make_io_engine()` 是唯一异步 IO 替换点；当前返回跨平台线程池实现，Linux 上 io_uring 版落地后由构建开关接管工厂返回值
3. 锁页调用是 best-effort：失败仅 LOG_WARN，不阻断启动（Windows 无特权时会静默退化）

## 各平台里程碑对应

| 里程碑 | Windows | Linux | macOS |
|---|---|---|---|
| M0 已完成 | ✅ DRAM/NVMe/FLOPS 实测 | 待 CI(Linux SPR 需真机或云裸金属) | 可跑 |
| M0 AMX 微内核 | ➖ | 代码+自测门禁已就位, 待 SPR 实跑 | ➖ |
| M1 权重链路 | ✅ 线程池版实测 | ✅ 代码就绪(io_uring 已编入, 待 Linux 实机) | 可跑 |
| M2 性能冲刺 | ❌(带宽不够) | ✅ | ❌ |
