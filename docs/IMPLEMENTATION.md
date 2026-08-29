# 实现文档：LLM-on-CPU 推理引擎

> 版本: v0.1 (待评审)
> 对应架构: docs/ARCHITECTURE.md **v0.4 (冻结基线)**
> 状态: 🟡 评审通过后开始编码

---

## 1. 实现语言推荐与理由 ⭐

### 1.1 结论

| 层 | 语言 | 说明 |
|---|---|---|
| **核心引擎**（kernel / weight-manager / scheduler / kv-manager / server） | **C++20** | 全部在 decode 热路径与内存关键路径上 |
| **离线工具链**（router-stats / convert_lwc / 对拍脚本 / bench 分析） | **Python 3.11+ 或 Node≥18 双栈** | 下载/转换已有 `.py` 与零 npm 依赖的 `.mjs` 对等实现（产物 MD5 一致，CI 双跑互验）；不在任何热路径上 |
| 测试胶水 | pybind11（仅 tests 用） | 可选 |

### 1.2 选 C++20 的理由（按权重排序）

1. **依赖生态零 FFI**：本项目三大命脉依赖——oneDNN/AMX intrinsics(`immintrin.h`)、liburing(io_uring)、CUDA(cuBLAS/cuDNN/FlashInfer C++ API)——全部是 C/C++ 一等公民。选 Rust 意味着三者都要维护手工 binding，FFI 边界恰好在性能最敏感处。
2. **延迟确定性**：工业级 30 tok/s SLO 是 P99 口径。C++ 无 GC/JIT 抖动，能做：
   - `mlock` 锁页 + `numa_alloc_onnode` 本地分配 + hugepage
   - O_DIRECT 要求的 4K 对齐缓冲精确控制
   - 无锁双缓冲预取队列（seqlock/RCU）
3. **业界惯例**：主流高性能推理引擎多为 "C++/CUDA 内核 + Python 外壳" 结构，人才池与踩坑经验可复用。
4. **AMX 裸指令可达**：D5 要求 BF16 tile GEMM 达到理论带宽，oneDNN 做兜底、自写 tile 微内核做快路径——这只有 C++ 能干净实现。

### 1.3 为什么不是其他语言

| 备选 | 否决原因 |
|---|---|
| Rust | 内存安全收益大，但 oneDNN/CUDA/liburing binding 成本高；推理内核参考实现少；本项目瓶颈是 IO/带宽工程而非内存安全。若团队偏好 Rust，可后续用 Rust 重写调度层（非内核）作为独立演进项 |
| Go/Java | GC 停顿破坏 P99；无 SIMD/NUMA 底层生态，不适合做 kernel 层 |
| 纯 Python | 仅作控制面。decode 循环内任何 Python 解释器开销都会吃掉预算里的毫秒级余量 |

## 2. 目录结构

```
llm-on-cpu/
├─ docs/                       # ARCHITECTURE.md / IMPLEMENTATION.md
├─ src/
│  ├─ common/                  # 日志/配置/线程池/numa_alloc/metrics registry
│  ├─ hal/                     # DeviceHAL 接口 + 双后端
│  │  ├─ include/hal/device.h          # Ops 抽象基类
│  │  ├─ cpu/                          # amx_gemm.cpp fused_attn.cpp rmsnorm.cpp silu.cpp
│  │  └─ cuda/                         # M5 阶段启用
│  ├─ weights/
│  │  ├─ lwc_format.h/.cpp             # LWC 权重容器格式读写(见 §4)
│  │  ├─ io_engine.cpp                 # io_uring 双缓冲预取(O_DIRECT)
│  │  ├─ lru_hotset.cpp                # 全局 LRU 专家缓存(mlock)
│  │  ├─ planner.cpp                   # PlacementPlanner.solve()/rebalance()
│  │  └─ weight_manager.cpp            # pin/prefetch 门面, 三模式统一路径
│  ├─ kv/
│  │  ├─ page_pool.cpp                 # paged KV(256 tok/page)
│  │  ├─ radix_tree.cpp                # 前缀基数树
│  │  └─ anchors.cpp                   # 语义锚点检查点
│  ├─ model/
│  │  ├─ deepseek_v4_flash.cpp         # 计算图: attn→gate→experts→combine
│  │  ├─ mtp_head.cpp                  # nextn 草拟头
│  │  └─ verify_loop.cpp               # spec-decode 主循环(M3)
│  ├─ sched/
│  │  ├─ mode_controller.cpp           # 启动探测选①②③
│  │  ├─ batcher.cpp                   # continuous batching
│  │  └─ engine.cpp                    # 组装入口 Engine::start()
│  └─ server/
│     ├─ http_api.cpp                  # OpenAI 兼容 /v1/chat/completions + SSE
│     └─ metrics_endpoint.cpp          # /metrics Prometheus 文本格式
├─ tools/
│  ├─ m0_bandwidth/                    # M0: DRAM/NVMe/PCIe/VRAM 带宽压测(C++)
│  ├─ m0_amx_flops/                    # M0: AMX GEMM flops
│  ├─ router_stats.py                  # 专家频次采样 → warmup_plan.json
│  ├─ convert_lwc.py                   # HF safetensors → .lwc
│  └─ compare_logits.py                # 与 PyTorch 参考实现对拍
├─ configs/
│  └─ engine.yaml                      # 运行配置(见 §7)
├─ tests/
│  ├─ unit/                            # gtest: 不触真实盘/GPU, mock IOEngine & HAL
│  ├─ integration/                     # docker 内真实小模型逐层对拍
│  └─ goldens/prompts.jsonl
├─ scripts/
│  ├─ build.sh / Dockerfile.ci         # Linux CI 构建环境
│  └─ dev_windows.ps1                  # 开发机仅构建逻辑层单测(MSKL mock 后端)
└─ third_party/                        # FetchContent: oneDNN liburing gtest benchmark
```

## 3. 构建系统与依赖

- **CMake ≥ 3.25**，`CMakePresets.json`：`linux-release` / `linux-cuda`(M5) / `windows-dev`
- 编译器：gcc 12+ 或 clang 16+，`-march=sapphirerapids`（CPU 后端）；CUDA 12.x optional `-DLLMOC_WITH_CUDA=ON`
- Windows 开发机构建范围：`common/tests/unit 中不依赖 posix 的目标` + MSKL(mock kernel layer)；io_uring/AMX 目标不编译（对应架构 R4）

| 依赖 | 版本 | 引入方式 | 用途 |
|---|---|---|---|
| oneDNN | v3.x | FetchContent | CPU GEMM 兜底 |
| liburing | 2.x | FetchContent | io_uring 预取 |
| gtest / google-benchmark | latest | FetchContent | 测试 |
| pybind11 | latest | FetchContent | 测试对拍胶水 |
| cpp-httplib | 0.15+ | FetchContent | HTTP/SSE（无重依赖，符合"先单机交付"） |

## 4. LWC 权重容器格式（v1，已按本节实现）

> 修订: 初版曾计划 JSON 头; 实现阶段改为**自描述二进制目录**(零外部依赖、C++/Python
> 双端严格一致)。本节为最终口径。

取代直接读 safetensors —— 解决"专家粒度随机访问 + O_DIRECT 对齐"两个硬需求：

```
文件布局:
[0 ..24)                preface: magic 'LWC1' + version(u32)
                                 + catalog_len(u64) + catalog_crc(u64, fnv1a64)
[24 ..24+catalog_len)   catalog 二进制目录:
                          dtype(u32) block_align(u32)
                          n_tensors(u64) n_groups(u64)
                          每张量: name + shape + offset + nbytes + checksum
                          每专家组: layer(u32) expert_id(u32) + 3 个张量名(gate,up,down)
[block N ...]           数据区: 每 block 起始 4K 对齐(O_DIRECT 硬要求),
                        专家 {gate,up,down} 连续段 → 单专家可用少量 READV 取齐
```

- dtype: BF16=1 / F16=2 / F32=3（不量化承诺内的全部精度）
- **checksum=0 哨兵**: Python 转换器落盘 0（FNV 链式计算在 Python 是分钟级），
  由 `lwc_verify --update` 以 C++ 速度回填真实校验和并经 `RewriteCatalog` 原位重写目录
- 读路径 A: 常驻区整块读入 DRAM 并 `mlock`（WeightManager）
- 读路径 B: 运行期按 block 粒度异步直读进双缓冲槽（ExpertPrefetcher）
- 工具链: `tools/convert_lwc.py`(HF→LWC) · `tools/lwc_verify`(校验/回填/config 交叉核对/预算判定)
  · `scripts/make_fake_hf.py`(微型假模型, 全链自检)

## 5. 核心模块实现要点

### 5.1 io_engine（对应 D2）
```cpp
// 双槽环形缓冲, io_uring depth=64, 单数产 thread
future<Span> prefetch(LayerId l, ExpertIds e); // 提交 READV 到空闲槽
Span slot_swap();                              // 前向线程每层边界 swap 一次, miss 时同步兜底读
```
- 缓冲区从 `posix_memalign(4096)` 分配并 mlock
- prefill：按整层提交批量 READV；decode：gate 出来后仅拉 miss expert

### 5.2 weight_manager + planner（D1/D6）
- 三种 tier 归一为 `Tier{DRAM_HOT, VRAM_HOT, NVME_COLD}`，weight_manager 只处理 tier 序列，对模式透明
- `PlacementPlanner::solve()`：贪心装袋——专家按 warmup_plan 频次降序装入预算区直到满；`rebalance()` 每 5s 依据 `{miss_rate, bw_util, kv_pool_pressure}` 调整 tier 配比（迁移单位=block）
- LRU 热区命中计数与路由日志落盘（供 router_stats.py 周期性再训练计划）

### 5.3 model 计算图与 spec-decode（D3）
```
verify_loop(step):
  h = main_forward(last_token, k_draft_tokens=3 组成的候选序列)
  accepted = greedy_match(logits, draft_tokens)      # 前缀接受
  drafts'  = mtp_head(h, accepted_last)              # 为下一轮补草稿
  emit(accepted_tokens)
```
- forward 线程组 = 每 NUMA node 一个池，GEMM 在本地池执行（D5）
- 金线规则：**调度线程不做任何浮点计算**，只搬指针和事件

### 5.4 kv-manager（D4）
- page=256 token；radix 节点 hash 按 chunk(64 tok) 粒度；anchor=page 引用集快照 + 版本号
- 淘汰顺序：冷 radix 叶 → 冷 anchor → FIFO page

### 5.5 server 与运维面
- `/v1/chat/completions`（stream=true 走 SSE）、`/healthz`、`/metrics`
- 认证：`LLMOC_API_KEY` env 可选 Bearer
- metrics 关键项：`tps{mode}, miss_rate, bw_util_dram/nvme/pcie, kv_util, queue_depth, ttft`

## 6. 线程模型与锁边界

| 线程 | 数量 | 职责 | 同步 |
|---|---|---|---|
| forward-pool | 每 NUMA node N/2 个 | AMX GEMM/attn | 层边界 barrier |
| io-engine | 1~2 | io_uring 提交/收割 | SPSC 环 |
| scheduler | 1 | batching/spec 编排 | 事件队列传令 |
| rebalancer | 1 (低频) | D6 调拨 | RCU 读指针切换 |
| server | httplib 自带 | HTTP/SSE | 无锁队列入 sched |

锁原则：热路径只有 atomic load/store 与 SPSC/MPSC 无锁队列；tier 元数据用 seqlock。

## 7. 配置 schema（configs/engine.yaml 示例）

```yaml
model: { path: models/v4flash.lwc, dtype: bf16, mtp: true }
mode: auto            # auto|pure_cpu|hybrid_gpu|pure_gpu  (auto=ModeController探测)
tiers:
  dram_hot_gb: 32
  kv_pool_gb: 8
  prefetch_buf_gb: 6
decode: { spec_k: 3, accept_min: 2.0 }
rebalance: { interval_s: 5 }
server: { port: 15085, api_key_env: LLMOC_API_KEY }
```

## 8. 测试方案（对齐架构 §8 验收）

| 层级 | 内容 | 工具 |
|---|---|---|
| unit | lru/planner/lwc 读写回环/radix/batcher 状态机（mock IOEngine+MockHAL，纯 CPU 秒级跑完） | gtest |
| integration | docker(Ubuntu22.04) 内加载小型 MoE 金样模型逐 token 与 torch 对拍 cos>0.999；注入 fake-disk 延迟验证降级路径 | pytest + pybind |
| perf(M0) | tools/m0_bandwidth、m0_amx_flops 实测数字回填 ARCHITECTURE §2 公式评审表 | 自研 bench |
| perf门禁 | bench_tps 单流 ≥30 且 P99 波动 <15%（目标硬件 CI runner 上夜间执行） | google-benchmark |
| soak | 72h 持续负载 RSS 曲线平坦 | prometheus + alert |

## 9. 里程碑交付物映射

| 里程碑 | 交付物（目录） | 验收命令（示例） |
|---|---|---|
| M0 | tools/m0_* | `./m0_bandwidth --json > hw_profile.json` |
| M1 | src/weights/lwc+io_engine, hal/cpu, tools/convert_lwc.py, compare_logits.py | `pytest tests/integration/test_golden_logits.py` |
| M2 | lru_hotset + planner(初版) + 双缓冲全链 | `bench_tps --target 20` |
| M3 | mtp_head + verify_loop | `bench_tps --target 30` ✅主干验收 |
| M4 | batcher/radix/anchors/server/metrics | `wrk -c8 ... &&Prometheus 采集核对` |
| M5 | hal/cuda + PlacementPlanner 完整 + mode②③ | 三模式 bench 矩阵报告 |

## 10. 开放项（编码期间关闭）

| # | 开放项 | 关闭手段 |
|---|---|---|
| O1 | V4-Flash config.json 实测参数（ARCH Q1） | 到位后回填 §4 header 生成参数与 §5.3 draft_k 默认值 |
| O2 | MTP 头是否随权重发布（ARCH R2） | 有→直载；无→M3 先空实现接口，走纯流水线达标路线评估 |
| O3 | 外部引擎对标基线（ARCH Q5） | M2 前可选跑一轮同机/同体量基准留档对比 |
