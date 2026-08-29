# 架构设计文档：LLM-on-CPU 纯CPU大模型推理引擎

> 版本: v0.4 (已评审·冻结基线)
> 修订记录: v0.4 全局复查修正——第2节 DRAM 带宽项口径改为 B_total(含缺页字节)；3.1 补充模式②算子分工；第5节预算表标注适用范围；决策点按默认建议推进（Q1 列为实现阶段校准项）
>           v0.3 三种执行模式（纯CPU[默认]/GPU+CPU混合/纯GPU）、PlacementPlanner/DeviceHAL、M5、R6、Q6
>           v0.2 先例分析、D6 弹性内存调拨器、R5、Q5
> 目标: DeepSeek-V4-Flash-0731 类稀疏 MoE 模型，**不量化**；三模一引擎，默认纯 CPU 模式下单流 decode ≥ 30 token/s
> 状态: ✅ 设计冻结，进入实现文档阶段

---

## 1. 背景与目标

### 1.1 SLO（服务等级目标）

> 以下指标针对**默认纯 CPU 模式**；混合/纯 GPU 模式的 SLO 在 3.1 节定义，随 M5 里程碑另行验收。

| 指标 | 目标值 | 说明 |
|---|---|---|
| 单流 decode 速度 | **≥ 30 tok/s** | 核心验收指标 |
| TTFT（首 token 延迟） | P99 < 800ms @ 输入2K | 分数与预取预热后的稳态 |
| 并发吞吐 | batch=8 时聚合 ≥ 150 tok/s | 拉伸目标 |
| 精度 | 与 BF16 参考实现对拍 logits 余弦相似度 > 0.999 | 不量化承诺 |
| 可用性 | 7×24, OOM/盘慢自动降级不崩溃 | 工业级要求 |

### 1.2 硬约束

- 主机内存: 64 GB（含 OS 占用后可用 ~58 GB）
- 默认模式无 GPU；可选模式见 3.1 节
- **不量化**（BF16 全精度权重，所有模式一致）
- 模型为稀疏 MoE 结构（假设值，待 config.json 实测校准）:
  - 每 token 激活参数 ≤ 10B（BF16 ≤ 20GB/token 读取量）
  - 路由专家数 N_experts 大（如 128~256），每 token top-k（如 8）

## 2. 第一性原理与核心公式

四级存储阶梯:

```
AMX寄存器(tile) → L3(~100MB/路) → DRAM(64G) ← NVMe(容量无限, 吞吐有限)
```

每 token decode 时间（流水线重叠后取最大者，而非相加）:

```
T_token = max(
    T_compute          ,   # AMX 计算时间
    B_total   / BW_dram ,   # DRAM 带宽承载全部字节(常驻读取 + 缺页数据落地后被消费)
    B_miss    / BW_nvme ,   # 缺页专家从 NVMe 流式读取, 可与计算/DRAM 流水重叠
)

原始速度 = BW_dram / B_total(BF16 激活字节)
有效速度 = 原始速度 × MTP接受倍数(k≥2 时 2~2.5×)
约束     = B_miss × 有效速度 ≤ BW_nvme
```

**达标判据（以 M0 实测带宽代入计算为准）**：
- 激活 ≤ 4B → 原始即可达标；激活 6~10B → 必须叠加 MTP；激活 > 12B → 判定单机不可行。

## 3. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│ api-server   OpenAI 兼容 HTTP + SSE 流式 / 指标导出             │
├──────────────────────────────────────────────────────────────┤
│ mode-controller  执行模式选择(①cpu默认/②hybrid/③gpu) + 运行时切换 │
├──────────────────────────────────────────────────────────────┤
│ scheduler       continuous batching                           │
│                 └ spec-decode verify 循环(MTP)                 │
├──────────────┬─────────────────────────┬─────────────────────┤
│ kv-manager   │ weight-manager ⭐核心      │ router-stats        │
│ paged pool   │ └ PlacementPlanner       │ (离线工具产物)         │
│ + radix 树   │   热温冷 N 级驻留求解       │ 专家频次表/预热计划    │
│ + 语义锚点    │ 双缓冲 io_uring 预取       │                     │
├──────────────┴─────────────────────────┴─────────────────────┤
│ DeviceHAL (设备抽象)                                            │
│   ├─ CPU 后端: AMX BF16 GEMM(oneDNN) / fused attn / NUMA 分配   │
│   └─ CUDA 后端: tensor core GEMM / FlashInfer 类 attention       │
├──────────────────────────────────────────────────────────────┤
│ 存储层   VRAM(L1热区,模式②③) ─ DRAM(mlock, L2) ─ NVMe(O_DIRECT, L3) │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 执行模式设计（三模一引擎，默认纯 CPU）

统一原则：**一套调度/KV/权重管理代码，只有"资源分层拓扑"与"算子后端"不同**。由 PlacementPlanner 按实测带宽求解驻留方案（带宽自适应 q★ 策略）。

| 模式 | 触发条件 | 权重放置策略 | 性能定位 |
|---|---|---|---|
| ① pure-cpu（默认） | 无可用显卡 | attn/dense/共享专家常驻 DRAM；路由专家 LRU@DRAM；冷专家 NVMe | 本文全篇公式，目标 ≥30 tok/s |
| ② hybrid-gpu | 存在 8~24G 消费级卡 | **固定入 VRAM**: 全部 attn 权重+KV+MTP头（体积小、访问频繁）；VRAM 余量作 L1 专家热区放最热专家；DRAM 作 L2 热区；NVMe 兜底 | VRAM ~1000GB/s 抬高速度上限，PCIe(32~64GB/s) 成为新瓶颈项参与 T_token 取 max |
| ③ pure-gpu | 全模型激活路径 + KV 可装入全部 VRAM | 权重整体常驻 VRAM，offload 路径整体关闭 | 退化为经典服务引擎，吞吐(batching)优先 |

T_token 公式泛化（三模统一）：

```
T_token = max(
    T_compute(当前模式设备),
    max_over_tiers( B_tier / BW_tier ),        # 各级存储读取，流水线重叠后逐级取max
    B_pcie / BW_pcie,                          # 仅②存在: host↔device 流式项
)
```

要点：
- 模式②中 PCIe 是串行链路：预取流水必须以 PCIe 实测带宽为约束重排深度
- **模式②算子分工**：attention/KV/MTP 头相关算子在 GPU 侧执行（数据已驻 VRAM），路由专家 GEMM 在 CPU 侧执行；两层间传输的仅为 MB 级激活向量（不是 GB 级权重），PCIe 往返计入 T_compute 而非流式项
- 模式③复用同一 weight-manager，只是三级全部命中（NVMe/DRAM 路径闲置），代码零分支特殊化
- 模式选择在加载期确定；运行时切换为拉伸目标（依赖 D6 调拨器）

### 3.2 关键机制一览

| 机制 | 在本引擎中的落点 |
|---|---|
| Paged KV + continuous batching | kv-manager page 化；scheduler 批调度 |
| 前缀 radix 复用 | kv-manager 内置前缀基数树 |
| 参数热温冷分级卸载 | weight-manager 的三级驻留（专家粒度） |
| 层级流式按需加载 | 内存告急时的降级模式（退化到层粒度） |
| 多 token 并行产出 | MTP 投机解码 verify 循环 |
| 专家级 offload + 带宽自适应 | weight-manager / PlacementPlanner / D6 |

## 4. 关键设计决策

### D1 权重驻留策略（空间换时间，LRU 为运行时主体）
- **常驻区**: embedding、所有层的 attention 权重、共享专家、router —— mlock 锁页防换出
- **热专家区**: 全局 **LRU 缓存**为运行时主体；启动时用离线路由统计表做**预热填充**以消除冷启动颠簸（两者混合）
- **冷专家**: 留在 NVMe，请求时 demand paging

### D2 流水线预取（并行③——IO/计算重叠，双路径划分）
- **prefill 路径**: 整层双缓冲流式（IO 友好）
- **decode 路径**: 专家粒度双缓冲——当前层计算期间，后台经 io_uring(O_DIRECT) 将下一层所需 tensor DMA 进预备缓冲
- MoE 门控在第 L 层输出后才可知 → L+1 层专家在 gate 后才拉取，属于"晚预取"；但 attention/dense 部分可全程提前流水，整体仍是零等待为主

### D3 投机解码（并行④——带宽乘法器）
- 主模型一次前向 + MTP(nextn) 头草拟 k 个后续 token → 一次 verify 读一遍权重验 k 个
- acceptance ratio 目标 ≥ 2.0（offload 场景下 MTP 仍有效，量级约 1.8x～2.5x），等效把带宽墙左移 ~2×
- 若 V4-Flash 未随附 MTP 头权重 → 见风险 R2 的替代方案

### D4 KV 管理（paged + radix + 语义锚点）
- KV pool 按 page（如 256 token/page）分配，杜绝碎片
- radix tree 缓存共享前缀 KV，多轮对话/系统提示词免重算
- **语义锚点检查点**：在工具调用/思考块等"编辑型上下文"边界保存 KV 锚点，上下文修改时只从最近有效锚点重算，面向 agent 型负载

### D5 算子双后端与 NUMA（并行②——节点内张量分割）
- DeviceHAL 统一算子接口，两后端实现：
  - **CPU 后端**: BF16 + Intel AMX tile 指令 GEMM（oneDNN 兜底），NUMA 本地分配本地执行，禁 cross-node 访问
  - **CUDA 后端**: tensor core GEMM + FlashInfer 类 fused attention（服务模式②③）
- 同一权重驻留计划在不同后端间保持一致的 granular 权重格式

### D6 弹性内存调拨器（q★ 带宽自适应，跨 VRAM/DRAM/NVMe 三级）
- 相邻层级间的配比**运行时可动态调整**（不重启、不重载权重）：
  - 模式①: 热专家缓存区 ↔ KV pool（DRAM 内部）
  - 模式②: VRAM 专家热区 ↔ VRAM KV 区；DRAM L2 ↔ NVMe
  - 模式③: VRAM 内部 KV 区 ↔ 冗余缓冲
- 控制信号：实测各级带宽利用率 + 缺页率 + KV 池压力 → 周期性微调配比

## 5. 内存预算表（64G 主机，**适用模式①默认布局**；模式②③由 PlacementPlanner 动态求解）

| 区域 | 预算 | 备注 |
|---|---|---|
| 常驻区(dense+attn+emb+shared) | ~8 GB | 视层数而定 |
| 热专家驻留区 | ~32 GB | N_hot 按 M0 统计动态调整 |
| KV page pool | 8 GB | 支持 radix 复用 |
| 预取双缓冲 | ~6 GB | 覆盖下一层所有需拉取 expert |
| MTP 头 + workspace | ~2 GB | — |
| OS + 监控余量 | ~5 GB | cgroup 保护线 |
| **合计** | **≈61 GB** | 触发告警阈值 90% |

## 6. 模块职责与关键接口（接口级定义，实现细节归实现文档）

```
WeightManager:
  plan(residency_spec)            # 加载热温冷计划(由 PlacementPlanner 求解)
  prefetch(layer_id, expert_ids)  # 异步双缓冲预取 (返回 future, 约束于当前模式带宽链)
  pin(layer_id, expert_ids)->ptrs # 就绪指针, miss 时同步兜底
PlacementPlanner:
  solve(hw_profile, workload_profile)->residency_spec   # q★推广: 按 M0 实测带宽求解三级驻留
  rebalance(stats)->delta                               # 运行时调拨(D6)
KVManager:
  alloc(n_pages)/radix_lookup(prefix_hash)/anchor_save/load/evict()
Scheduler:
  submit(req)->stream             # batching + 投机解码编排
RouterStatsTool(离线):
  sample_dataset -> expert_freq.json -> warmup_plan.json
DeviceHAL:
  gemm/attn/rmsnorm/silu/transfer  # CPU(AMX) 与 CUDA 双后端实现同一接口
```

## 7. 降级与容错

| 触发条件 | 动作 |
|---|---|
| 内存水位 >85% | 热区收缩（降低 N_hot），冷拉增多 |
| NVMe 读延迟飙升 | 降并发批大小，优先保已接请求 SLO |
| 连续 OOM 风险 | 退化为层级全流式模式（慢但不死） |
| 路由偏差超阈 | RouterStats 定期重采样自动更新驻留计划 |

## 8. 测试与验收标准

1. **正确性**: 固定 prompt 下与 PyTorch CPU BF16 参考实现对拍，logits 余弦相似度 > 0.999
2. **性能门禁**: 目标硬件实测单流 ≥30 tok/s（M0 数据代入预测先行评审，跑不进预算直接回炉）
3. **压力**: 72h 稳定性 + 内存水位监控不泄漏
4. **单元测试覆盖**: weight-manager/KVManager/scheduler 各自独立 mock 存储/内核测试

## 9. 风险清单

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 实测激活参数 >12B | 物理不可达 30t/s | M0 先测，及时止损转多机/混合方案（纯 CPU 结论以 M0 实证为准） |
| R2 | 无现成 MTP 头 | 少了最大乘法器 | 自训 EAGLE 式轻量头（数据 20~50B token 子集），或退化为纯流水线方案 |
| R3 | 路由偏斜度不足 → LRU 缺页率高打爆 NVMe | 缺页率高打爆 NVMe | 扩大热区/缩小中间维微调（非量化剪枝另议）；D6 调拨器自动收缩 KV 池补给专家缓存 |
| R4 | Windows 开发环境 vs Linux 生产目标差异 | io_uring/AMX 不可用于开发机 | CI 目标环境为 Linux 容器；开发机只跑逻辑层+mock |
| R5 | 与外部开源引擎机制重叠，重复建设 | 维护成本 | 见 Q5：机制自研落地，外部仅作可选对标 |
| R6 | 三模式并存导致开发面与测试矩阵膨胀 | 交付延期 | DeviceHAL 抽象先行，模式①为主干主线；②③各作为独立里程碑验收（M5），不阻塞默认模式达标 |

## 10. 里程碑

- **M0** 平台基线压测（DRAM 带宽 / NVMe 顺序读 / AMX GEMM flops；如有卡: VRAM 带宽 + PCIe 双向传输带宽）→ 校准第 2/3.1 节公式
- **M1** weight-manager + DeviceHAL(CPU后端) 单流逐层推理正确性对拍
- **M2** 热驻留 + 预取流水线 → 模式①单流冲 20+ tok/s
- **M3** MTP verify 循环 → 模式①冲 30 tok/s 达标线（**主干验收点**）
- **M4** batching / radix KV+语义锚点 / API 服务化 / 监控告警 / 降级开关
- **M5** PlacementPlanner 落地，启用模式②hybrid-gpu 与 ③pure-gpu：CUDA 后端接入、PCIe 流水调优、D6 跨级调拨验证

---

## ⚠️ 待确认决策点（回复编号即可）

| # | 决策点 | 默认建议 |
|---|---|---|
| Q1 | 模型 config.json 实际参数（总参/激活/N_experts/top_k/是否有MTP头） | 需要，直接影响预算成立性 |
| Q2 | 三模式硬件矩阵：主机是否 SPR 至强；消费级卡型号区间（决定模式②基线） | SPR + NVIDIA RTX 30/40/50 系 |
| Q3 | 技术栈：C++20 核心 + Python 离线工具链 | C++ 核心，oneDNN 兜底底座 |
| Q4 | 部署形态是否纳入 K8s 编排（还是仅单机进程级交付） | 先单机交付 |
| **Q5** | **工程路线：自研引擎为主**；专家 offload / LRU / 双缓冲等机制按本文档落地 | **自研落地**；可选同机外部引擎仅作性能对标下界，不作运行时依赖 |
| **Q6** | 模式②③是否需要非 NVIDIA 显卡（ROCm/AMD、Intel Arc）支持 | 首版仅 CUDA；HAL 预留扩展位 |
