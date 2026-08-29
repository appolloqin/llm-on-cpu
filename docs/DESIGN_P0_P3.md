# 设计文档：四阶段（P0→P3）

> 版本: v1.0（**已确认** 2026-08-29 — 用户全部同意 DESIGN§9 / IMPL 选项）  
> 状态: 🟢 **设计已确认，编码进行中（P0）**  
> 关联: [ARCHITECTURE.md](ARCHITECTURE.md) v0.4 · [IMPLEMENTATION_P0_P3.md](IMPLEMENTATION_P0_P3.md)  
> 原则: **机制进本引擎**，不把第三方推理服务当默认后端。

---

## 1. 目标与非目标

### 1.1 目标（按序）

| 阶段 | 名称 | 一句话目标 |
|---|---|---|
| **P0** | 带宽乘法器落地 | MTP 进入真实 generate；decode 逼近本机带宽墙 |
| **P1** | 模型无关执行 | 新架构 = LayerSpec + 权重映射，不新增巨型 `*_model.cpp` |
| **P2** | 调度与分页 KV | continuous batching + paged KV |
| **P3** | MoE 主航道 | 专家 LRU/预取端到端 + 同机对标下界 |

### 1.2 非目标

- 把第三方推理服务当默认后端
- P0 阶段就重写全部模型前向（P0 仍挂现有 Qwen 图验证速度）
- P1 一次支持全部 HF 架构（先 Qwen3.5 配方化 + 一个标准 Transformer）
- P2 做分布式多机
- P3 在无真实 MoE LWC 时伪造「达标」

### 1.3 成功判据（产品口径）

| 阶段 | 必须满足 |
|---|---|
| P0 | 暖机单流 decode 相对当前基线 **≥3×**；MTP `tokens_per_step` 在可测 α 下贴近理论；同机同体量外部 CPU 引擎对标报告落地 |
| P1 | 第二架构（标准 GQA Transformer）**零新增前向 .cpp** 跑通 chat |
| P2 | 双请求 continuous batch 正确；paged KV 下长上下文 RSS 可控；radix 命中省去重复 prefill 算力 |
| P3 | 选定 MoE LWC 端到端对话；专家缺页路径不崩；与同机外部引擎（或书面说明不可比条件）对标表 |

---

## 2. 总体数据流（四阶段完成后）

```
Client
  → HttpApi / UI
  → Scheduler (P2: 多槽 continuous batch)
       │
       ├─ SpecDecode loop (P0: 真接入)
       │     draft_k → verify 前向 → accept+next
       │
       └─ GraphExecutor (P1)
             for layer in LayerSpec[]:
               ops(attn|linear|moe) + WeightManager get/prefetch (P3)
             → logits
  → 流式/非流式响应
```

与现状差距：

| 现状 | 目标 |
|---|---|
| `Generator` 逐步 `forward(1)` | P0：`SpecDecodeRunner` 驱动，一次 verify 可吐多 token |
| `Qwen35Model::layer_forward` 硬编码 | P1：`GraphExecutor` + Spec |
| `SessionCache` 连续大数组 + 假 radix | P2：块表 paged + 真前缀复用 |
| MoE 仅自测/小样 | P3：真实专家 LRU+双缓冲上业务路径 |

---

## 3. P0 设计 — MTP + 热路径收紧

### 3.1 问题

- `spec_decode` 仅 mock/单测，**未进入** `Generator::generate`
- 每 token 完整扫权重 → 带宽墙无法被「每步多 token」摊薄
- BF16 路径仍有大量临时 `vector`；INT4 已部分 scratch

### 3.2 方案

**A. 真 MTP 接线（主路径）**

```
history H
loop:
  draft = MtpDraft.propose(H, k)     # 优先：权重内 MTP 头；若无则「浅草稿」降级
  logits_seq = Target.verify_forward(H, draft)  # 一次前向覆盖 |H|+|draft| 或逐步等价
  accept, next = prefix_match(draft, greedy(logits_seq))
  emit draft[0..accept) + next
  H += emitted
```

降级策略（权重无 MTP 头时，ARCH O2）：

| 模式 | 行为 | 配置 |
|---|---|---|
| `mtp: true` + 有 MTP 权重 | 真 MTP 头草稿 | 默认目标 |
| `mtp: auto` + 无头 | **自回归草稿禁用**，退回 greedy 逐步；日志警告 | 不假装加速 |
| `mtp: false` | 永不投机 | 对照基线 |

> 不允许用「随机 MockDraft」上线充数；Mock 仅留在 `tests/unit`。

**B. Verify 与 Cache**

- Prefill：prompt 一次写入 KV / linear state  
- Verify：对 draft 序列做因果前向；**拒绝后缀不得污染** 已提交 cache（reject 时回滚 seq/linear 或 verify 用影子状态）  
- 推荐：**影子 cache 克隆提交点**，accept 后 `commit()`，reject 丢弃影子（实现简单、正确优先）

**C. 热路径（非 MTP 也做）**

- BF16 `Qwen35Model` 对齐 INT4：`thread_local` scratch  
- `lm_head`：大 M 分块 + 可选采样 top-k 截断接口（配置开关，默认全词表保证正确）  
- OpenMP：禁止嵌套；`M < 64` 串行（已有）

### 3.3 接口草图

```text
ICausalLM 增补（可选能力探测）:
  bool has_mtp() const;
  void draft_propose(history, k, out_tokens);   // 无则 Generator 禁用投机
  // verify 仍走 forward；由 Generator 控制 cache 提交

Generator:
  if (cfg.mtp && model->has_mtp()) SpecPath else GreedyPath
```

### 3.4 风险

| 风险 | 缓解 |
|---|---|
| 无 MTP 权重 | auto 降级，不对用户虚报 tok/s |
| cache 回滚错误 | 影子提交 + 单测强制 reject 场景 |
| α 过低变慢 | `accept_min` / 动态关投机 |

---

## 4. P1 设计 — LayerSpec + GraphExecutor

### 4.1 问题

每架构一个 C++ 类 → 换模型必写代码；与「配方 + 通用核」路线相反。

### 4.2 核心抽象

```text
ModelSpec
  meta: hidden, vocab, rms_eps, rope, tie_embed, ...
  layers: LayerSpec[L]
  weight_prefix: "language_model." | ""
  mapping: logical_name → LWC/QLWC tensor name

LayerSpec
  type: full_attention | linear_attention | moe_sparse | mlp_only
  attn?: { n_heads, n_kv, head_dim, q_gate: bool, partial_rotary, ... }
  linear?: { n_k, n_v, dk, dv, conv_k }
  moe?: { n_experts, top_k, ... }
  mlp: { intermediate, act: silu }
  norms: { input, post, q_norm?, k_norm?, one_plus_weight: bool }
  weights: { q,k,v,o, gate_up_down, ... 逻辑键 }
```

**GraphExecutor**

- `load(ModelSpec, WeightStore)`  
- `forward(tokens, SessionCache, logits)`：按 `layers[]` 调 **唯一** Op 实现  
- Op 集合（与架构无关）：`gemm_*` / `rmsnorm` / `rope` / `attn_*` / `gated_delta_*` / `moe_route+experts` / `embed` / `lm_head`

### 4.3 迁移策略

1. 从现有 `Qwen35Model` **提取** 为 `models/recipes/qwen3_5.json`（或 YAML）+ 薄适配  
2. 旧类改为对 Executor 的调用包装（过渡期），或删除前向改走 Executor  
3. 第二模型：`qwen2` / `llama` 稠密 GQA 配方（无 Gated DeltaNet）验证零新内核  

### 4.4 权重与量化

- BF16：`WeightManager` / LWC  
- INT4：`QlwcStore`；Spec 标注 tensor `dtype: bf16|int4`  
- Executor 按 dtype 选 `gemm_bias_free` / `gemm_int4`  

### 4.5 风险

| 风险 | 缓解 |
|---|---|
| Spec 表达力不够盖住 Qwen3.5 hybrid | LayerSpec.type 联合字段；未知 type → 加载失败而非静默错 |
| 性能回退 | P0 优化保留在 Op 层；Executor 禁止层间堆分配 |

---

## 5. P2 设计 — Paged KV + Continuous Batching

### 5.1 问题

- KV 按 `max_seq` 连续分配，长上下文与多会话内存差  
- Scheduler 实质单飞；radix 只存长度指纹，不复用张量  

### 5.2 Paged KV（CPU 简化）

```text
BlockSize = 16 (可配)
PhysicalBlockPool: 固定块数，每块存 K/V 或 linear 状态快照句柄
BlockTable[layer][logical_block] → physical_block_id
```

- Full attention：标准 paged K/V  
- Linear / DeltaNet：**按 token 递推**，分页存的是「可 fork 的 state 检查点」或仍按 session 独占 state（首版：linear state **每请求独占**，仅 full-attn KV 分页——降低复杂度）  

### 5.3 Continuous Batching

```text
RunningQueue: 若干 SequenceGroup
每步:
  收集 decode 中的 seq（各 1 query）
  可选：合并 prefill chunk
  一次或多次 GraphExecutor.forward_batch
  采样 → 结束者移出，槽位给 waiting
```

首版约束（可用）：

- CPU 上 batch ≤ 4（配置）  
- 同模型同 Spec  
- 不做 chunked prefill 高级特性（可列 P2.1）  

### 5.4 Radix 真复用

- 前缀命中 → 复制/共享 **已填满的 KV block**（引用计数）  
- 与 Paged 一体；废弃「只存 int 长度」的假复用作为主路径  

### 5.5 风险

| 风险 | 缓解 |
|---|---|
| DeltaNet 难分页 | 首版独占 state；文档写明 |
| 批处理正确性 | 单测：两会话交错 decode 文本不变 |

---

## 6. P3 设计 — MoE 主航道

### 6.1 问题

架构已写专家 LRU + 双缓冲；缺真实大 MoE 业务闭环与对标。

### 6.2 机制（自研）

| 机制 | 本仓 |
|---|---|
| 专家粒度驻留 | `WeightManager` LRU + pin |
| gate 后预取 | `ExpertPrefetcher` 双槽 |
| 带宽自适应 | PlacementPlanner / q★ 类预算（可先规则版） |
| 层流式降级 | 整层冷路径（OOM 时） |
| MTP | P0 已接入，offload 下仍跑（α 可能降） |

### 6.3 对标

- 同机、同模型（或官方声明的最接近替代）、记录 tok/s / TTFT / 缺页率  
- 外部引擎仅作 **下界参照**，不是依赖库  

### 6.4 风险

| 风险 | 缓解 |
|---|---|
| 无权重 | P3 启动门槛：必须有可加载 MoE LWC |
| 对标不公 | 报告写明 GPU/CPU、量化、是否 MTP |

---

## 7. 阶段依赖与顺序

```
P0 ──────────────────────────────► 速度基线 + MTP
        │
        ▼
P1 ── Graph 化（P0 的 Op 优化不丢）──► 多模型
        │
        ▼
P2 ── 批调度 + paged KV ───────────► 服务化密度
        │
        ▼
P3 ── MoE offload 业务化 ──────────► 大模型主航道
```

禁止并行打乱：P1 依赖 P0 的 cache/MTP 语义稳定；P2 依赖 P1 Executor 的 batch 入口；P3 依赖 P1 MoE Op + P2 调度。

---

## 8. 配置面（设计级）

```yaml
# 示意，实现文档细化键名
model:
  path: models/Foo.lwc
  recipe: models/recipes/qwen3_5.json   # P1
  dtype: bf16 | int4
decode:
  mtp: auto | true | false
  spec_k: 3
  accept_min: 1.5
sched:
  max_batch: 4                          # P2
kv:
  page_block: 16                        # P2
  pool_gb: 8
tiers:
  dram_hot_gb: 32
```

---

## 9. 确认清单（请逐条回复是否同意）

1. P0：无 MTP 权重时 **降级逐步 decode**，不用 mock 草稿上线 — **同意？**  
2. P0：verify 用 **影子 cache 提交** — **同意？**  
3. P1：第二验证模型选 **标准 GQA Transformer（Qwen2 或 Llama 系稠密）** — **同意或点名？**  
4. P2：Linear/DeltaNet state **首版每请求独占**，仅 full-attn KV 分页 — **同意？**  
5. P3：启动门槛为 **真实 MoE LWC 到位** — **同意？**  
6. 编码顺序严格 **P0→P1→P2→P3**，每阶段验收后再开下一阶段 — **同意？**  

全部确认后，以 [IMPLEMENTATION_P0_P3.md](IMPLEMENTATION_P0_P3.md) 为施工单开始编码。
