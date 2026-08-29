# 实现文档：四阶段（P0→P3）

> 版本: v1.0（**已确认** 2026-08-29 — 开始按 PR1=P0 编码）  
> 状态: 🟢 **实现方案已确认**  
> 对应设计: [DESIGN_P0_P3.md](DESIGN_P0_P3.md)  
> 对应架构: [ARCHITECTURE.md](ARCHITECTURE.md) M2–M4 + MoE 主航道  

确认选项（已锁定）：无 MTP→greedy；快照 rollback；第二模型 Qwen2/Llama 稠密；P2 linear 独占；P3 需真实 MoE LWC；P0→P1→P2→P3；P2 首版串行 forward。

---

## 0. 公共约定

| 项 | 约定 |
|---|---|
| 语言 | C++20 热路径；Python/Node 仅工具 |
| 分支策略 | 每阶段独立 PR：`feat/p0-mtp` → `feat/p1-graph` → … |
| 兼容 | P0/P1 不破坏现有 `llmoc_server` / `_int4` 可启动；旧路径可 `#ifdef` 或 runtime flag 过渡一期 |
| 禁止 | 引入「转发第三方推理服务」作为默认；MockDraft 进生产 generate |

---

## 1. P0 实现 — MTP 接入 + 热路径

### 1.1 目录与文件

| 动作 | 路径 |
|---|---|
| 改 | `src/model/generate.h/.cpp` — Spec 路径 / Greedy 路径 |
| 改 | `src/model/causal_lm.h` — `has_mtp` / 可选 draft API |
| 改 | `src/model/qwen3_5_model.*`、`qwen3_5_int4_model.*` — scratch 对齐；MTP 探测 |
| 改 | `src/model/spec_decode.*` — 增加 `CacheTransaction` 或 commit API 文档化用法 |
| 增 | `src/model/mtp_draft.h/.cpp` — 真权重草稿（无则编译桩 + runtime false） |
| 增 | `tools/bench_decode_tps/` — 暖机测 tok/s |
| 改 | `configs/engine.yaml`、`engine_int4.yaml` — `decode.mtp` |
| 改 | `tests/unit/test_spec_*.cpp` — 影子 cache reject 用例 |

### 1.2 关键实现步骤

1. **CacheTransaction**（可放 `kv_cache.h`）  
   - `begin()` 快照 `seq` + linear 标志位（深拷贝 state 若体积可接受；4B linear state 需评估，过大则 copy-on-write 或 verify 不写正式 cache直到 accept）  
   - **推荐首版**：verify 阶段 draft token **先写正式 cache**，reject 时 `rollback(seq0)` + 重置 linear 到快照（快照必做）  
2. **Generator::generate**  
   ```
   if (enable_mtp && model->has_mtp())
     loop: propose → forward drafts on txn → accept → commit/rollback
   else
     现有逐步采样
   ```  
3. **配置** `decode.mtp: false|true|auto`（`engine_config` 已有 `mtp` 字符串，复用并文档化）  
4. **bench_decode_tps**：固定 prompt、warm 1 次、测 N 个新 token，打印 tps / TTFT  
5. **BF16 scratch**：移植 INT4 `Int4Scratch` 模式到 `qwen3_5_model.cpp`

### 1.3 无 MTP 头时的行为

```cpp
bool Qwen35Model::has_mtp() const {
  return wm_->header().find(prefix_ + "mtp...") != nullptr; // 实际键名以权重为准
}
```

- `auto`：无头 → `has_mtp()==false` → greedy  
- 日志：`I mtp disabled: weights missing`

### 1.4 验收

```text
# 单元
llmoc_unit_tests  # 含 reject/rollback、原 spec mock 仍过

# 性能（本机记基线）
bench_decode_tps --config configs/engine_int4.yaml --new 64 --warm 1
# 期望：相对文档记录的「优化前暖机 ~2 t/s」有 ≥3×；有 MTP 时 tokens_per_step 写入 metrics
```

### 1.5 工时量级（参考）

3–7 人日（含 rollback 正确性与 BF16 scratch）。

---

## 2. P1 实现 — LayerSpec + GraphExecutor

### 2.1 目录与文件

```
src/model/graph/
  model_spec.h          # ModelSpec / LayerSpec POD + JSON 加载
  model_spec.cpp
  graph_executor.h
  graph_executor.cpp    # forward 主循环
  ops_dispatch.h        # 薄封装调 hal::
models/recipes/
  qwen3_5.json          # 从现网 config 生成
  qwen2_dense.json      # 第二模型（名称以确认清单为准）
tools/
  export_recipe.py/.mjs # HF config.json → recipe
```

### 2.2 LayerSpec JSON 示意

```json
{
  "name": "qwen3_5",
  "hidden_size": 2560,
  "vocab_size": 248320,
  "rms_norm_eps": 1e-6,
  "rms_one_plus_weight": true,
  "rope": { "theta": 10000000, "partial_rotary_factor": 0.25 },
  "tie_word_embeddings": true,
  "weight_prefix": "language_model.",
  "layers": [
    {
      "type": "linear_attention",
      "linear": {
        "num_key_heads": 16, "num_value_heads": 32,
        "key_head_dim": 128, "value_head_dim": 128,
        "conv_kernel": 4
      },
      "mlp": { "intermediate_size": 9216, "act": "silu" }
    },
    {
      "type": "full_attention",
      "attn": {
        "num_heads": 16, "num_kv_heads": 4, "head_dim": 256,
        "q_gate": true
      },
      "mlp": { "intermediate_size": 9216, "act": "silu" }
    }
  ]
}
```

逻辑权重键在 executor 内拼：`layers.{i}.self_attn.q_proj.weight` 等（与现 LWC 名一致）。

### 2.3 GraphExecutor API

```cpp
class GraphExecutor final : public ICausalLM {
 public:
  void load(ModelSpec spec, wt::WeightManager* wm /*或 QlwcStore*/, WDtype);
  const CausalLmMeta& meta() const override;
  void init_cache(...) const override;
  void forward(...) override;
  bool has_mtp() const override;
};
```

- `main.cpp`：若存在 `model.recipe` → `GraphExecutor`，否则旧 `Qwen35Model`（过渡一期）  
- 单测：同 prompt 旧路径 vs Executor logits cos > 0.999（BF16 小层或截断层数自测模型）

### 2.4 第二模型

- 转换：现有 `convert_lwc` 流程 + recipe  
- **不新增** `llama_model.cpp`；仅 `recipes/*.json` + 映射  

### 2.5 验收

```text
llmoc_server --config configs/engine_recipe_qwen35.yaml   # recipe 路径
# chat 冒烟
# 第二配方 configs/engine_recipe_qwen2.yaml 冒烟
# git：禁止为第二模型新增 src/model/*_model.cpp 前向实现文件
```

### 2.6 工时量级

7–14 人日。

---

## 3. P2 实现 — Paged KV + Continuous Batching

### 3.1 目录与文件

```
src/kv/
  block_pool.h/.cpp      # 物理块分配 / 引用计数
  block_table.h/.cpp     # 逻辑→物理
  paged_cache.h/.cpp     # 替换 SessionCache 主路径（或适配层）
src/sched/
  scheduler.cpp          # 多槽 running
  batcher.h/.cpp         # 组 batch、调度策略 FCFS
tests/unit/
  test_paged_kv.cpp
  test_batch_decode.cpp
```

### 3.2 BlockPool

- `block_size` 默认 16 tokens  
- `pool_bytes` 来自 `tiers.kv_pool_gb`  
- `alloc` / `free` / `add_ref` / `release`  
- OOM：拒绝新请求或抢占 idle 前缀（首版拒绝 + 503）

### 3.3 Scheduler 行为

```text
enqueue(req) → waiting
loop:
  fill running until max_batch
  build BatchInput (token ids per slot, cache handles)
  executor.forward_batch(batch)   # P1 需暴露；首版可串行 forward 模拟 batch 正确性，再合并核
  sample each
  finished → detach blocks (ref--)
```

**首版可串行执行各 seq 的 forward**（仍共享 BlockPool），先保证调度与生命周期正确；再做真 batched GEMM（列为 P2.1）。

### 3.4 Linear state

按 DESIGN：每请求 `LinearAttnState` 独占，挂在 `Sequence` 上，不进 BlockPool。

### 3.5 验收

```text
# 两并发 chat，结果各自连贯
# 长上下文 RSS 低于「连续 max_seq×会话数」旧公式
# radix 命中：第二次相同前缀 prefill 时间下降（metrics 计数 prefix_hits）
```

### 3.6 工时量级

10–18 人日（含 P2.1 batched GEMM 则更长；可拆第二 PR）。

---

## 4. P3 实现 — MoE 主航道

### 4.1 前提

- 仓库内存在可加载 MoE `*.lwc`（或文档指定路径）  
- P1 recipe `type: moe_sparse` 或现有 `MoeModel` 收编进 Executor  

### 4.2 任务

| # | 任务 |
|---|---|
| 1 | `MoeModel` / Graph MoE Op 与 `ExpertPrefetcher` 在 server 路径默认开启 |
| 2 | 缺页/预算耗尽：日志 + 降级整层流式，进程不崩 |
| 3 | metrics：`expert_miss_total`、`prefetch_hit_ratio`、`io_wait_ms` |
| 4 | `docs/BENCH_EXTERNAL.md` 对标表模板 + 一次实测填写 |
| 5 | MTP（P0）在 MoE 上可开关；缺省 auto |

### 4.3 验收

```text
llmoc_server --config configs/engine_moe.yaml
# chat 冒烟；杀磁盘延迟注入（若有 fake io）不崩溃
# BENCH 表：本仓 vs 外部对标引擎列齐
```

### 4.4 工时量级

依赖权重到位；工程 7–14 人日 + 对标 2–3 人日。

---

## 5. 配置键汇总（实现时写入 engine_config）

| 键 | 阶段 | 含义 |
|---|---|---|
| `decode.mtp` | P0 | `auto\|true\|false` |
| `decode.spec_k` | P0 | 草稿长度 |
| `decode.accept_min` | P0 | 低于则关投机（可选） |
| `model.recipe` | P1 | recipe JSON 路径 |
| `sched.max_batch` | P2 | 默认 1→4 |
| `kv.page_block` | P2 | 默认 16 |
| `tiers.kv_pool_gb` | P2 | 块池 |
| `tiers.dram_hot_gb` | P3 | 专家热区 |

---

## 6. 测试矩阵

| 阶段 | unit | 集成/冒烟 | perf |
|---|---|---|---|
| P0 | rollback、spec mock | INT4/BF16 chat | `bench_decode_tps` |
| P1 | recipe 加载、logits 对拍 | 两配方 server | 无回退超过 10% |
| P2 | block refcount、双会话 | 并发 chat | prefix_hit 延迟 |
| P3 | prefetch mock | MoE chat | vs 外部对标表 |

---

## 7. 编码门禁

**以下全部满足才开写代码：**

- [ ] DESIGN_P0_P3.md §9 确认清单得到明确回复  
- [ ] 本实现文档无未裁决歧义（第二模型名、是否 P2.1 合并进 P2）  
- [ ] 基线：记录一笔当前 `bench_decode_tps` 数字写入 `docs/BENCH_BASELINE.md`（编码 P0 当日可补测）  

确认后执行顺序：

```text
PR1 P0 → 验收通过
PR2 P1 → 验收通过
PR3 P2 → 验收通过
PR4 P3 → 验收通过
```

---

## 8. 待你确认的实现选项（与 DESIGN §9 对齐）

| # | 选项 | 默认建议 |
|---|---|---|
| I1 | P2 首版 batch 内 **串行 forward** 还是必须 **真 batched GEMM** | 串行先验收调度；batched 作 P2.1 |
| I2 | 第二配方模型 | Qwen2.5-3B/7B 稠密（有 LWC 转换优先） |
| I3 | MTP 无头时 | auto→greedy（DESIGN 已写） |
| I4 | P0 cache | 快照 rollback（DESIGN 已写） |
