# Layer Stream 模式（跑起来优先）

> 版本: v0.1 · 2026-09-01  
> 状态: **已实现**（QLWC lazy + INT4 forward 接线 + BF16 WM stream_dense + auto 预算切换；CUDA 可选 H2D 镜像）  
> 优先级: **可运行 > 吞吐 SLO**（与常驻路径的 ≥30 tok/s 门禁拆分）

---

## 1. 产品策略

```
工作集能否常驻（DRAM 和/或 VRAM）？
  ├─ 是 → pure_cpu | hybrid_gpu | pure_gpu   （现方案，冲吞吐）
  └─ 否 → layer_stream                       （本模式，先保证出 token）
```

| 模式 | 目标 | 性能门禁 |
|---|---|---|
| 常驻三模 | 快 | 按 IMPLEMENTATION §10 / 机型谈 tok/s |
| **layer_stream** | **能跑** | 固定 prompt 生成 N token、不 OOM；**不承诺** 30 tok/s |

对标 AirLLM：**按层（或层窗口）装入 → 算 → 释放 → 下一层**；双缓冲预取下一层以叠 IO。

---

## 2. 何时进入

**显式**：`mode: layer_stream`（配置 / API）。

**自动**：`mode: auto` 且落到 CPU、权重文件大小 `> dram_hot_gb`（`layer_stream.auto: true`，默认开）→ 切 `layer_stream` 并打 WARN。  
有 CUDA 时 `auto` 仍可优先 hybrid/pure_gpu；仅 CPU 路径且超预算才分层。

---

## 3. 数据面（接口）

```
ILayerStreamLoader:
  open(lwc|qlwc path, LayerStreamConfig)
  // 常驻：embed / final_norm /（可选）lm_head；其余按层窗口
  pin_layer(layer_id) -> LayerView     # 同步装入；若已在窗口则命中
  prefetch_layer(layer_id)             # 异步预取到备用槽
  release_layer(layer_id)              # 窗口外可丢（保留 pin 的常驻）
  window_bytes() / stats()             # 当前窗口占用、miss、IO 等待
```

**LayerStreamConfig（建议默认）**

| 项 | 默认 | 说明 |
|---|---|---|
| `window_layers` | 2 | 当前层 + 预取层（双缓冲） |
| `device` | `cpu` 或 `cuda:0` | 计算设备；权重可 DRAM↔VRAM |
| `resident` | embed+norm(+lm_head) | 永不换出 |
| `io_workers` | 2 | 与现 `WeightManager` IO 池对齐 |

**与现有模块关系**

- 复用 `lwc`/`qlwc` + `IoEngine`（O_DIRECT/线程池）；不必新容器格式。  
- `WeightManager` 今日是「张量级 LRU」；layer_stream 要 **层粒度 pin/release API**（可包一层 `LayerStreamWeightSource`）。  
- MoE：一层内仍可只拉 top-k 专家（层窗口 ∩ 专家稀疏）。

---

## 4. Forward 编排（目标形态）

```
prefill / decode step:
  for L in 0..n_layers-1:
    prefetch_layer(L+1)          # 与计算重叠
    view = pin_layer(L)
    layer_forward(L, view, …)
    release_layer(L-window+1)    # 滑窗
  lm_head(resident or pin once)
```

- **CPU**：`device=cpu`，窗口在 DRAM。  
- **GPU**：层权重 H2D → 算 → 可丢 VRAM 槽；激活可留 GPU。小显存时窗口=1～2 层。  
- KV / GDN 状态：**不随层权重丢弃**（与 AirLLM 相同：状态常驻，权重流式）。

---

## 5. 门禁（可运行优先）

### 5.1 正确性（P0）

| ID | 门禁 |
|---|---|
| LS-C1 | `mode=layer_stream` 可解析；未知模式不静默当成 pure_cpu |
| LS-C2 | 假权重 / 小模型：layer_stream 与 pure_cpu 同 prompt greedy token 一致（允许浮点噪声阈值另定） |
| LS-C3 | 窗口=1 与窗口=2 输出一致 |

### 5.2 可运行（P0）

| ID | 门禁 |
|---|---|---|
| LS-R1 | 人为限制 `window` 预算 < 全模 50% 时仍能生成 ≥32 token，进程 RSS/VRAM 不超过配置帽 |
| LS-R2 | 缺层文件 / IO 失败 → 明确错误，不半截胡话 |

### 5.3 性能（P1，不挡「能跑」）

| ID | 门禁 |
|---|---|
| LS-P1 | 记录 `io_wait_ms` / `compute_ms`；双缓冲相对同步装层有加速或持平 |
| LS-P2 | **不**与常驻路径比 ≥30 tok/s；可选对照 AirLLM 同模同机「能对话」 |

---

## 6. 实现分期

| 阶段 | 交付 | 状态 |
|---|---|---|
| **S0–S4** | 模式 + QLWC lazy loader + INT4/BF16 server + CUDA 可选 H2D + auto 预算切换 | ✅ |

---

## 7. 配置草案

```yaml
mode: layer_stream   # 显式；与 pure_cpu|hybrid_gpu|pure_gpu|auto 并列
layer_stream:
  window_layers: 2
  device: cpu        # cpu | cuda:0
  # 预算帽（可运行门禁）；0 = 不限制（仅开发）
  max_window_mb: 2048
```

`layer_stream.*` 已由 `EngineConfig` 解析（见 `configs/engine_int4.yaml`）。

---

## 8. 非目标（本模式）

- 不承诺普通 CPU / 小显存上达到常驻路径的 tok/s。  
- 不替代 hybrid/pure_gpu 的加速实现（GPU 核仍按 GAP 推进）。  
- 不把 HF AWQ 仓直接当层流式源（仍经 LWC/QLWC）。
