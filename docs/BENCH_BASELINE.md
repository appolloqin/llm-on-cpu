# P0 性能基线

## 口径（写死）

| 指标 | 定义 | 说明 |
|---|---|---|
| **chat_e2e_tps / e2e_tps / last_tps** | `completion_tokens / generate墙钟`（含 prefill） | **对话验收以此为准**；与 `llmoc_last_tps` 同公式 |
| **pure_decode** | prefill 后只计逐步 `forward`（argmax 在计时外） | 实验室 decode 上限；**不以 e2e 冒充** |

复测模板（**中位数 ×3 次**，warm≥1；中长回复摊薄 prefill）：

```powershell
.\build\msvc-x64\bin\bench_decode_tps.exe --config configs\engine_int4.yaml --new 256 --warm 1
# ×3，取 e2e_tps 中位数；同时记 pure_decode
# 短聊对照：加 --short --new 32
.\build\msvc-x64\bin\int4_gemm_bench.exe
```

## 记录表

| 日期 | 机器 | 配置 | warm | new | e2e_tps | pure_decode | 备注 |
|---|---|---|---|---|---|---|---|
| 2026-08-29 | i7-14650HX | 优化前 | 1 | ~16 | ~2.1 | — | 早期 e2e |
| 2026-08-29 | i7-14650HX | scales/LayerPack/GDN | 1 | 32 | 2.46 | ~7.3 | 旧 pure（含 argmax） |
| 2026-08-29 | i7-14650HX | pure_decode≥10 方案 | 1 | 32 | ~3.7 | **10.74** | 短 EOS；pure 达标 |
| 2026-08-29 | i7-14650HX | **+batched GEMM prefill + last-only lm_head** | 1 | **256** | **10.32** | **~11.0** | 三次 e2e：10.35 / 10.29 / 10.32；pure≈11.07/10.85/11.01 |
| 2026-08-29 | i7-14650HX | 同上短聊 | 1 | 32 | **~4.85** | ~10.9 | `--short` completion=9；GDN 串行下界，**不**宣称短聊=10 |
| 2026-08-29 | i7-14650HX | 同上中长 128 | 1 | 128 | ~9.85 | ~11 | 略低于 10；验收用 256 |

## 内核优化摘要

| 项 | 改动 | 影响面 |
|---|---|---|
| 测量 | pure 只计 forward；greedy sample OpenMP | bench / generate |
| GDN | conv_k=4；dk=dv=128 快路径 | cpu_ops / INT4 model |
| INT4 | 4-row + **`gemm_int4_batch`**（权重行复用 n token） | int4_ops |
| Prefill | layer 内 QKV/MLP/out batch；`forward()` 只算最后 lm_head | qwen3_5_int4_model |
| Bench | 默认长 prompt + `--new 256`；`--short` 短聊；与 last_tps 同口径 | bench_decode_tps |

## 诚实结论（2026-08-29）

| 口径 | 数值 | 含义 |
|---|---|---|
| **中长聊 e2e**（验收） | **中位 ≥10**（`--new 256` → **10.32**） | 对话墙钟吞吐达标 |
| 短聊 e2e（hi→一句） | **~4.9** | prefill+GDN 串行；物理上难稳 ≥10 |
| pure_decode | **~11** | 流式出字间隔参考；保持不回退 |
| GEMM-only 微基准 | 非 e2e | 忽略 GDN/整网 |

MTP 保持 `false`。短聊再冲 e2e 需 SPR+AMX 或更强 GDN，不靠口径游戏。

## 2026-08-31 提速（pure_cpu Qwen INT4，HX 开发机）

| 项 | 说明 |
|---|---|
| OpenMP | 未设 `OMP_NUM_THREADS` 时 server/bench 默认 cap **24→8**（勿用 12） |
| lm_head | greedy decode 走 `gemm_int4_argmax`，免写全词表 + 二次 argmax |
| attn | `attn_decode_one` 复用 thread_local scores 缓冲 |
|  profiling | `LLMOC_PROFILE=1` 在 `forward(n=1)` 打印 linear/full/lm_head ms |
| 未实现缺口 | 见 [`IMPLEMENTATION_GAP_2026-08-31.md`](IMPLEMENTATION_GAP_2026-08-31.md) |

**HX 笔记本现象（2026-08-31 实测）**：长跑 decode 会在 **~200ms（≈5 t/s）** 与 **~100ms（≈10 t/s）** 间交替，均速 **6–8 t/s**；主因散热/频率节流，不是 OMP 单独能解。验收仍用 **bench 暖机后 64–256 token 窗口** 看中位数。**勿用 `OMP_NUM_THREADS=32`**（`start_int4.cmd` 旧版曾默认 32）。

### MTP（2026-08-31，Qwen3.5-4B INT4 @ HX）

| 模式 | decode t/s | mtp_alpha | 结论 |
|---|---|---|---|
| greedy（`mtp:false`） | **~6.8–7.1** | — | **CPU 默认** |
| MTP `spec_k=3`（`configs/engine_int4_mtp.yaml`） | **~4.9–7.5** | **~0.28**（29/105） | 草稿+verify 步 ~500ms，慢于 greedy |

- 权重：`has_mtp=1`（qlwc 含 MTP 头）
- 配置：`model.mtp: auto` → INT4 CPU **自动 greedy**；强制 MTP：`mtp: true` 或 `LLMOC_MTP=1`
- 优化：`pin_first` 锚定 greedy 首 token，去掉 draft[0] 冗余校验
- **≥30 tok/s** 仍依赖 SPR+AMX+MTP 热路径优化，非当前 HX INT4

```powershell
$env:OMP_NUM_THREADS='8'
.\build\msvc-x64\bin\bench_decode_tps.exe --config configs\engine_int4_mtp.yaml --new 64 --warm 0
```

## P0 验收对照

| 项 | 状态 |
|---|---|
| MTP / cache / bench 等 P0 接线 | ✅（见历史） |
| **纯 decode ≥10** | ✅ |
| **中长聊 chat_e2e ≥10** | ✅（中位 10.32 @ new=256） |
| 短聊 e2e ≥10 | ❌ 未达（~4.9，已分列记录） |
