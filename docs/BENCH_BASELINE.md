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

## P0 验收对照

| 项 | 状态 |
|---|---|
| MTP / cache / bench 等 P0 接线 | ✅（见历史） |
| **纯 decode ≥10** | ✅ |
| **中长聊 chat_e2e ≥10** | ✅（中位 10.32 @ new=256） |
| 短聊 e2e ≥10 | ❌ 未达（~4.9，已分列记录） |
