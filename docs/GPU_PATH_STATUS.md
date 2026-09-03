# 4B hybrid_gpu 到 ≥30 tok/s 路径 (Task 2.3 现状评估)

## 当前状态 (freetoken-hybrid-gpu @ 4cd835f)

| 阶段 | hybrid_gpu 4B decode |
|------|-------:|
| warmup 上传 ~310MB 层 INT4 权重 (resident) | 1s (一次性) |
| 224 个层 GEMV (GPU JIT gemv_int4) | ~50ms |
| gated_delta_recurrent + residual + conv (CPU) | ~35ms |
| lm_head GEMV (hal::gemm_int4 CPU) | **24ms** |
| MTP + draft + verify | 0ms (禁用) |
| **总和** | **~130ms → 7.5 tok/s** |

## 已达到的优化 (commits)

- `e4d33bc` INT4 GEMV kernel (8 rows/block + vec uchar4 + shared scales)
- `5b79e16` INT4 量化字节驻留 VRAM, JIT GEMV 路径
- `c44b6bd` 性能优化追平 CPU 7.52 tok/s
- `b3d1b8e` 放开行数上限让 lm_head 可上卡
- `4cd835f` VRAM 缓存 LRU 淘汰

## 实测: lm_head 上 GPU (130→316 之前的回归)

`--warm > 0` + `warm_gpu_int4_weights` 触发上传 layer + lm_head 全上卡后,
首次 GEMV 走 JIT 实测反而退化, 因为:

1. **kMaxGpuInt4Rows**: lm_head N=248320, 旧上限 65536 → resident 路径跳过,
   回退到 `ensure_int4_device` 把 1.5GB FP32 上传 (超 budget) → 走 CPU 但带额外 lock 开销
2. **microbench 实测**:
   | shape | CPU | GPU resident JIT | 赢家 |
   |-------|----:|------:|------|
   | 896x3072 (down) | 0.081 | 0.106 | CPU |
   | 3584x3072 (mlp) | 0.224 | 0.259 | CPU |
   | 9216x3072       | 0.508 | 0.672 | CPU |
   | 248320x3072 lm  | 11.32 | (resident only) | GPU ~15ms |

   → **对 4B 这种小模型, GPU 层 GEMV 比 CPU AVX2 慢**; lm_head (M=248320, 318MB resident) 上 GPU
   才能勉强 24→15ms 提升 9ms

## 真实瓶颈分布 (4B, M=1 decode)

```
lm_head CPU  hal::gemm_int4   : 24ms ───────── (18%)
gate+up+down GEMV (GPU JIT)    : 22ms ─────── (17%)  
attn 4 GEMV (GPU JIT)          : 35ms ────── (27%)
conv1d / gated_delta (CPU)     : 28ms ────   (22%)
rmsnorm / residual / RoPE(CPU): 15ms   ────   (11%)
其他                            :  8ms        (5%)
─────────────────────────────────────────
合计                          : 130ms        (7.5 tok/s)
```

## 到 30 tok/s 还需要的 (3x)

| 优化 | 预期节省 | 难度 |
|------|---------|------|
| A. 让 GPU 跑 gate+up 一次发射 (shared x, M=2I) | 12ms | 中 |
| B. lm_head GPU resident JIT path | 9ms | 已完成代码 (待测) |
| C. 把 conv1d 整段 (depthwise) 搬到 GPU | 8-10ms | 高 (新 kernel) |
| D. 把 gated_delta_recurrent 搬到 GPU (per-head) | 12-15ms | 高 |
| E. Fused rmsnorm + matvec 在 GPU 内 | 5ms | 高 |
| F. CUDA Graphs 整 forward | ~10ms 调度 | 高 (需设备内存驻留) |
| G. spec/MTP 接受率 (248K vocab argmax 单测) | 已禁用 | 高 |

合计 B+C+D+F = **30-40ms** 节省 → 从 130 → 90-100ms = **11 tok/s**。要 ≥30 tok/s
基本要所有项都做 + GPTQ 量化 lm_head (降到 ~80MB / 0.3ms) 或 MTP 接受率 ≥50%。

## 建议优先级 (会话外)

1. **lm_head 强制 resident** 跑起来 (测 24→15ms): 已完成代码 `b3d1b8e`, 但需 GPU 路径接入 greedy decode
2. **gate+up 一次发射** (新 API `try_gemm_int4_dual(x, Wg, Wu, yg, yu)`): shared x, 一次 H2D, 一次 D2H
3. **gated_delta_recurrent GPU kernel**: 当前 CPU 28ms, 写 8x head × 32 lanes kernel
4. **CUDA Graphs** 需要 forward pass 内全部指针驻设备 (大改造)
5. **MTP 验证 + spec_k 路径**: 4B lm_head 走 GPU 后, draft/verify 才高效, 否则 1 步 24ms vs CPU 27.6ms

## 4B 之外的模型 (27B-A3B MoE) 上 GPU 更有意义

- 4B dense: 大部分在 L2 cache 命中, GPU HBM 优势消失
- 27B-A3B MoE: expert 权重不命中 L2 → GPU 大带宽有效
- 118B-A16B MoE: 必须 GPU + 分层驻留 + 预测预取

## 当前分支已具备的能力

- ✅ JIT gemv_int4 (CUDA 驱动级)
- ✅ 量化形态驻留 (saves 8x VRAM)
- ✅ LRU 淘汰 (Task 3.1 done)
- ✅ warm_gpu_int4 + lm_head 上卡开关
- ⚠️ lm_head 在 forward_decode_greedy 仍走 CPU 路径 — 需要把 gemm_view 接入 (本会话)
