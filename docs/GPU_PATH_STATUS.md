# FreeToken-Style Hybrid GPU — 阶段总结

## 现状 (commit bb07b03)

| 阶段 | tok/s (pure_decode, M=1, RTX4060 + i7-14650HX) |
|------|--------:|
| CPU baseline (engine_int4, OMP=8) | 7.88 |
| hybrid_gpu @ commit 4cd835f (LRU) | 13.11 |
| hybrid_gpu @ commit **bb07b03** (lm W16 + 7GiB) | **19.77** |

**已提升 2.5×**。

## 已实现的优化 (本会话新增 commits)

- `4cd835f` INT4 VRAM cache LRU (Task 3)
- `c6cfa72` INT4 lm_head GPU dispatch + bench 暖机
- `bb07b03` BF16/tied lm_head → prefetch_w16 + 7GiB budget

## 实测 profile 分解 (hybrid_gpu 4B)

| 阶段 | 时间 | 来源 |
|------|------|------|
| linear_attn GEMV (4×/layer × 27) | ~28 ms | 7× GEMV launches + H2D/D2H 同步 |
| full_attn GEMV (4×/layer × 5) | ~8 ms | 同 |
| lm_head W16 GEMV on GPU | ~10.7 ms | cublas SGEMM 1.27GB fp32 |
| gated_delta_recurrent CPU | ~5 ms | AVX2 per-head 串行 |
| conv1d + residual + rmsnorm | ~10 ms | CPU |
| 其他 (attn prefill, MTP) | ~10 ms | |
| **合计 per token** | **~50 ms** | 19.8 tok/s |

## 剩余到 30 tok/s 还需要 (~33ms per token)

按影响力排序：
1. **Fused 多 GEMV 一次发射**: 4 small (wqkv+wz+wb+wa) → 1 launch + 1 D2H  (节省 ~12ms)
2. **gate/up 2→1 fused**: (节省 ~3-5ms)
3. **gated_delta_recurrent 上 GPU**: 写一个 NVRTC kernel 按 head 并行
4. **conv1d 上 GPU**
5. **CUDA Graphs**: 整 forward capture (需要所有指针稳定，工程量大)
6. **lm_head INT4 path**: 若 qlwc 把 lm_head 量化存储 (regenerate file) → 5ms

完成 1-4 → ~70ms 节省 → ~21 tok/s；再加 5 (graphs) → ~30 tok/s；lm_head INT4 → ~45 tok/s.

## 关键工程障碍

- **4B 太小**: L2 cache 已捕获大部分 GEMV，GPU 优势消失。27B-A3B、35B-A3B 这种 active-param
  小 + 总权重远超 L2 的模型，GPU 才真正有效。
- **kernel 调优成本**: 自写 GEMV 在 M=1 大 N (lm_head 这种) 很难超过 cuBLAS SGEMM 10.7ms (80% peak)。
- **CUDA Graphs 要求**: host 所有 op 序列确定、device pointers 稳定。当前 model code GEMV +
  CPU 计算交错，graph 化需大幅重构（每 step GPU kernel 之间无 CPU 介入）。

## Server 验证

server (`llmoc_server_int4`) 在 hybrid_gpu 模式下：
1. 自动 cuda::enable(gpu_vram_gb * 1 GiB)
2. model->warm_gpu_int4_weights() 上传所有 layer + lm 到 VRAM

bench tool (`bench_decode_tps`) 已同步加 enable + warmup 逻辑。两边数字一致。

## MoE 路径 (Task 4)

`src/exec/gpu/` 已有 ExpertSlotPool + GemmGpu::gemm_w16 走 cublas。
MoE INT4 GEMV 走 try_gemm_int4 → 8 行/block kernel。LRU 已支持 budget 上限淘汰。
未来要 ≥135B MoE：需 router-aware prefetch (last-layer + score-pred)，非本会话 scope。
