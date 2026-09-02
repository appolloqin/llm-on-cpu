# Baseline (2026-09-02, before FreeToken GPU work)

## Hardware / toolchain
- GPU: NVIDIA GeForce RTX 4060 (Ada, sm_89), VRAM 8188 MiB (~8 GB), 显存带宽 272 GB/s
- Driver 566.24, CUDA 12.7
- CUDA toolkit v12.5 已安装: cudart64_12.dll, cublas64_12.dll, nvrtc64_120_0.dll, nvcuda.dll(System32) — NVRTC JIT 条件齐备
- RAM: 32 GB; 模型: Qwen3.5-4B INT4 (qlwc 3.12 GiB) / BF16 (lwc 8.68 GiB)

## 正确性（Task 0.1 结论）
- 生成空白问题真相：此前测试连接的是**残留旧服务器进程**（内存中为修复前模型）。修复后的 `Qwen3.5-4B.lwc` / `.int4.qlwc` + 当前二进制输出完全正常：
  - INT4 (port 15085, pure_cpu): "What is 2+2?" → "The answer is **4**. In standard arithmetic, adding 2 and 2 together results in 4. $$2 + 2 = 4$$ ..."
  - INT4 中文: "你好，请用一句话介绍你自己。" → "我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的语言理解、逻辑推理及多模态分析能力..."
  - BF16 (port 15085, pure_cpu): "What is 2+2?" → "The answer is **4**."
- 结论：权重修复 + 模板 + 前向均正确；无正确性障碍。

## 速度基线（tools/toks_bench.mjs, greedy, 128 tokens, 3 runs median）

| 模型 | 模式 | TTFT | decode tok/s |
|---|---|---|---|
| Qwen3.5-4B INT4 | pure_cpu | ~2.2 s | **7.88** |

（hybrid_gpu / pure_gpu 基线因当前 FP32-dequant 缓存路径几乎必然退化，待 Chunk 1 完成后一并测量。）

## 目标
- Qwen3.5-4B INT4 pure_gpu / hybrid_gpu: decode ≥ 30 tok/s（RTX 4060 272 GB/s 下，2.2GB INT4 全量上卡理论上限 ~120 tok/s，30+ 现实可达）
