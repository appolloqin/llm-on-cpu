# 实现缺口清单（相对 IMPLEMENTATION_2026-08-31）

> **索引**：设计正文见 [`IMPLEMENTATION_2026-08-31_三模独立-MoE激活GPU-分族KV.md`](IMPLEMENTATION_2026-08-31_三模独立-MoE激活GPU-分族KV.md)  
> **验收标准**：仅认该文 **§10 全部门禁**；本文 §12 的 `[x]` 为**设计冻结**，不代表代码已交付。  
> **更新**：2026-08-31（HX 开发机实测 INT4 ~10 tok/s；设计 SLO ≥30 未达）

---

## 1. 总览

| 类别 | 状态 | 说明 |
|---|---|---|
| G1 三模独立 `exec/*` | 🟡 骨架 | `exec/cpu|hybrid|gpu` + factory；热路径仍多走 `hal/` 直连 |
| G2 hybrid GPU attn / CPU 专家 | 🟡 部分 | GLM BF16 attn GPU；**hybrid 专家仍 CPU**；pure_gpu 专家可走 AWQ/NVFP4 dequant+cuBLAS |
| G3 pure_gpu 专家 VRAM 槽 | 🟡 部分 | Qwen INT4 + GLM AWQ/NVFP4：`try_gemm_*` dequant 路径；真 TC / `gemm_swiglu` 设备核未做 |
| G4 pure_cpu 零回归 | 🟢 | Qwen INT4/BF16 对话路径保持；注意 OMP 线程数 |
| G5 四族 pack + 正确 KV | 🟡 | Qwen/GLM 真图；**DS/Kimi = STUB-v0** latent+MoE（非论文级） |
| G6 配置 `devices` / mesh | 🟡 部分 | 解析 + planner；多卡运行未闭环 |
| G7 失败语义（不静默降级） | 🟡 部分 | pure_gpu 无 CUDA 会 fail；NCCL 缺失时 stub 行为待收紧 |
| G8 引擎内单机多卡 EP/TP | 🔴 未交付 | NCCL 动态加载有；**EP 数据面 = host stub** |

图例：🟢 可用　🟡 脚手架/部分　🔴 未实现或不可验收

---

## 2. 分族 forward / server

| 族 | Server | Forward 图 | KV / Cache | 量化 | 备注 |
|---|---|---|---|---|---|
| **Qwen3.8** | `llmoc_server` / `_int4` | 🟢 `Qwen35Model` / `Qwen35Int4Model` | 🟢 DeltaNet + GQA paged KV | BF16 LWC / INT4 QLWC | GPU：BF16/INT4 投影 dequant+cuBLAS |
| **GLM-5.3-Flash** | `llmoc_server_glm` | 🟡 `GlmFlashModel`（KDA+DSA+MoE） | 🟡 线性状态 + KPool SDPA | AWQ / NVFP4 GLMQ | pure_gpu：AWQ/NVFP4 可上 GPU；hybrid 专家 CPU |
| **DeepSeek-V4** | 🟡 `llmoc_server_ds` | 🟡 **DS-STUB-v0** latent+MoE | 🟡 latent 占位（非 CSA/HCA） | NVFP4 专家 + BF16 | fake `make_fake_ds`；勿当论文复现 |
| **Kimi-K3** | 🟡 `llmoc_server_kimi` | 🟡 **Kimi-STUB-v0**（同骨架） | 🟡 latent 占位 | NVFP4 | 单卡 pure_gpu → **自动 layer_stream**；推荐 hybrid |

**CI 假权重**：`make_fake_glmq`、`make_fake_ds`、`make_fake_kimi`、`make_fake_hf`。

---

## 3. §10.1 正确性门禁

| 门禁 | 状态 | 缺口 |
|---|---|---|
| `llmoc_unit_tests` 全绿 | 🟢 | 维持 |
| `LLMOC_EXEC_CPU_ONLY` 隔离编译 | 🔴 | 未加 CMake 选项 |
| Active-set planner PASS/FAIL | 🟡 | 有单测；与真权重未全联调 |
| Pure-gpu 专家槽 UAF/H2D | 🟡 | `test_exec_mesh` 等；未接真 GEMM |
| KV rollback（各族 snapshot） | 🟡 | Qwen MTP 路径有；DS/Kimi 无 |
| 分族 fake forward | 🟡 | Qwen/GLM 有；DS/Kimi 无 |
| **Mesh EP 2 卡 logits 对拍** | 🔴 | 无 `world_size=2` 验收单测 |
| **Mesh TP 2 卡对拍** | 🔴 | 未实现 |
| world_size>1 无 NCCL → 启动失败 | 🟡 | 需与 stub 行为对齐 §10.3 |
| Worst-case slots planner | 🟡 | 待补 |

---

## 4. §10.2 性能门禁

| 项 | 状态 | 备注 |
|---|---|---|
| pure_cpu INT4 Qwen4B @ HX | 🟡 ~10 tok/s | 设计 SLO ≥30；见 [`BENCH_BASELINE.md`](BENCH_BASELINE.md) |
| hybrid_gpu > pure_cpu（同量化） | 🟡 | Qwen/GLM 投影 GPU GEMM 已接；待同机 bench |
| pure_gpu > hybrid（GLM NVFP4 / DS） | 🟡 | GLM/DS stub pure_gpu 可走 NVFP4 dequant GPU；真专家槽核仍缺 |
| pure_gpu ×2 EP 吞吐 | 🔴 | 未测 |
| MTP 进热路径 | 🔴 | `generate` 有逻辑；默认关；未乘吞吐 |
| AMX → forward | 🔴 | `amx_gemm` 未接线 |
| **layer_stream（可运行优先）** | 🟢 | [`DESIGN_LAYER_STREAM.md`](DESIGN_LAYER_STREAM.md)；QLWC lazy+INT4 forward；BF16 `stream_dense_layers`；`auto` 超预算切换 |
| `BENCH_BASELINE.md` 2026-08-31 节 | 🔴 | 待 hybrid/gpu 实测后补 |

**当前提速方向（pure_cpu Qwen INT4，不阻塞上表）**：

- OpenMP 默认 cap（HX 上 8/16，勿 12/24）
- decode 热路径：复用缓冲、`gemm_int4_argmax`（greedy 免全词表写回）
- GDN / full-attn / lm_head 占比待 `LLMOC_PROFILE=1` on `forward(n=1)`

---

## 5. §11 文档同步（未做）

| 文件 | 状态 |
|---|---|
| `docs/ARCHITECTURE.md` §3.1 多卡指针 | 🔴 |
| `docs/MODEL_GLM53_FLASH.md` pure_gpu + devices | 🟡 部分 |
| `docs/IMPLEMENTATION.md` exec/families 树 | 🔴 |
| `README.md` 模式表 | 🔴 |
| `configs/engine_glm_*.yaml` devices 注释 | 🟡 |

---

## 6. 建议施工顺序（与 design §13 一致）

0. **layer_stream S1–S2**（可运行优先）：`LayerStreamWeightSource` + Qwen BF16/INT4 按层 forward；门禁 LS-R1  
1. **pure_cpu 提速**：Qwen INT4 达标路径（OMP、lm_head argmax、GDN/attn）  
2. `exec/gpu` 真核 + Qwen INT4 投影已走 `hal::cuda::try_gemm_int4`；后续 IExecBackend / 真 INT4 TC  
3. `mesh_nccl` EP 数据面 + §10.1 Mesh 单测  
4. `families/deepseek_v4`、`families/kimi_k3` forward + server  
5. AMX / MTP 接线 + §10.2 bench 回填  

---

## 7. 变更记录

| 日期 | 说明 |
|---|---|
| 2026-08-31 | 初版：对照代码库与 §10，区分设计冻结 vs 实现交付 |
| 2026-09-01 | 四族 GPU：AWQ/NVFP4 `try_gemm_*`；GLM/Qwen warm；`llmoc_server_ds`/`_kimi` STUB-v0 |
