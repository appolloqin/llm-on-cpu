# GLM-5.3-Flash（独立模块）

> 状态: 🚧 骨架已挂载（`src/glm/` + `llmoc_server_glm`）  
> 约束: **与 Qwen / DeepSeek 实现隔离**；不修改 `llmoc_server` / `llmoc_server_int4` 默认路径。

## 1. 目标模型

| 项 | 值 |
|---|---|
| 参考 | zai-org/GLM-5.3-Flash（~320B / ~18B active MoE） |
| 图 | KDA 线性注意力 + DeepSeek 稀疏注意力 + mHC + shared/routed expert |
| 量化 | **AWQ INT4** 与 **NVFP4** 并列（本模块自有容器，不复用 QLWC/`int4_ops`） |

**不要**直接把 `LibertAIDAI/GLM-5.3-Flash-NVFP4` 喂给旧 `convert_lwc`；用 `tools/glm/`。

## 2. 三种执行模式（与架构文档对齐）

| `mode` | 含义 | 64G+单卡~22G |
|---|---|---|
| `pure_cpu` | 权重与算子均在 CPU；专家盘预取 | 可跑；约数 tok/s 量级 |
| `hybrid_gpu` | 热层/注意力优先 GPU，冷专家 CPU/NVMe | 单卡 22G 的主推荐 |
| `pure_gpu` | 尽量全在 GPU（显存不足则启动失败或明确降级策略） | 22G 通常无法整模，需多卡或拒绝 |

配置：`configs/engine_glm_int4.yaml` / `configs/engine_glm_nvfp4.yaml` 中的 `mode:`。

## 3. 目录

```
src/glm/           运行时（模型图、HAL、权重、device）
tools/glm/         convert / awq quantize / nvfp4 import
configs/engine_glm_*.yaml
models/recipes/glm53_flash.json
bin/llmoc_server_glm
```

## 4. 启动

```bat
bin\llmoc_server_glm.exe --config configs\engine_glm_nvfp4.yaml   :: 默认
bin\llmoc_server_glm.exe --config configs\engine_glm_int4.yaml
```

日志应出现 `arch=glm mode=... quant=...`。

## 5. 内存（经验）

- 18B 激活 BF16 ≈ 36 GiB → 不要指望 32G 全 BF16 常驻。
- 4bit 专家 + 预取：32G/64G 才现实；单卡 22G 走 `hybrid_gpu`。

## 6. 实现进度

| 项 | 状态 |
|---|---|
| 独立 CMake target / server | ✅ |
| mode: pure_cpu / hybrid_gpu / pure_gpu | ✅ |
| AWQ / NVFP4 CPU GEMM 参考核 | ✅ |
| GLMQ v2 读写 + `make_fake_glmq` | ✅ |
| 最小 MoE forward（GQA 站位 + router top-k + shared expert） | ✅ 假模型可出 logits |
| KDA / MLA / mHC / KPool-DSA (SDPA 参考) | ✅ CPU；Ada/消费级走 KPool+SDPA（FlashMLA 需 SM90+） |
| HF convert / AWQ pack / NVFP4 import | ✅ |
| CUDA hybrid/pure_gpu 实算 | ✅ 动态加载 cudart+cublas；hybrid 预热注意力 BF16，专家仍 CPU |

> **说明**: RTX 40 系为 SM89，官方 FlashMLA 面向 SM90。本引擎在消费级卡上使用与配置一致的 **KPool 压缩索引 + SDPA**，在 hybrid/pure_gpu 下用 **cuBLAS** 加速 BF16 GEMM。

### 权重流水线

一键（推荐，与发布包 `download_glm.*` 相同；**默认 NVFP4**）：

```bat
download_glm.cmd              :: LibertAIDAI/GLM-5.3-Flash-NVFP4 → models/GLM-5.3-Flash.nvfp4.glmq
download_glm.cmd --awq        :: zai-org/GLM-5.3-Flash → AWQ → models/GLM-5.3-Flash.awq.glmq
:: 或: node tools\glm\prepare_glm.mjs --quant nvfp4 --prune-hf
```

分步：

```bat
:: BF16 HF → GLMQ
node tools\glm\convert_glm_lwc.mjs --src models\GLM-5.3-Flash-hf --out models\GLM-5.3-Flash.bf16.glmq

:: 专家 MLP → AWQ INT4
node tools\glm\quantize_glm_awq.mjs --src models\GLM-5.3-Flash.bf16.glmq --out models\GLM-5.3-Flash.awq.glmq

:: ModelOpt NVFP4 HF（如 LibertAIDAI/GLM-5.3-Flash-NVFP4）→ GLMQ
node tools\glm\import_glm_nvfp4.mjs --src models\GLM-5.3-Flash-NVFP4 --out models\GLM-5.3-Flash.nvfp4.glmq

:: 本地假权重自测
build\msvc-x64\bin\make_fake_glmq.exe --out models\_glm_selftest.glmq
build\msvc-x64\bin\llmoc_unit_tests.exe
```
