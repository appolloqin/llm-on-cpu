# 使用说明：模型放置 · 转换 · 验证 · 基准

> 本文回答「模型放在哪里」「怎么转成引擎格式」「转完能干什么」。
> 当前目标模型 = **Qwen3.8-27B**（内存预算/结构假设见 [MODEL_QWEN3.8-27B.md](MODEL_QWEN3.8-27B.md)）。
> 引擎尚在 M4 之前：**可完整跑通 权重转换→校验→预取/投机基准**；聊天式推理服务见 §5 路线。

---

## 1. 模型放在哪里（目录约定）

```
llm-on-cpu/
└─ models/                     # 已 gitignore, 不入库
   ├─ qwen3827-hf/             # ① 原始 HuggingFace 目录(config.json + *.safetensors)
   │   ├─ config.json
   │   ├─ model-00001-of-000xx.safetensors
   │   └─ model.safetensors.index.json
   └─ qwen3827.lwc             # ② 转换产物 —— 引擎只认这个格式
```

下载权重（**多源可切换，国内默认 ModelScope**，auto 失败自动降级 hf-mirror → hf）：

```bash
pip install modelscope          # 国内源; 或 pip install huggingface_hub (hf/hf-mirror 源)

# auto: modelscope → hf-mirror → hf 依次尝试(推荐)
python tools/download_model.py --model Qwen/Qwen3.8-27B --out models/qwen3827-hf

# 显式指定源 / 两端仓库 ID 不同时
python tools/download_model.py --model Qwen/Qwen3.8-27B --source hf-mirror
python tools/download_model.py --model org/name --ms-id org/name-ms --hf-id org/name
```

> 默认只拉 `*.safetensors + 配置/分词器`，跳过 `*.bin` 防止双份占盘（`--all` 关闭过滤）。

> 磁盘预算：HF 原始目录与 .lwc 各存一份，`models/` 需 ≈2× 模型体积（27B BF16 ≈ 54G ×2）。
> 不量化承诺 → 转换不做任何精度压缩，只做**布局重排**（专家块 4K 对齐、组内相邻）。

## 2. 转换：HF → LWC

**日常推荐一键**（下载→转换→校验回填→终检→可选清盘）：

```powershell
node tools/prepare_model.mjs --model Qwen/Qwen3.8-27B
# 已有 *-hf 目录:  --skip-download
# 只要转换不要校验: --no-verify
# 校验通过后删权重分片(省盘, 留 config/tokenizer): --prune-hf
# 校验通过后删整个 *-hf: --remove-hf
```

> 默认**不**删 HF 目录：方便重转 / `--config` 再核对。磁盘紧时加 `--prune-hf`（推荐）或 `--remove-hf`。
> 清理只在校验通过后执行；与 `--no-verify` 互斥。

分步（调试或换 Python 栈时）：

```powershell
# ① 转换(Python 或 Node 双栈等价, 产物字节级一致; 逐张量流式, 内存占用≈单个最大张量)
python tools/convert_lwc.py --src models/qwen3827-hf --out models/qwen3827.lwc
# 或(零 npm 依赖, Node≥18):
node tools/convert_lwc.mjs --src models/qwen3827-hf --out models/qwen3827.lwc

# ② 校验和回填(C++, 秒级; FNV 链式计算放 Python 会慢百倍, 故放这步)
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc --update

# ③ 终检(应显示 "N/N filled, 0 bad")
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc

# ④ 与 config.json 交叉核对 + 内存预算判定(推荐, 退出码 0=通过 3=不匹配)
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc --config models\qwen3827-hf\config.json
```

`--config` 自动核对项（即 MODEL_*.md 校准清单的自动化）：层数 / 专家数 /
专家组数(=(层-前置稠密)×专家) / dtype / **每专家体积漂移(<1%)** / 词表 vs embed 维度；
并输出档A(全常驻)预算判定 —— `plan-A fits host memory` 为 OK 才说明 64G 方案成立。

**convert_lwc.py 参数**

| 参数 | 说明 |
|---|---|
| `--src DIR` | HF 目录（识别 `model.safetensors.index.json` 分片） |
| `--out FILE` | 输出 .lwc |
| `--limit-experts N` | 每层只转前 N 个专家 —— **小样自检**，几十 MB 验证全链路 |
| `--verify` | 写完立即做 header/catalog 结构自检 |

支持张量命名：`layers.N.mlp.experts.M.{gate,up,down}_proj.weight`（DeepSeek 系）；
命名不匹配的张量按稠密权重原名保留。**bf16 权重需要 `pip install torch`**（fp16/fp32 纯 numpy 即可）。

依赖安装：`pip install numpy safetensors`（+可选 torch）。

## 3. 转换产物的自检

```powershell
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc --info
# file        : models/qwen3827.lwc (xxx GiB)
# dtype/align : BF16 / 4096
# tensors     : N (dense=A, expert=B)
# MoE         : layers=48 experts/layer>=128 (groups=48)   ← 与 config.json 核对
```

> 可执行文件路径：Windows `.\build\msvc-x64\bin\*.exe`；Linux/macOS `./build/release/bin/`（须先 configure/build）。裸写 `lwc_verify` 会报「无法识别」——它不在 PATH 里。

若 `--info` 的 layers/experts 与 config.json 不符 → 转换器没识别出专家命名，停下来排查，
不要继续。仓库自带微型假模型可全程演练（秒级）：

```powershell
python scripts/make_fake_hf.py
python tools/convert_lwc.py --src models/_selftest-hf --out models/_selftest.lwc --verify
.\build\msvc-x64\bin\lwc_verify.exe models\_selftest.lwc --update
.\build\msvc-x64\bin\lwc_verify.exe models\_selftest.lwc
```

## 4. 转换完成后怎么用模型

### 4.1 启动 OpenAI 兼容服务

```powershell
cmd /c scripts\build_dev.cmd
.\build\msvc-x64\bin\llmoc_server.exe --config configs\engine.yaml
```

要求：`models/<name>.lwc` + `models/<name>-hf/tokenizer.json` + `config.json`（`--prune-hf` 后仍在）。4B BF16 建议主机内存 ≥16G。

```powershell
Invoke-RestMethod http://127.0.0.1:15085/healthz
Invoke-RestMethod http://127.0.0.1:15085/v1/chat/completions -Method Post -ContentType 'application/json' `
  -Body '{"messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":64}'
# 流式 SSE: Body 加 "stream":true
Invoke-WebRequest http://127.0.0.1:15085/metrics
```

### 4.1b INT4（QLWC）—— 独立路径，不改 BF16

对齐 GPTQ/AWQ 布局的 **BF16→INT4 量化** + **`llmoc_server_int4` 推理**（原 `llmoc_server` / `.lwc` 不变）：

```powershell
# ① LWC(BF16) → QLWC(INT4)；method=awq|gptq，默认 group=128（Python / Node 等价；大模型建议 Python 更快）
python tools/quantize_int4.py --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method gptq
# 或(零 npm 依赖, Node≥18):
node tools/quantize_int4.mjs --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method gptq

# ② 启动 INT4 服务（默认端口 15085）
.\build\msvc-x64\bin\llmoc_server_int4.exe --config configs\engine_int4.yaml

Invoke-RestMethod http://127.0.0.1:15085/v1/chat/completions -Method Post -ContentType 'application/json' `
  -Body '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

- 配置里请显式设 `model.tokenizer_dir: models/Qwen3.5-4B-hf`（`.qlwc` 名无法从 path 推对 HF 旁路）。
- 仅量化 2D 且 `K % group_size == 0` 的权重；norm / embedding 等透传原精度。
- MoE INT4 尚未接线；当前面向 Qwen3.5 hybrid。
- 模式：`configs/engine_int4.yaml` 的 `mode:` 支持 `pure_cpu|hybrid_gpu|pure_gpu|auto|layer_stream`；GPU 预算 `tiers.gpu_vram_gb`。发布包可用 `start_int4.cmd [config]`。

### 4.1c DeepSeek / Kimi stub（冒烟，非真权重）

```powershell
.\build\msvc-x64\bin\make_fake_ds.exe models\fake_ds.dskq
.\build\msvc-x64\bin\llmoc_server_ds.exe --config configs\engine_ds_nvfp4.yaml
# 或发布包: download_ds.cmd && start_ds.cmd   （端口 15086）

.\build\msvc-x64\bin\make_fake_kimi.exe models\fake_kimi.kimiq
.\build\msvc-x64\bin\llmoc_server_kimi.exe --config configs\engine_kimi_hybrid.yaml
# 单卡 pure_gpu 会自动降级 layer_stream；推荐 hybrid
```

### 4.2 基准工具（权重/IO，非对话）

```powershell
$BIN = ".\build\msvc-x64\bin"
& "$BIN\lwc_verify.exe" models\Qwen3.5-4B.lwc --info
& "$BIN\m2_pipeline_bench.exe" --file models\Qwen3.5-4B.lwc --topk 8 --compute-us 5000
```

**结果解读**（重要，小模型会"反向"）：

| 现象 | 含义 |
|---|---|
| speedup ≈ 1.5~2x | IO 是瓶颈 —— 流水线在正确工作 |
| speedup < 1 | 模型太小/全在页缓存 —— 不是 bug |
| io_wait 双双≈0 | 加 `--part-mib` 或换更大模型 |

### 4.3 双后端：Qwen3.5 vs MoE

| `.lwc` 特征 | 后端 | 说明 |
|---|---|---|
| `groups == 0`（如 Qwen3.5-4B） | `Qwen35Model` | hybrid linear_attn + full_attention |
| `groups > 0`（如 Qwen3.8-27B） | `MoeModel` | GQA + router top-k + `ExpertPrefetcher` |

改 `configs/engine.yaml` 的 `model.path`（及旁路 `*-hf` tokenizer/config）即可切换。MoE 需专家命名可被 `convert_lwc` 识别：`layers.N.mlp.experts.M.*_proj`。

### 4.4 能力边界

| 能力 | 状态 |
|---|---|
| 权重转换/校验/预取/投机状态机 | ✅ |
| Qwen3.5 文本对话 + `/v1/chat/completions` | ✅ CPU 参考实现 |
| MoE 对话（预取接线） | ✅ |
| SSE / `/metrics` / radix / 调度串行队列 | ✅ |
| 多模态 vision / 真 MTP draft | ⏳ |
| hybrid-gpu / pure-gpu | ❌ M5 |

## 5. API 一览

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 对话；`stream=true` → SSE |
| `GET /healthz` | 存活 |
| `GET /metrics` | Prometheus 文本 |

认证：环境变量 `LLMOC_API_KEY`（可选 Bearer）。详见 `configs/engine.yaml`。

## 6. 故障排查

| 症状 | 原因/处理 |
|---|---|
| `VirtualLock failed (1453)` 警告 | Windows 工作集配额不足，best-effort 锁页降级，**不影响正确性**；生产在 Linux 用 mlock |
| 日志出现 `falling back to thread-pool` | io_uring 不可用（非 Linux / 缺 liburing）——自动降级 |
| `bf16 权重需要 torch` | `pip install torch`，或转换 fp16 版模型 |
| `LWC checksum mismatch` | 盘上数据损坏，重新转换该文件 |
| `--file: header lacks layers.0.experts.0.gate` | 转换源不含 MoE 专家或命名不匹配，检查 §2 命名说明 |
| `cannot open tokenizer` | 勿用 `--remove-hf`；需保留 `*-hf/tokenizer.json` |
| 首 token 极慢 / OOM | 正常：全量 BF16 常驻 + 朴素 GEMM；加内存或减小 `max_tokens` |

里程碑对照见 README；设计依据见 `docs/ARCHITECTURE.md`。
