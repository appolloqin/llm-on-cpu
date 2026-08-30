# 目标模型实例化：Qwen3.8-27B

> 版本: v0.2  
> 角色: **当前交付目标模型之一**（与 Qwen3.5-4B hybrid 同族）  
> 状态: 🟢 以官方 `config.json` 为准（**稠密 hybrid，不是 MoE**）

---

## 1. 结构（官方）

| 参数 | 值 |
|---|---|
| 类型 | Dense + 视觉，`model_type=qwen3_5` |
| 参数量 | ~27B（含视觉约 28B） |
| 层数 / hidden / vocab | 64 / 5120 / 248320 |
| 注意力 | Gated DeltaNet + 每隔 4 层 Gated Attention（`layer_types`） |
| full attn | heads=24, kv=4, head_dim=256 |
| linear attn | key_heads=16, **value_heads=48**, dk=dv=128 |
| FFN intermediate | 17408 |
| `tie_word_embeddings` | **false**（必须用独立 `lm_head.weight`） |

> 转换结果出现 `expert=0` / `groups=0` **是正常的**：本模型无 MoE 专家组。  
> 引擎走 `backend=qwen3_5`，勿当 Qwen3-MoE 用。

## 2. 内存预算（BF16）

权重 LWC ≈ **52 GiB** + KV/工作区 → 建议主机 **≥64G** 常驻。

### 2.1 速度预期（CPU，当前引擎）

| 路径 | 常驻内存 | 单流 decode（经验） |
|---|---|---|
| BF16 `.lwc` | ~52 GiB | 常约 **0.3–2 tok/s**（带宽墙；无 SPR+AMX 时很难到 30） |
| INT4 `.qlwc` | ~13–16 GiB | 通常明显快于 BF16，优先用于聊天 |
| Qwen3.5-4B | 数 GiB | 交互体验最好 |

日志里若出现：

- `VirtualLock failed` / `locked=0.x MiB` 且 `resident≈50GiB`
- 生成时任务管理器 **磁盘狂跳 / 硬错误**

则多半在 **换页**，会远慢于上表。先保证物理内存够用，再谈优化。

带宽受限时 `OpenMP=88`（超线程）往往无益，可试：

```bat
set OMP_NUM_THREADS=32
start_bf16.cmd
```

（改为本机物理核数更佳。）

### 2.2 推荐：聊天用 INT4

```bat
download_int4.cmd --model Qwen/Qwen3.8-27B
REM 编辑 configs/engine_int4.yaml → path 指向 *.int4.qlwc，tokenizer_dir 指向 *-hf
start_int4.cmd
```

`dram_hot_gb` 对稠密 27B 几乎不影响「是否整模常驻」；缺内存时换 INT4 / 更小模型，而不是只调这个旋钮。

## 3. 操作

```bash
# 下载 + 转 LWC + prune（发布包脚本会自动跳过已完成步骤）
download_bf16.cmd --model Qwen/Qwen3.8-27B

# 配置 configs/engine.yaml:
#   model.path: models/Qwen3.8-27B.lwc
# tokenizer 旁路: models/Qwen3.8-27B-hf （须含 config.json）

start_bf16.cmd
```

启动日志应类似：
`Qwen35Model: layers=64 hidden=5120 heads=24/4 ... tie=0 lm_head=lm_head.weight`  
若仍 `tie=1` 或 heads=16，说明读到了错误/缺失的 config.json。
