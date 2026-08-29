# 目标模型实例化：Qwen3.8-27B

> 版本: v0.1（架构冻结基线上的模型实例化文档，不改动 ARCHITECTURE v0.4 结论）
> 角色: **当前交付目标模型**（DeepSeek-V4-Flash 保留为远期场景，见 ARCHITECTURE §1）
> 状态: 🟡 结构参数为 Qwen3 MoE 家族典型值，**待真实 config.json 校准**

---

## 1. 结构假设（校准前基准值）

| 参数 | 假设值 | 来源 |
|---|---|---|
| 总参数 | ~27-30B | 模型卡 |
| 每 token 激活 | ~3B（Qwen3-A3B 家族形态） | 家族推断 |
| 层数 | 48 | 家族推断 |
| 路由专家数 / top-k | 128 / 8 | 家族推断 |
| hidden × moe_intermediate | 2048 × 768 | 家族推断 |
| 前置稠密层 | first_k_dense_replace（若有）按稠密处理 | 转换器自动识别 |
| MTP/nextn 头 | 未知 → 开放项 O2 | 权重发布确认 |

## 2. 内存预算三档（BF16 不量化）

| 档位 | 构成 | 合计 | SPR(350GB/s) 预期 | 结论 |
|---|---|---|---|---|
| **A 全常驻（推荐）** | 权重54G + KV 6G + 工作区 2G + OS 2G | **~64 G** | 原始 ~55 t/s；MTP 后 80+ | **64G 机器正解**：零缺页，NVMe 路径闲置仅作兜底 |
| B 均衡 offload | dense 3.5 + 热专家 LRU 32 + KV 8 + 缓冲 4 + OS 2 | ~50 G | 35~45 t/s | 为 KV 大并发/长上下文留 KV 时选用 |
| C 极限 18G | dense 3.5 + 热专家 10 + KV 2 + 缓冲 1.5 + OS 1 | 18 G | **≤5 t/s** | 热区仅 ~10% 专家覆盖，缺页率过高；只满足"能跑"不满足 30 t/s SLO |

> 每专家体积 ≈ 3×2048×768×2B ≈ 9.4 MB；每层专家 1.2 GB；全部专家 ≈ 58 GB。
> **结论：18G 不成立。Qwen3.8-27B 是 64G 机器"全常驻"甜蜜点——比 V4-Flash 场景更宽裕。**

## 3. 与引擎机制的对应

| 机制 | 在本模型上的表现 |
|---|---|
| 三级驻留(D1) | 档位A = LRU 预算 ≥ 专家总体积 → 100% 命中，退化为纯 DRAM 引擎 |
| 双缓冲预取(D2) | 档位A下无缺页可预取；档位B/C才生效 |
| MTP 乘法器(D3) | 档位A主力：原始 ~55 t/s 已超 SLO，MTP 为余量 |
| 开发机(i7-14650HX, 68GB/s) | 档位A预计 ~11 t/s（无 AMX、带宽受限）—— 可跑通全链路，速度不作数 |

## 4. 操作手册（对应 docs/USAGE.md）

```bash
# 1. 下载(auto: modelscope→hf-mirror→hf, 国内默认走 ModelScope)
python tools/download_model.py --model Qwen/Qwen3.8-27B --out models/qwen3827-hf

# 2. 转换(专家命名 model.layers.N.mlp.experts.M.*_proj 与转换器正则天然兼容)
python tools/convert_lwc.py --src models/qwen3827-hf --out models/qwen3827.lwc

# 3. 校验和回填 + config 交叉核对 + 内存预算判定
#    Windows: .\build\msvc-x64\bin\lwc_verify.exe ; Linux/macOS: ./build/release/bin/lwc_verify
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc --update
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc --config models\qwen3827-hf\config.json --host-gb 64

# 4. 小样自检(可选, 几十MB走通全链路)
python tools/convert_lwc.py --src models/qwen3827-hf --out models/_mini.lwc --limit-experts 4

# 5. 真实权重进预取基准
.\build\msvc-x64\bin\m2_pipeline_bench.exe --file models\qwen3827.lwc --topk 8 --compute-us 5000
```

## 5. 校准清单（config.json 到位后一条命令闭环）

```powershell
.\build\msvc-x64\bin\lwc_verify.exe models\qwen3827.lwc --config models\qwen3827-hf\config.json
```

自动核对：层数 / 专家数 / 组数(=(层-fkdr)×专家) / dtype / 每专家体积漂移(<1%) / 词表维度，
并判定档A全常驻预算是否塞进主机内存（退出码 0=全部通过）。人工仍需确认的仅剩：

- [ ] 是否含 `mtp`/`nextn` 结构（O2：决定 M3 直接接入还是自训轻量头）
- [ ] `num_experts_per_tok`（权重文件不含路由，工具仅打印 INFO）
- [ ] 按实测总体积复核 §2 三档预算数字
