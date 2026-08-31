# 实现文档：三模独立执行 · MoE 激活路径 GPU · 分族 Cache/Attn · 单机多卡

> **日期**: 2026-08-31（同日修订：补 **引擎内单机多卡**）  
> **状态**: 🟢 设计目标已确认，按本文一次性落地（**不分阶段、不设 P0/P1 里程碑拆交付**）  
> **取代口径**:
> - `ARCHITECTURE.md` §3.1 中「pure_gpu = 全模权重常驻 VRAM」→ **作废**，改用本文 §2  
> - `MODEL_GLM53_FLASH.md` §2 中「pure_gpu 需整模进显存 / 22G 不够」→ **作废**，改用本文 §2 / §5.3  
> - 现有 M5（`hal/cuda_backend` 仅 BF16 cuBLAS 透传）视为**过渡脚手架**，须按本文替换为独立 `exec/*`，不得继续在共享 hot path 上堆 `if (mode)`  
> - 「pure_gpu / hybrid 仅单卡」→ **作废**；单卡是 `devices` 默认特例，多卡见 **§2.3 / §6.5 / G8**

---

## 0. 目标一览（一次全部达成）

| # | 设计目标 | 验收口径（必须同时满足） |
|---|---|---|
| G1 | **三模独立模块** `exec/{cpu,hybrid,gpu}` | 仅共享契约层；改 GPU 不得改 CPU 前向实现文件 |
| G2 | **Hybrid** = GPU 做 Attn+KV(+共享稠密)；**专家算在 CPU** | 层间仅传激活；专家权重不因 hybrid 强制上 GPU |
| G3 | **Pure-GPU（MoE）** = **激活路径全 GPU 算** + **top‑k 专家按需/预取进 VRAM** | 判定条件是「单层 active working set ≤ **每卡/网格** VRAM」，**禁止**用总参量判死刑 |
| G4 | **Pure-CPU** 路径零回归 | 默认 `mode: pure_cpu`；无 CUDA 构建行为与现网一致；单元测试绿 |
| G5 | **分族适配** Qwen3.8 / DeepSeek-V4 / GLM-5.3 / Kimi-K3 | 各族独立 model pack + 正确 cache/attn 布局；禁止一个 `MoeModel` 硬塞四族 |
| G6 | **热路径重写** MoE / Attn / KvStore | 各族按自身论文/官方图实现；禁止物化无用的全长 K/V（DeepSeek 系 latent） |
| G7 | 配置与二进制可切换三模 | `pure_cpu` / `hybrid_gpu` / `pure_gpu` / `auto`；`pure_gpu` 无 CUDA → **启动失败**（不静默降级） |
| G8 | **引擎内单机多卡**（同进程 device mesh） | `devices.ids` ≥2 时同一推理图跨卡；**EP 为主、TP 为辅**；单卡配置行为不变；**不做多机** |

**非目标（明确不做）**：**多机 / 跨节点** TP·EP·PP 集群调度与分布式 launcher；把第三方 vLLM/SGLang 当默认后端；视觉多模态端到端（族内可留 hook，本文不验收）。

> **澄清**：多进程「每进程绑一张卡、各服各的请求」= 部署层 DP，**不算** G8。G8 要求 **一个 server 进程内** 多卡协同完成 **同一次** forward。

---

## 1. 硬约束与原则

1. **纯 CPU 不可破坏**：`exec/cpu` 与现有 `llmoc_server` / `_int4` / `_glm` 的 CPU 热路径不得因 CUDA 头文件、静态链接 cudart、或共享 `if (cuda)` 分支被污染。CUDA 仅出现在 `exec/gpu`、`exec/hybrid` 与动态探测层。
2. **独立优于复用**：模式之间、模型族之间 **禁止共享前向实现**。允许共享的仅限：
   - `common/`（log、config、alloc、HTTP、tokenizer）
   - **契约头文件** `contracts/`（见 §3）
   - 权重容器读写若格式相同可共享 I/O，但 **runtime pin/prefetch 策略按模式分文件**
3. **Sparse MoE 显存判据**（修订后的第一性原理；**按 rank 计**）：

```
# 单卡（world_size=1）—— 兼容原式
W_active(layer) = bytes(attn_or_shared_on_device)
                + Σ_{e ∈ top-k(layer)} bytes(expert_e)
                + bytes(kv_or_latent_working_set)
                + bytes(workspace)

# 单机多卡 —— 每张卡各自满足（见 §2.3）
W_active[rank](layer) = bytes(attn_shard[rank]) + bytes(kv_shard_or_replicate[rank])
                      + Σ_{e ∈ top-k ∩ ExpertsOwned[rank]} bytes(expert_e)   # 最坏再加槽裕量
                      + bytes(workspace[rank]) + bytes(comm_scratch[rank])

可行(pure_gpu) ⇔ ∀rank: max_over_layers W_active[rank] ≤ VRAM_budget[rank] − margin
可行(hybrid)   ⇔ ∀rank用于attn者: bytes(attn+KV份额) ≤ reserve[rank]
                 ∧ CPU 侧专家热槽可装 top-k（DRAM/NVMe 预取）
```

4. **「专家很少」≠「激活一定小」**：例如 Kimi-K3 选 16/896 但仍 ~104B 激活 → **单卡**消费级 `pure_gpu` **默认拒绝**；**多卡 EP** 下按 §2.3 重新求解，拟合则允许（见 §5.4）。
5. **单机多卡通信**：仅进程内 NCCL（或等价 CUDA IPC + 自研 all-to-all）；禁止依赖外部编排器。无 NCCL 且 `world_size>1` → **启动失败**（不静默单卡）。

---

## 2. 三模语义（最终口径）

| 模式 | 计算落点 | 权重驻留 | PCIe 角色 | 启动失败条件 |
|---|---|---|---|---|
| **pure_cpu** | 全部 CPU | Attn/共享 DRAM；专家 LRU@DRAM + NVMe | 无 | 无（默认） |
| **hybrid_gpu** | **Attn+KV(+MTP/共享稠密) → GPU**；**Routed experts → CPU** | VRAM：attn/KV/共享；DRAM/NVMe：专家 | 激活 H↔D（MB 级） | CUDA 缺失 → **降级 pure_cpu** 并打 WARN |
| **pure_gpu** | **全部含专家 → GPU**（可跨多卡） | 每卡 VRAM：**本 rank 专家槽 + attn/KV 份额**；其余专家 DRAM/NVMe | **专家块 H2D** + **卡间 all-to-all/all-reduce** | CUDA 缺失 → **失败**；任一 rank `W_active` 超预算 → **失败**；`world_size>1` 无 NCCL → **失败** |

### 2.1 与旧文档差异（必须写进代码注释）

| 旧说法 | 新说法 |
|---|---|
| pure_gpu = 整模进 VRAM、offload 关闭 | pure_gpu = **算全在 GPU**；专家 **激活集** 流式/双缓冲进 VRAM |
| 320B ≫ 24G → pure_gpu N/A | 看 **~18B act × 量化** 能否放入「一层 top‑k + attn + KV」 |
| PlacementPlanner pure_gpu「尽量全塞 VRAM」 | Planner 目标改为 **最小化 PCIe miss + 保证 W_active 拟合** |

### 2.2 统一时间模型

```
T_token = max(
  T_compute(device of this mode),
  B_dram / BW_dram,          # CPU 专家或 host 驻留
  B_nvme / BW_nvme,          # 冷专家
  B_pcie / BW_pcie,          # hybrid: 激活; pure_gpu: 专家块+激活
  B_vram / BW_vram,          # GPU 常驻权重读取
  B_nvlink_or_pcie_p2p / BW_p2p   # 仅 world_size>1: 卡间通信
)
```

### 2.3 单机多卡（引擎内 Device Mesh）— G8

**范围**：同一 OS、同一进程、`cudaSetDevice` + NCCL communicator；`world_size = |devices.ids|`。  
**不做**：跨主机、MPI 多进程训练式 launch、Ray/K8s 编排。

#### 2.3.1 策略优先级（冻结）

| 策略 | 代号 | 用途 | 何时用 |
|---|---|---|---|
| **Expert Parallel** | `ep` | 专家按 `expert_id % ep_size`（或连续分片）归属各卡；token 按路由 **all-to-all** 到拥有该专家的卡，算完再 gather/combine | **MoE pure_gpu 默认** |
| **Tensor Parallel** | `tp` | Attn/共享 Linear 按列/行切；每步 **all-reduce** | 稠密族（Qwen）或 MoE 的 attn/共享；`ep` 后单卡 attn 仍爆显存时叠加 |
| **EP+TP** | `ep_tp` | `world = ep_size × tp_size` 二维 mesh | 大激活 MoE（如 Kimi）或 attn+专家双爆 |
| **Hybrid 多卡** | `hybrid_mesh` | Attn/KV **仅 GPU 侧**可 TP；专家仍在 **CPU**（与 G2 一致） | `mode=hybrid_gpu` 且多卡时：加速注意力，不把专家算力搬上多卡 |

**禁止**作为本引擎 G8 主路径：仅「把整层流水到不同卡」的纯 PP（实现复杂、decode 气泡大）；若族实现内部用微流水，不暴露为用户策略。

#### 2.3.2 默认自动策略 `devices.strategy: auto`

```
if mode == pure_cpu:           world_size 强制 1（忽略多余 ids，打 WARN）
elif mode == hybrid_gpu:
  if n_gpu == 1:               单卡 attn
  else:                        tp 切 attn（ep_size=1）；专家 CPU
elif mode == pure_gpu:
  if family has MoE:
    if n_gpu == 1:             单卡 active-slot
    else:                      ep_size = n_gpu, tp_size = 1
                               若 ∀rank W_active 仍失败 → 尝试 ep_tp（折半 ep、引入 tp）仍失败则启动失败
  else:                        # 稠密
    tp_size = n_gpu
```

#### 2.3.3 EP 运行时数据面（pure_gpu）

```
gate(L) on rank0 (或每卡复制 router 权重做本地 gate)
  → 得到 top-k ExpertId
  → all-to-all 调度：token 分片发往 owner(rank)
  → 各 rank: pin/prefetch 本地专家槽 → GPU expert GEMM
  → all-to-all / reduce-scatter 回传加权输出 → combine
```

- **最坏负载**：top-k 全落同一 rank → Planner 按 `min(top_k, n_local_experts)` 为每卡预留专家槽，不得按「平均 top-k/ep_size」低估。  
- **冷专家**：仍可 NVMe→DRAM→**owner 卡** H2D；非 owner 不持有该专家 VRAM 槽。  
- **P2P**：优先 NVLink/P2P；不可用则 NCCL 走 PCIe，计入 §2.2 的 `B_p2p`。

#### 2.3.4 TP 运行时数据面（attn/共享）

- Column/Row parallel GEMM + 必要 **all-reduce**（与常见 Megatron 风格一致，实现可自研最小集）。  
- KV：默认 **按 TP 切头（GQA/MLA head shard）**；无法切的 latent 状态 **复制** 并在文档/日志标明复制开销。

#### 2.3.5 与「多进程单卡」边界

| | 引擎内多卡 (G8) | 多进程 DP |
|---|---|---|
| 进程数 | 1 | = 卡数 |
| 一次 forward | 多卡协同 | 单卡完成 |
| 配置 | `devices.ids: [0,1]` | 多个 server + 不同 `CUDA_VISIBLE_DEVICES` |

---

## 3. 目录与模块边界（落地结构）

```
src/
├─ contracts/                         # 唯一允许跨模式/跨族依赖的头
│  ├─ exec_mode.h                     # ExecMode 枚举 + 解析（从 sched/ 上移或薄包装）
│  ├─ tensor_view.h                   # DevicePtr / HostPtr / dtype / shape；含 device_ordinal
│  ├─ device_mesh.h                   # DeviceMesh{ids, ep_size, tp_size, rank_of_expert()}
│  ├─ weight_block.h                  # ExpertId, BlockHandle, nbytes, checksum
│  ├─ kv_store.h                      # IKvStore：append / rollback / prefix_hit
│  ├─ moe_router.h                    # IRouter：gate → topk ExpertId[]
│  ├─ linear_op.h                     # IGemm / IDequantGemm 契约
│  ├─ collective.h                    # ICollective：allreduce / alltoall（进程内）
│  └─ family_pack.h                   # IFamilyPack：build_graph / make_kv / make_moe
│
├─ exec/                              # 三模独立实现（禁止互相 #include 对方 .cpp 细节）
│  ├─ cpu/
│  │  ├─ exec_cpu.h/.cpp              # 组装 CPU HAL + CPU KvStore + CPU MoE
│  │  ├─ gemm_cpu.*                   # 现有 cpu_ops / amx / int4 / nvfp4-cpu 迁入或适配
│  │  ├─ attn_cpu.*
│  │  └─ expert_runtime_cpu.*         # pin/prefetch 仅 DRAM/NVMe
│  ├─ hybrid/
│  │  ├─ exec_hybrid.h/.cpp           # world_size≥1；多卡时仅 TP attn
│  │  ├─ attn_gpu.*                   # 仅 attn/KV/共享稠密
│  │  ├─ expert_runtime_cpu.*         # 专家仍走 CPU（可链到 exec/cpu 的专家运行时，经契约）
│  │  └─ activation_xfer.*            # H2D/D2H 激活
│  └─ gpu/
│     ├─ exec_gpu.h/.cpp              # 持有 DeviceMesh；world_size=1 为退化路径
│     ├─ attn_gpu.*                   # 单卡或 TP
│     ├─ expert_runtime_gpu.*         # 本地专家槽；EP 时只服务 ExpertsOwned[rank]
│     ├─ nvfp4_gemm_gpu.*
│     ├─ residency_gpu.*              # 每 rank W_active、双缓冲槽
│     ├─ mesh_nccl.*                  # ICollective 的 NCCL 实现（动态加载）
│     └─ ep_dispatch.*                # gate→alltoall→gemm→combine
│
├─ families/                          # 分族 pack（图 + cache 布局）
│  ├─ qwen38/
│  ├─ deepseek_v4/
│  ├─ glm53/                          # 现有 src/glm/ 迁入或作为本目录实现体
│  └─ kimi_k3/
│
├─ weights/                           # 容器 I/O（LWC/QLWC/GLMQ）保持；驻留策略调用 exec 侧 runtime
├─ sched/
│  ├─ mode_controller.*               # 解析 mode + devices → IExecBackend
│  └─ placement_planner.*             # active-set × mesh（每 rank 预算）
├─ server/                            # main* 只选 family + exec，不写算子
└─ common/
```

**独立规则（CI 可加依赖图检查，建议）**：

- `exec/cpu/**` 不得 `#include` `exec/hybrid` / `exec/gpu` 下任何头（`contracts` 除外）。
- `families/qwen38/**` 不得依赖 `families/glm53/**` 等其他族实现。
- `llmoc_server`（Qwen）链接 `exec` + `families/qwen38`；`llmoc_server_glm` 链接 `exec` + `families/glm53`。

> 迁移策略：允许短暂保留旧路径文件为 thin wrapper（转发到新目录），但 **功能完成标准** 是热路径源文件落在 `exec/*` 与 `families/*`，旧 `if (mode)` 透传删除。

---

## 4. 核心契约（接口级，实现必须对齐）

### 4.1 `IExecBackend`

```cpp
struct ExecCaps {
  bool experts_on_gpu = false;
  bool attn_on_gpu = false;
  int world_size = 1;
  int ep_size = 1;
  int tp_size = 1;
  size_t vram_budget_per_rank = 0;
  size_t dram_budget = 0;
};

class IExecBackend {
 public:
  virtual ~IExecBackend() = default;
  virtual ExecCaps caps() const = 0;
  virtual const DeviceMesh& mesh() const = 0;  // world_size==1 时单元素 mesh
  virtual void configure(const PlacementPlan&) = 0;
  virtual IGemm* gemm() = 0;                   // 当前 rank 视角；TP 时为分片 GEMM
  virtual IExpertRuntime* experts() = 0;
  virtual IKvDevicePolicy* kv_policy() = 0;
  virtual ICollective* coll() = 0;             // world_size==1 时为 no-op 实现
};
```

- `make_exec(ExecMode, HwProfile, DeviceMeshSpec)`：`ModeController` 唯一入口。
- `pure_gpu` → `ExecGpu`；`hybrid_gpu` → `ExecHybrid`；`pure_cpu` → `ExecCpu`（忽略 mesh 或强制 size=1）。

### 4.2 `IExpertRuntime`（Hybrid vs GPU 分叉点）

```cpp
class IExpertRuntime {
 public:
  virtual void prefetch(int layer, const ExpertId* ids, int n) = 0;
  virtual void pin(int layer, const ExpertId* ids, int n, BlockHandle* out) = 0;
  virtual void gemm_swiglu(const BlockHandle&, const TensorView& x, TensorView& y) = 0;
  virtual void release(const BlockHandle*, int n) = 0;
  // EP：仅处理本 rank 拥有的专家；调用方先经 ep_dispatch 过滤
  virtual bool owns(ExpertId) const = 0;
};
```

| 实现 | `gemm_swiglu` 设备 | `prefetch` |
|---|---|---|
| `ExpertRuntimeCpu` | CPU | NVMe→DRAM；`owns` 恒 true |
| `ExpertRuntimeGpu` | GPU（本 rank） | DRAM/NVMe→**本卡 VRAM 双槽**；`owns` 按 mesh |

### 4.2b `ICollective`（仅 hybrid/gpu，cpu 为 noop）

```cpp
class ICollective {
 public:
  virtual void allreduce(TensorView inout, /*sum*/) = 0;   // TP
  virtual void alltoall(TensorView send, TensorView recv) = 0;  // EP dispatch
  virtual void broadcast(TensorView, int root) = 0;
};
```

### 4.3 `IKvStore`（分族实现，禁止统一物化满 K/V）

```cpp
class IKvStore {
 public:
  virtual void append_token(/* family-specific write */) = 0;
  virtual Snapshot snapshot() = 0;
  virtual void restore(const Snapshot&) = 0;
  virtual int prefix_reuse(const TokenSpan&) = 0;
};
```

各族必须提供自己的 store（见 §5），不得强迫 DeepSeek 走「每层满长 K/V float buffer」。

### 4.4 `IFamilyPack`

```cpp
class IFamilyPack {
 public:
  virtual const char* name() const = 0;
  virtual std::unique_ptr<IKvStore> make_kv(const ModelConfig&, IExecBackend&) = 0;
  virtual std::unique_ptr<ICausalLM> make_model(WeightStore&, IExecBackend&) = 0;
  // 向 Planner 提供：每层 attn/共享字节、单专家字节、默认 top-k
  virtual ActiveSetProfile active_profile() const = 0;
};
```

`PlacementPlanner::solve(mode, hw, mesh, family.active_profile(), expert_freq)`：

- **hybrid**：先扣每卡 `vram_attn_reserve`（TP 时按 shard）；专家计算仍 CPU。
- **pure_gpu 单卡**：每层 **≥ top‑k 专家槽 + 预取深度 D**；装不下 → plan 失败。
- **pure_gpu 多卡 EP**：每 rank 槽位数按 **最坏** `min(top_k, n_experts_owned[rank])` + prefetch；专家所有权 `expert_id % ep_size`（可配置 `ep_shard: contiguous|mod`）。

---

## 5. 分族适配矩阵（全部写入本次实现）

### 5.1 Qwen3.8-27B（稠密 Hybrid Attention）

| 项 | 要求 |
|---|---|
| 图 | 48× Gated DeltaNet（线性状态 O(1)）+ 16× GQA 全注意力 |
| Cache | `DeltaNetState`（conv/recurrent）+ 仅全注意力层的 paged K/V |
| MoE | **无**；FFN 随模式整层在 CPU 或 GPU |
| pure_gpu | 量化后 **全激活常驻/可分页 FFN**；多卡默认 **TP** |
| 入口 | 现有 `llmoc_server` / `_int4` → `families/qwen38` |
| 量化 | BF16 LWC / AWQ INT4 QLWC（保持） |

### 5.2 DeepSeek-V4（MoE + 压缩/稀疏 KV）

| 项 | 要求 |
|---|---|
| 图 | CSA/HCA 压缩 KV + indexer；Flash 档 ~13B act；Pro ~49B act |
| Cache | **Latent / compressed cache**（\(c_{KV}\) 或等价）；禁止默认展开为全长 K、V 再 SDPA，除非调试开关 |
| MoE | FP4/NVFP4 专家优先；shared + routed |
| HiSparse | hot KV device / cold host（可与 hybrid/pure_gpu 驻留策略对齐） |
| pure_gpu | Flash：单卡或 EP；Pro：优先 **EP**，单卡允许层内槽轮转 |
| 入口 | 新建 `llmoc_server_ds` **或** recipe 分支；**不要**污染 Qwen `MoeModel` 默认语义 |

### 5.3 GLM-5.3-Flash（独立，已有 `src/glm/`）

| 项 | 要求 |
|---|---|
| 图 | KDA + DSA/MLA-like + mHC + shared/routed expert |
| Cache | KDA 线性状态 + 稀疏/压缩注意力索引；与 DeepSeek 类似，避免无用满 KV |
| 量化 | **AWQ INT4 与 NVFP4 一等公民**（GLMQ） |
| Hybrid | Attn/KDA/DSA + KV 策略在 GPU（多卡可 TP）；routed experts CPU |
| Pure-GPU | **NVFP4 专家**；单卡 active-slot 或 **多卡 EP**；`W_active[~18B 量化]/ep` 拟合 多×16–24G |
| 入口 | `llmoc_server_glm`；实现迁入 `families/glm53`（或 `src/glm` 实现 `IFamilyPack`） |

### 5.4 Kimi-K3（超大激活 MoE）

| 项 | 要求 |
|---|---|
| 图 | KDA + Gated MLA (NoPE) + AttnRes + LatentMoE（16/896，~104B act） |
| Cache | Gated MLA latent；禁止按稠密 GQA 全量物化 |
| Pure-GPU **单卡** | **默认 `ActiveSetDoesNotFit` → 启动拒绝** |
| Pure-GPU **多卡** | **EP 或 ep_tp** 按 §1/§2.3 逐 rank 求解；拟合则允许，否则失败并提示加卡/改 hybrid |
| Hybrid | 主推荐（少卡）：GPU attn/latent（可 TP），CPU 专家 + 激进预取 |
| 入口 | `families/kimi_k3` + 独立 server 或显式 recipe；未就绪权重时可先 fake 图 + 单测 |

---

## 6. PlacementPlanner / Residency（实现算法）

### 6.1 输入

- `HwProfile`：DRAM、**每卡 VRAM**、P2P/NVLink 是否可用、BW 标定
- `DeviceMesh`：`ids` / `ep_size` / `tp_size`
- `ActiveSetProfile`：`per_layer_attn_bytes`、`expert_bytes`、`top_k`、`n_layers`、`n_experts`
- `expert_freq[]`（可选 warmup）
- `ExecMode`

### 6.2 Pure-GPU 求解（必须）

**单卡（world_size=1）**：

```
slots_per_layer = top_k + prefetch_extra   # prefetch_extra ≥ top_k（下一层）
need = attn_bytes + shared_bytes + slots_per_layer * expert_bytes
       + kv_working_bytes + workspace
if need > vram_budget: FAIL
else:
  allocate VRAM: attn/shared/KV 固定区 + 全局专家槽环形缓冲
  DRAM: 其余专家（mlock 热集按 freq）
  NVMe: 冷集
```

运行时：`gate(layer L)` → `prefetch(L+1, topk)` → `pin(L)` → `gpu_gemm` → `release` 复用槽。

**多卡 EP（§2.3）**：

```
for each rank:
  owned = ExpertsOwned[rank]
  slots = min(top_k, |owned|) + prefetch_extra_local
  need[rank] = attn_bytes/tp_size + kv_share[rank]
             + slots * expert_bytes
             + workspace + comm_scratch
  if need[rank] > vram_budget[rank] - margin: FAIL
assign expert home = shard(expert_id, ep_shard, ep_size)
```

### 6.3 Hybrid 求解

```
vram[rank] = attn_shard + kv_share (+ 可选极热专家只读缓存，不算专家算力设备)
experts: 全部 ExpertRuntimeCpu（与卡数无关）
world_size>1 → TP on attn only
```

### 6.4 与现实现差异

删除/改写 `PlacementPlanner` 中「pure_gpu 把专家列表尽量塞满 VRAM 直至预算」的语义为 **按槽位数 + 热度填充 DRAM 热集**，VRAM 专家区按 **槽** 计而非「试图装下全部专家」。多卡时预算表是 **每 rank 一张**，不是把多卡 VRAM 简单相加后当单池（除非策略显式 `ep` 已分片）。

### 6.5 DeviceMesh 构造

```
parse devices.ids | auto → list
validate cuda device count
resolve strategy (auto 规则见 §2.3.2)
ep_size * tp_size == world_size  # 必须整除
init NCCL (gpu/hybrid, world_size>1)
```

---

## 7. 算子与内核清单（一次做齐）

| 内核 | CPU (`exec/cpu`) | Hybrid | GPU (`exec/gpu`) |
|---|---|---|---|
| BF16/F16 GEMM | AMX/oneDNN（已有） | 共享稠密走 GPU cuBLAS | 同左 |
| AWQ INT4 GEMM | 已有 int4 / glm awq | 专家用 CPU | 专家用 GPU dequant+GEMM 或 INT4 tensor core 路径 |
| NVFP4 GEMM | CPU dequant 参考（已有 glm） | 专家 CPU | **主路径**：H2D 量化块 + GPU dequant/GEMM |
| GQA SDPA | 已有 | GPU | GPU |
| Gated DeltaNet | Qwen 已有 CPU | GPU 移植 | GPU |
| KDA | GLM CPU | GPU | GPU |
| MLA / CSA / HCA / indexer | DeepSeek/GLM 族实现 | GPU | GPU |
| mHC | GLM | 随 attn 侧设备 | 同左 |
| Router top-k | CPU 可接受 | CPU 或 GPU 小核 | GPU 小核优先 |
| Expert combine | 随专家设备 | CPU | GPU（EP 后 combine） |
| AllReduce / AllToAll | noop | TP attn 时需要 | EP+TP 需要（NCCL） |

**消费级 SM89 说明**：无官方 FlashMLA 时，用族内 **KPool/压缩索引 + SDPA/等价核**；不得假装 SM90 专用指令可用。

---

## 8. 配置契约

扩展（各族 yaml 共用字段名）：

```yaml
mode: pure_cpu | hybrid_gpu | pure_gpu | auto

# 引擎内单机多卡（G8）。省略或 ids: [0] = 单卡，行为与旧版一致。
devices:
  ids: [0, 1]                 # 或 auto（使用全部可见 CUDA 设备）
  strategy: auto              # auto | ep | tp | ep_tp
  ep_size: 0                  # 0 = 由 strategy/auto 推导；须 ep_size*tp_size=world_size
  tp_size: 0
  ep_shard: mod               # mod | contiguous
  nccl: true                  # world_size>1 时必须 true；失败则启动失败

tiers:
  dram_hot_gb: 40
  gpu_vram_gb: 20             # 每卡预算；多卡不要写成「总和」
  kv_pool_gb: 2
  prefetch_buf_gb: 4
  pure_gpu:
    expert_slot_extra: 1
    strict_vram: true
    margin_mb: 512

family: qwen38 | deepseek_v4 | glm53_flash | kimi_k3
```

推荐默认（文档与示例配置一致）：

| 族 | 64G + 单卡 16–24G | 同机 ≥2×16–24G |
|---|---|---|
| Qwen3.8 INT4 | `hybrid_gpu` / `pure_gpu` | `pure_gpu` + `tp` |
| DeepSeek-V4-Flash 量化 | `pure_gpu` | `pure_gpu` + `ep` |
| GLM-5.3 NVFP4 | `pure_gpu` 或 `hybrid_gpu` | **`pure_gpu` + `ep` 优先** |
| Kimi-K3 | `hybrid_gpu`；单卡 `pure_gpu` 拒绝 | `pure_gpu` + `ep`/`ep_tp` 若拟合 |

---

## 9. Server / 构建接线

| Target | Family | Exec |
|---|---|---|
| `llmoc_server` | qwen38 BF16 | cpu/hybrid/gpu（含 mesh） |
| `llmoc_server_int4` | qwen38 INT4 | cpu/hybrid/gpu |
| `llmoc_server_glm` | glm53 | cpu/hybrid/gpu |
| `llmoc_server_ds`（新增） | deepseek_v4 | cpu/hybrid/gpu |
| `llmoc_server_kimi`（新增） | kimi_k3 | cpu/hybrid/gpu（单卡 pure_gpu 可 reject） |

CMake：

- `LLMOC_WITH_CUDA` 编入 `exec/hybrid` + `exec/gpu`（含 `mesh_nccl`）。
- NCCL：**动态加载**（与 cudart 策略一致）；缺库且 `world_size>1` → 启动失败。
- `pure_cpu` 默认预设 **不链接** CUDA/NCCL 导入库。

启动日志必须可 grep：

```
family=glm53_flash mode=pure_gpu mesh=ep world=2 ep=2 tp=1 devices=0,1
experts_on=gpu attn_on=gpu W_active_est_mb_per_rank=... vram_budget_mb_per_rank=...
```

---

## 10. 测试与验收（一次性门禁）

### 10.1 正确性

| 用例 | 要求 |
|---|---|
| CPU 回归 | 现有 `llmoc_unit_tests` 全绿；Qwen/GLM fake 权重 logits 路径不挂 |
| 模式隔离编译烟测 | 定义 `LLMOC_EXEC_CPU_ONLY` 构建时，`exec/gpu` 源不参与链接 |
| Active-set planner | 单测：给定 top_k/expert_bytes，VRAM 临界上下分别 PASS/FAIL |
| Pure-gpu 专家槽 | 单测：两层交替 prefetch，槽复用无 UAF；miss 时同步 H2D 兜底 |
| KV rollback | 各族 snapshot/restore（投机路径） |
| 分族 pack | 每族至少一个 fake 模型 forward → 有限 logits |
| **Mesh EP** | fake MoE：`world_size=2`，logits 与 `world_size=1` 余弦 > 0.999（同 seed） |
| **Mesh TP** | fake 稠密/attn：2 卡 TP vs 1 卡，余弦 > 0.999 |
| **无 NCCL** | `world_size>1` 且强制关闭 NCCL → 启动失败，不得静默单卡 |
| **Worst-case slots** | planner 单测：top-k 全映射到同一 rank 时仍按满槽预算 |

### 10.2 性能（记录基线，不设虚假承诺）

在目标硬件记 `bench_decode_tps`：

| 模式 | 相对 pure_cpu 同量化 | 记入 |
|---|---|---|
| hybrid_gpu | Attn 占比高时应明显上升 | `docs/BENCH_BASELINE.md` 追加 2026-08-31 节 |
| pure_gpu（GLM NVFP4 / DS-Flash） | 应高于 hybrid（专家不再吃 CPU 带宽墙），除非 PCIe miss 打满 | 同上 |
| pure_gpu ×2 EP | 相对单卡 pure_gpu：吞吐升或能跑起单卡跑不起的模型；若更慢且通信打满须在基线注明 | 同上 |

若 pure_gpu ≤ hybrid 且 PCIe util 低 → **判定 GPU 专家核未真正跑起来**（验收失败，不是「调参问题」）。

### 10.3 失败信息（用户可见）

```
E pure_gpu: active working set 18432 MB > vram budget 20480 MB - margin 512 MB (rank0)
   hint: use hybrid_gpu, add GPUs (devices.ids), reduce kv_pool_gb, or stronger expert quant

E pure_gpu mesh: world_size=2 but NCCL unavailable
   hint: install NCCL or set devices.ids: [0]
```

禁止：静默退回 CPU 或静默砍成单卡却仍打印 `mode=pure_gpu mesh=ep`。

---

## 11. 文档与代码同步义务

> **实现进度（未交付项）**：见 [`docs/IMPLEMENTATION_GAP_2026-08-31.md`](IMPLEMENTATION_GAP_2026-08-31.md)（相对 §10 门禁逐项 🟢/🟡/🔴）。

落地完成后必须同步改写（可在同一变更集）：

| 文件 | 动作 |
|---|---|
| `docs/ARCHITECTURE.md` §3.1 | 纯 GPU 行改为本文 §2；增补单机多卡一句指针 |
| `docs/MODEL_GLM53_FLASH.md` §2 / §5 | pure_gpu 激活驻留 + `devices` 示例 |
| `docs/IMPLEMENTATION.md` | 目录树改为 `exec/` + `families/` + mesh；注明以本文为准 |
| `README.md` 模式表 | 删除「整模进显存」；注明单机多卡 `devices` |
| `configs/engine_glm_*.yaml` | `devices` + `pure_gpu` 子配置注释 |

本文文件名即索引：`IMPLEMENTATION_2026-08-31_三模独立-MoE激活GPU-分族KV.md`。

---

## 12. 确认清单（编码前冻结）

- [x] 三模为**独立模块**，只共享 `contracts/`
- [x] Hybrid = GPU attn/KV，CPU experts
- [x] Pure-GPU = GPU 全算 + **MoE 激活专家 VRAM 槽**，不用总参判死刑
- [x] Pure-CPU 零回归硬约束
- [x] 四族 pack + 正确 cache（DeltaNet / latent MLA-CSA / KDA+DSA / Gated MLA）
- [x] **不分阶段**：上列 G1–G8 同一次实现交付；过程中可按 PR 拆文件，但**无「先只做 M5 透传」类半成品验收**
- [x] Kimi-K3 **单卡** pure_gpu 默认拒绝；**多卡 EP/ep_tp** 可拟合则允许
- [x] **引擎内单机多卡**（G8）：EP 为主、TP 为辅；多机不做

---

## 13. 实现顺序建议（仅施工顺序，不是分期目标）

> 以下为降低合并冲突的**编码次序**；每一项完成前整体仍视为「未交付」。验收只认 §10 全部门禁。

1. 落地 `contracts/`（含 `device_mesh` / `collective`）+ `IExecBackend` 工厂；`ModeController` 解析 `devices`。  
2. 抽出 `exec/cpu`（行为对齐现网 pure_cpu）。  
3. 实现 `exec/gpu` 单卡专家槽 + NVFP4/INT4 GPU；`exec/hybrid` 接线。  
4. `mesh_nccl` + `ep_dispatch` + TP all-reduce；planner 每 rank 预算。  
5. `families/*`：Qwen（含 TP）→ GLM（含 EP）→ DeepSeek → Kimi（单卡 reject / 多卡求解）。  
6. 更新配置/文档/ bench 基线；删掉共享 hot path 上的模式透传。

---

**签字栏（产品确认）**

| 角色 | 确认 |
|---|---|
| 设计目标 | 2026-08-31：一次实现全部目标；同日补 **引擎内单机多卡 (G8)** |
| 实现依据 | 本文全文；与本文冲突的旧里程碑文档一律以本文为准 |
