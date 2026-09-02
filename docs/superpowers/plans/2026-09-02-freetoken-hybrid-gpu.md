# FreeToken 式混合 GPU 推理引擎改造计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**改造目标 (Goal)：** 把 llm-on-cpu 改造为 FreeToken 式 GPU 主导的混合推理引擎，在 8GB 显存 + 32GB 内存单机上实现：

1. **dense 量化小模型**（Qwen3.5-4B INT4，权重 ~2.2 GiB）：量化权重**原样常驻显存**，GPU 原生 dequant-GEMV，decode ≥ 30 tok/s（预期 40–80）。
2. **MoE Flash 大模型**（27B-A3B / 35B-A3B / Qwen3.8-Next-Flash / GLM5.3-Flash / DeepSeek-V4-Flash / Kimi-K3，总参数超显存）：共享权重 + 热专家 GPU 常驻，RAM 冷专家**预测性异步预取**与计算重叠，decode ≥ 30 tok/s。
3. 保持本项目约束：**no-nvcc**（运行时 NVRTC 动态 JIT 编译 CUDA kernel）、动态加载 cudart/cublas/nvcuda/nvrtc 四个 DLL、LWC/QLWC 权重格式与 OpenAI 风格 HTTP API 不变。

**架构：** 权重以量化形态（INT4 packed / NVFP4 packed + scales）直接驻留 VRAM；新增 NVRTC 运行时编译的 dequant-GEMV/GEMM kernel 在 GPU 上执行"按 group 反量化 + 矩阵乘"；dense 模型全量上卡（pure_gpu）；MoE 模型共享权重上卡、专家在 RAM 用 ExpertSlotPool (LRU) + 异步双缓冲预取；残差流全程驻留设备，每 token 仅 logits D2H 一次。

**Tech Stack：** C++20, MSVC, CUDA Runtime/Driver API（动态加载 cudart64/cublas64/nvcuda/nvrtc64），NVRTC JIT，cuBLAS，OpenMP，现有 QlwcStore / ExpertSlotPool / prefetch_pipeline。

---

## 现状诊断（改造前的关键事实，已核实）

- `src/hal/cuda_backend.cpp:241-262` `ensure_int4_device`：INT4 在 **host 反量化成 FP32** 后上传 → 4B 模型 FP32 化 ≈ 16 GiB，8G 显存放不下；`g_cache` 无淘汰，预算占满后新权重永久 `return nullptr` 回退 CPU。**这是 hybrid_gpu 慢的根因。**
- `src/hal/cuda_backend.cpp:82` `kMaxGpuInt4Rows = 65536`：词表 248320 行的 **lm_head 被硬编码排除在 GPU 外**，每 token 在 CPU 做 2560×248320 的 GEMV。
- `src/hal/cuda_backend.cpp:202-214` `gemm_dev`：每次 GEMM 同步 `cudaMemcpy H2D → cublasSgemm → memcpy D2H` + 全局互斥锁，逐层串行。
- 已有可复用骨架：`src/hal/quant_views.h`（AwqView/Nvfp4View）、`src/weights/qlwc_store`（Int4View/PassView）、`src/exec/gpu/residency_gpu.{h,cpp}`（ExpertSlotPool LRU）、`src/weights/prefetch_pipeline`、`src/glm/glm_expert_prefetch.cpp`、`tests/unit/test_gpu_quant_gemm.cpp`（TINY_TEST 风格）、`tools/nan_probe`（logits 探针）、`build/msvc-x64/bin/int4_gemm_bench.exe`。
- **P0 前置正确性问题**：修复 F32/BF16 权重后模型 logits 正常（`nan_probe` bad=0），但生成输出退化为空白（BF16 与 INT4 复现一致）。速度与正确性相互独立，但**验收以"正确输出 ≥30 tok/s"为准**，故 Chunk 0 必须先修。

---

## Chunk 0: 度量基线与正确性门槛

**目标：** 先让 4B INT4 能"正常对话"，并建立可重复的 tok/s 度量，所有后续改动以数字验收。

### Task 0.1: 修复生成空白问题（P0 门槛）

**Files:**
- Inspect: `src/model/qwen3_5_int4_model.cpp:660-940`（linear_attn forward / logits）、`src/model/generate.cpp`、`src/model/tokenizer_hf.cpp`（chat template）
- Test: `tests/unit/`（新增断言用 `tools/nan_probe/main.cpp` 扩展）

- [ ] **Step 1: 复现并定位。** 用 `chat_stream.mjs` 确认 `{"user":"hi"}` 在 max_tokens=64 下全为空白；`nan_probe` 已显示 top1=198(newline)。怀疑点：chat template 的 assistant 引导段与 tokenizer 特殊 token id 不一致，或 linear_attn(gated delta) 初始态/归一化错误。先在 `tokenizer_hf.cpp` 打印 apply_chat_template 后的 token ids 与特殊 token 表（im_start/im_end/assistant）逐一比对。
- [ ] **Step 2: 修复后回归。** 期望：`chat_stream.mjs 8081 "What is 2+2?"` 输出非空白、含 "4" 的连贯文本。若确认是 chat template 问题，改 template；若是 forward 问题（如 delta recurrence 数值），定位到具体层用 nan_probe 逐层打印隐藏态范数。
- [ ] **Step 3: Commit.**

### Task 0.2: 建立 tok/s 度量工具

**Files:**
- Create: `tools/toks_bench.mjs`（计时 server 生成 N tokens 的脚本，可复用 `chat_stream.mjs` 结构）
- Run: `build/msvc-x64/bin/llmoc_server_int4.exe --config configs/engine_int4.yaml`

- [ ] **Step 1: 写 `tools/toks_bench.mjs`：** 预热后发送固定 prompt，计时 `max_tokens=128, stream=true`，统计 `completion_tokens / elapsed`，打印 tok/s（区分 TTFT 与 decode 速度）。
- [ ] **Step 2: 记录基线。** pure_cpu / hybrid_gpu / pure_gpu 三种模式各跑 3 次取中位数，写入 `docs/superpowers/plans/baseline.md`。期望：pure_cpu ~15-25 tok/s，hybrid_gpu 因 FP32 膨胀与现状接近。
- [ ] **Step 3: Commit.**

---

## Chunk 1: NVRTC 运行时 + GPU 原生 INT4 dequant-GEMV kernel

**目标：** 不再把权重反量化成 FP32。上传 packed INT4 + scales 到 VRAM，kernel 内按 group dequant 并 GEMV。符合 no-nvcc 约束：运行时用 NVRTC 编译，driver API 启动。

**Design（kernel 语义，与 CPU 参考 `glm_awq_int4_ops` 一致）：**
- `gemv_awq_int4(const uint8_t* qweight, const uint16_t* scales, const float* x, float* y, int M, int K, int gs)`：`blockIdx.x = m`（一行一个 block），线程沿 K 并行点积；`nib = (qweight[m*(Kp/2) + k/2] >> ((k&1)?4:0)) & 0xF`；`w = (nib - 7) * f16(scales[m*ng + k/gs])`（AWQ 对称，offset 7）；归约写入 `y[m]`。
- `gemv_nvfp4(...)`：FP8(e4m3) scale × global_scale，同理。
- 预先用小矩阵与 CPU 参考对拍（复用 `tests/unit/test_gpu_quant_gemm.cpp` 的构造）。

### Task 1.1: NVRTC + driver API 动态加载层

**Files:**
- Modify: `src/hal/cuda_backend.h`（Api struct 增 nvrtc/nvcuda 函数指针）、`src/hal/cuda_backend.cpp:123-180`（load_apis 扩展）
- Test: `tests/unit/test_gpu_quant_gemm.cpp`

- [ ] **Step 1: 扩展 `load_apis`。** 动态加载：`nvcuda.dll`（`cuInit`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuCtxSetCurrent`）与 `nvrtc64_130_0.dll`/`nvrtc64_120_0.dll`/`nvrtc64_110_0.dll`（`nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTX`, `nvrtcGetPTXSize`, `nvrtcGetProgramLog`, `nvrtcDestroyProgram`）。找不到任一 → `probe_available()` 返回 false（优雅降级 CPU），`g_status` 记录原因。
- [ ] **Step 2: 写 JIT 封装。** `static CUfunction jit(const char* src, const char* name)`：NVRTC 编译（`--gpu-architecture=compute_XX` 按 `cudaGetDeviceProperties` 或默认 80）、driver 加载、返回 kernel handle，失败返回 nullptr 并记录 log。
- [ ] **Step 3: 单测：** 无 GPU 环境下 `llmoc_unit_tests.exe` 不崩（probe false → skip）；有 GPU 时 jit 一个 `__global__ void add1(float* x){x[0]+=1;}` 并验证。

### Task 1.2: INT4 AWQ GEMV kernel + 对拍

**Files:**
- Create: `src/hal/cuda_kernels.h`（kernel 源字符串常量）+ `src/hal/cuda_quant_gemm.{h,cpp}`（`try_gemv_int4(const float* x_dev_or_host, const Int4View&, float* y)` 薄封装）
- Modify: `src/hal/cuda_backend.cpp`（挂接 jit 编译入口）
- Test: `tests/unit/test_gpu_quant_gemm.cpp`（新增 `TINY_TEST(GpuQuant, AwqGemvJitMatchesCpu)`）

- [ ] **Step 1: 写失败测试。** 构造 M=24, K=128, gs=128 的 AWQ 数据；CPU 参考用 `glm::hal::gemm_awq_int4`；调用新 `try_gemv_int4`，断言逐元素误差 < 1e-3·(1+|cpu|)。
- [ ] **Step 2: 实现 kernel 源**（上文 Design 语义，带 `__shfl_down_sync` warp 归约，block 256 线程）+ jit + 启动封装。
- [ ] **Step 3: 跑通测试。** 再补 NVFP4 变体测试与实现（FP8 e4m3 → float 解码：`(h>>3)&0xF` exp, `h&0x7` man）。
- [ ] **Step 4: Commit.**

### Task 1.3: INT4 权重驻留（预算按量化字节计）

**Files:**
- Modify: `src/hal/cuda_backend.cpp:233-263`（`ensure_int4_device` 重写）
- Test: `tests/unit/test_gpu_quant_gemm.cpp`

- [ ] **Step 1: 重写 `ensure_int4_device` → `ensure_int4_resident`：** 上传 `qweight`（packed 字节）+ `scales`（+zeros, GPTQ 时），`nbytes = M*Kp/2 + 2*M*ng (+2*M*ng)`；预算检查用**量化字节**而非 FP32；缓存 key 仍为 `W.qweight`。
- [ ] **Step 2: `try_gemm_int4` 改为走 GEMV kernel**（host x → 临时 d_x H2D → kernel → D2H y）。保留旧 FP32 路径作为无 NVRTC 时的 fallback。
- [ ] **Step 3: 验证预算：** 4B 模型 `warm_gpu_int4_weights()` 后 `vram_used()` ≈ 2.5–3 GiB，全部 355 个 INT4 张量命中 GPU（日志 `ok/fail` 计数 = 355/0）。
- [ ] **Step 4: Commit.**

### Task 1.4: GEMV 性能基准

**Files:**
- Modify: `tools/int4_gemm_bench`（或新增 `bench/gemv_bench.cpp` 对比 CPU vs GPU kernel 吞吐）

- [ ] **Step 1: 基准：** M∈{2560, 10240, 248320(lm_head)}, K=2560, 1000 次迭代，打印 GB/s 与 GFLOP/s，CPU vs GPU 对比。
- [ ] **Step 2: 验收：** GPU kernel 对 2560×248320 lm_head 单次 < 1ms（读 ~318MB INT4 → 需 ≥ 320 GB/s 有效带宽）。
- [ ] **Step 3: Commit.**

---

## Chunk 2: dense pure_gpu 全量上卡（Qwen3.5-4B ≥30 tok/s）

**目标：** 4B dense 模型所有权重（INT4 + passthrough BF16 + lm_head）驻留 8G 显存，decode 全程 GPU，残差留在设备。

### Task 2.1: passthrough 权重上卡

**Files:**
- Modify: `src/hal/cuda_backend.{h,cpp}`（新增 `upload_pass_bf16(name, data, n)` 常驻表）
- Modify: `src/model/qwen3_5_int4_model.cpp:305-326`（gemv/gemm 封装）

- [ ] **Step 1: BF16 passthrough（embedding/norm/A_log/dt_bias/conv1d，共 383 个，~0.7 GiB）上卡**，作为设备指针常驻；embedding lookup / norm 仍可在 host 做（数据量小）或后续 kernel 化。
- [ ] **Step 2: 验证：** 加载 4B 后 `vram_used()` ≤ 4 GiB。

### Task 2.2: lm_head GEMV 上 GPU

**Files:**
- Modify: `src/hal/cuda_backend.cpp:82,235`（INT4 kernel 路径不受 `kMaxGpuInt4Rows` 限制；FP32 fallback 保留限制）、`src/model/qwen3_5_int4_model.cpp:812,934`（lm_head 调用改走 GPU GEMV）

- [ ] **Step 1: lm_head(248320×2560) INT4 GEMV 上 GPU；** 仅取 logits 处 D2H 一次（248320 floats = ~1MB）。
- [ ] **Step 2: 对拍：** nan_probe 的 top5 与 CPU 完全一致（id 与 logit 值容差内）。
- [ ] **Step 3: Commit.**

### Task 2.3: 残差驻留设备 + 逐层零拷贝

**Files:**
- Modify: `src/model/qwen3_5_int4_model.cpp:300-326,660-940`（forward 主循环）

- [ ] **Step 1: 残差 `h` 常驻设备。** 每层：层输入已在设备 → GEMV kernel（输入输出全为设备指针，**消除逐层 H2D/D2H**）；norm/残差加等 elementwise 用小 kernel 或 host（数据 2560 floats 极小，先 host 也可接受，每层一次 10KB 拷贝开销可忽略）。
- [ ] **Step 2: 量测：** `tools/toks_bench.mjs` pure_gpu 模式。
- [ ] **Step 3: 验收门槛：** `What is 2+2?` 输出正确且 **decode ≥ 30 tok/s**。不达则 profile（逐层计时）定位热点。
- [ ] **Step 4: Commit.**

### Task 2.4: linear_attn recurrent 路径评估

**Files:**
- Inspect: `src/model/qwen3_5_int4_model.cpp:660-700`（`gated_delta_recurrent`）

- [ ] **Step 1:** 若 recurrent 成为瓶颈（每层串行、隐藏态小）：先留在 CPU（状态 2560 维极小，H2D/D2H 每 token 每层 ~10KB 可接受）；若 profile 显示 >15% 时间，写 CUDA recurrent kernel。
- [ ] **Step 2: Commit.**

---

## Chunk 3: hybrid_gpu —— 超显存模型的分层驻留 + 异步预取

**目标：** 当模型（或工作集）超过显存预算时，热层 GPU、冷层 CPU INT4 GEMV，且**加载与计算重叠**；有淘汰策略。

### Task 3.1: 权重缓存加 LRU 淘汰

**Files:**
- Modify: `src/hal/cuda_backend.cpp`（g_cache 加 LRU 链表 + `evict_one`）

- [ ] **Step 1:** 预算占满时淘汰最久未用条目（cudaFree），替代现在"满则永久 nullptr"。
- [ ] **Step 2: 单测：** 小预算（如 256MB）下强制多轮淘汰，断言无泄漏（`vram_used()` 不超预算）且结果正确。
- [ ] **Step 3: Commit.**

### Task 3.2: 分层驻留策略

**Files:**
- Modify: `src/sched/placement_planner.cpp`（dense 模型按层序填充 VRAM；层0..N-1 依次上卡直到预算）
- Modify: `src/model/qwen3_5_int4_model.cpp`（每层 gemv 封装自动 GPU/CPU 分流，已具备 `try_gemm_int4` fallback 结构）

- [ ] **Step 1:** 预算内层 → GPU INT4 GEMV；预算外层 → CPU `hal::gemm_int4`。分流已天然存在，只需预算统计准确（INT4 字节）。
- [ ] **Step 2: 量测：** 人为把预算设为 1 GiB 模拟超显存，验证混合模式吞吐介于 pure_cpu 与 pure_gpu 之间且正确性不变。
- [ ] **Step 3: Commit.**

### Task 3.3: 异步双缓冲预取（层流式）

**Files:**
- Modify: `src/weights/layer_stream.cpp:137`（`upload_layer_cuda`）、`src/weights/prefetch_pipeline.cpp`
- Test: `tests/unit/test_layer_stream.cpp`

- [ ] **Step 1:** 用第二个 CUDA stream 做 H2D 拷贝 + pinned host buffer，预取第 N+1 层与第 N 层计算重叠；`cudaMemcpyAsync` + event。
- [ ] **Step 2: 量测：** layer_stream 模式（window=2）下 27B-dense 模拟，tok/s 对比（预取开/关）。
- [ ] **Step 3: Commit.**

---

## Chunk 4: MoE Flash 支持（27B-A3B / 35B-A3B / 各 Flash family）

**目标：** MoE 模型共享权重 GPU 常驻、专家 RAM 驻留 + 预测预取，达到 ≥30 tok/s。family 适配：qwen3.8-next-flash（复用 qwen3_5 骨架 + MoE MLP）、glm5.3-flash（复用 `glm_flash_model`）、deepseek-v4-flash（`ds_stub_model` → 真实）、kimi-k3（`kimi_stub_model` → 真实）。

### Task 4.1: MoE 运行时：专家槽位池接入 GPU GEMV

**Files:**
- Modify: `src/exec/gpu/exec_gpu.cpp:34-67`（prefetch/pin）、`src/exec/gpu/residency_gpu.cpp`（slot 上传走 INT4 kernel 所需 packed 形态）
- Modify: `src/model/moe_model.cpp`（专家 GEMV 调用走 `try_gemv_int4`）

- [ ] **Step 1:** 专家以 packed INT4 在 slot 驻留（不再 FP32 化）；专家 GEMV 复用 Chunk 1 kernel。
- [ ] **Step 2: 单测：** 构造 8 专家小模型对拍 CPU。
- [ ] **Step 3: Commit.**

### Task 4.2: 预测性专家预取（FreeToken 核心机制）

**Files:**
- Modify: `src/glm/glm_expert_prefetch.cpp`、`src/weights/prefetch_pipeline.cpp`

- [ ] **Step 1: 下一层预取：** 第 N 层 gate 计算后立即用 gate top-k 结果预测第 N+1 层专家（MoE 相邻层专家共现率高），异步 H2D 入 slot 池。
- [ ] **Step 2: 双缓冲：** slot 池分 A/B 半区，当前层与下一层各一半，避免相互驱逐。
- [ ] **Step 3: 命中率统计：** 日志输出 prefetch 命中率（FreeToken 论文级指标），目标 ≥ 90%。
- [ ] **Step 4: Commit.**

### Task 4.3: Qwen3.8-Next-Flash family

**Files:**
- Create: `src/families/qwen3_8_next/flash_model.{h,cpp}`（以 `qwen3_5_int4_model` 为模板，MoE MLP 段）
- Create: `tools/convert_lwc` 适配其权重名映射（experts 命名）
- Test: `tests/unit/`（构造微缩 MoE 权重对拍 logits）

- [ ] **Step 1–4: TDD：** 微缩权重 → CPU 前向正确 → GPU GEMV 一致 → 预取命中 → 全量权重 bench。

### Task 4.4: GLM5.3-Flash / DeepSeek-V4-Flash / Kimi-K3 family

**Files:**
- Modify: `src/glm/glm_flash_model.cpp`（GLM5.3）、`src/families/deepseek_v4/ds_stub_model.cpp`（→ 真实前向）、`src/families/kimi_k3/kimi_stub_model.cpp`（→ 真实前向）

- [ ] **Step 1:** 每个 family 先跑通 CPU 参考前向（现有 stub 升级），再接 Chunk 1/4.1/4.2 的 GPU/预取路径。
- [ ] **Step 2: Commit（每 family 独立）。**

---

## Chunk 5: 集成验收

### Task 5.1: 端到端基准矩阵

**Files:**
- Create: `docs/bench/freetoken_matrix.md`

- [ ] **Step 1:** 对每个可用模型 × {pure_cpu, hybrid_gpu, pure_gpu} 跑 `tools/toks_bench.mjs`，记录 decode tok/s、TTFT、VRAM/RAM 占用、prefetch 命中率。
- [ ] **Step 2: 验收标准：**
  - Qwen3.5-4B INT4 pure_gpu：≥ 30 tok/s，输出正确（2+2=4 等）。
  - MoE Flash（若模型可用）：hybrid_gpu ≥ 30 tok/s，prefetch 命中率 ≥ 90%。
  - 无 CUDA 环境自动降级 pure_cpu 且不崩溃。
- [ ] **Step 3: Commit.**

---

## 风险与开放问题

1. **GPU 型号未知** → 30 tok/s 的把握基于消费级 ≥ ~300 GB/s 显存带宽的卡；先在 Chunk 0 用 `cudaGetDeviceProperties` 记录实际带宽，若 < 200 GB/s 需重估目标。
2. **P0 正确性未修** → 全部速度工作前先完成 Chunk 0 Task 0.1，否则"快"没有意义。
3. **NVRTC/driver 版本矩阵** → nvrtc64 版本号随 CUDA 版本变化，加载按 13.0/12.x/11.0 依次探测（Task 1.1 已含）。
4. **27B dense 类模型** → 带宽数学决定其**不可能**在 PCIe 上达到 30 tok/s，仅 MoE-Flash 形态可达，需与用户确认目标清单里哪些是 MoE。
5. **NVFP4 native（Blackwell）** → 若有 RTX 50 系，可加 native FP4 tensor-core kernel 作后续优化（非本次必须）。
