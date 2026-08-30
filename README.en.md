# llm-on-cpu

[中文](README.md) | [English](README.en.md)

**CPU LLM Inference** · **Sparse MoE** · **BF16 / INT4** · **Qwen3** · **OpenAI-Compatible API** · **AMX** · **Windows / Linux / macOS**

Pure-CPU large language model inference engine — run sparse MoE LLMs in **BF16 without quantization** on a 64GB-RAM host.  
**Current delivery target: Qwen3.8-27B** (full BF16 ~54–61GB fits entirely in 64GB DRAM, zero page faults;  
expected ~55 tok/s raw, 80+ tok/s with MTP; see [docs/MODEL_QWEN3.8-27B.md](docs/MODEL_QWEN3.8-27B.md)).  
One engine, three execution modes: **pure CPU by default**, optional consumer-GPU hybrid (8–24GB), or pure GPU.  
DeepSeek-V4-Flash remains a longer-horizon large-model scenario (NVMe streaming; see architecture docs).

> Four-phase plan: [DESIGN_P0_P3](docs/DESIGN_P0_P3.md) · [IMPLEMENTATION_P0_P3](docs/IMPLEMENTATION_P0_P3.md) (**no coding until confirmed**).

## UI preview

Local chat UI (`http://127.0.0.1:15085/`): text chat and image+text multimodal.

<p align="center">
  <img src="images/text_chat.jpg" alt="llm-on-cpu text chat" width="720" />
</p>

<p align="center"><em>Text chat · Markdown rendering</em></p>

<p align="center">
  <img src="images/image_chat.jpg" alt="llm-on-cpu multimodal chat" width="720" />
</p>

<p align="center"><em>Image understanding · multimodal input</em></p>

## Core mechanisms (why pure CPU works)

```
T_token = max( compute, total_bytes/DRAM_bw, fault_bytes/NVMe_bw )   ← pipeline overlap → max
effective_speed = raw_speed × MTP_accept_multiplier                  ← bandwidth-wall multiplier
```

| Mechanism | Implementation |
|---|---|
| Expert-level 3-tier residency (pinned / LRU hot / NVMe cold) | `weights/weight_manager` |
| Async prefetch of next-layer experts after gate; double-buffer | `weights/prefetch_pipeline` |
| Pin hot experts (routing skew payoff) | same `Config.pinned` |
| MTP self-speculative decode verify loop | `model/spec_decode` |
| LWC weight container (expert-grain, 4K-aligned direct I/O) | `weights/lwc_format` |
| Async I/O: thread pool (3 OS) / io_uring+O_DIRECT (Linux) | `weights/io_engine*` |
| AMX BF16 tile microkernels (SPR Linux) | `hal/amx_gemm` |

## Platforms ([docs/PLATFORM.md](docs/PLATFORM.md) full matrix)

| | Windows | Linux | macOS |
|---|---|---|---|
| Role | Dev machine | **Primary production** (SPR) | Logic-layer validation |
| Status | ✅ End-to-end proven | ✅ Code ready; CI awaits bare metal | Same tree builds |
| AMX / io_uring | N/A → auto-fallback | ✅ | N/A |

## Quick start

### CI / prebuilt packages (GitHub Actions)

Pushes to `main`/`master` and PRs build **Windows / Linux / macOS** and upload app + source zips (Artifacts).  
Tags `v*` also create a GitHub Release with:

| Package | Contents |
|---|---|
| `llm-on-cpu-<ver>-windows-x64.zip` | `llmoc_server[.exe]` / `llmoc_server_int4` + configs |
| `llm-on-cpu-<ver>-linux-x64.zip` | same (built with liburing) |
| `llm-on-cpu-<ver>-macos-arm64.zip` | same (logic/API validation) |
| `llm-on-cpu-<ver>-src.zip` | full source (no models/build) |

Unzip and follow `RUN.txt`. Default port **15085**. Locally: `bash scripts/package_app.sh` / `bash scripts/package_source.sh`.

### Windows (dev, MSVC 2022 Build Tools)

```powershell
cmd /c scripts\configure_dev.cmd   # CMake (auto vcvars + Ninja)
cmd /c scripts\build_dev.cmd       # incremental build
.\build\msvc-x64\bin\llmoc_unit_tests.exe
```

### Linux (production target; or container)

```bash
./scripts/build_linux.sh                                  # bare metal (needs liburing-dev)
docker build -t llmoc-ci -f scripts/Dockerfile.ci . && docker run --rm llmoc-ci
```

### macOS

Same as Linux (clang): `./scripts/build_linux.sh` (logic-layer validation only).

### Models: place & convert (one command)

```
models/<name>-hf/   ← HF raw weights      models/<name>.lwc   ← engine only loads this
```

Download / convert to LWC / verify are three low-level tools (network · layout · C++ checksum). Day-to-day, chain them with one command:

```powershell
# Build once first (produces lwc_verify); Windows: configure_dev + build_dev
node tools/prepare_model.mjs --model Qwen/Qwen3.8-27B
# e.g. node tools/prepare_model.mjs --model Qwen/Qwen3.5-4B --prune-hf

# New model: download → convert → verify → prune large HF tensors (keep config/tokenizer)
node tools/prepare_model.mjs --model Qwen/Qwen3.5-4B --prune-hf

# Disk cleanup only:
node tools/prepare_model.mjs --model Qwen/Qwen3.5-4B --skip-download --skip-convert --prune-hf
# bf16 → int4 (Python / Node equivalent; embeddings pass through. Python is usually faster for large models)
python tools/quantize_int4.py --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method gptq
node tools/quantize_int4.mjs --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method gptq
```

Pipeline: `download` → `convert` → `lwc_verify --update` (cross-check config if present) → final check → **optional HF cleanup**.
- `--prune-hf`: after verify, delete `*.safetensors` under `*-hf` (~half disk saved); keep `config.json` / tokenizer  
- `--remove-hf`: remove entire `*-hf` directory (more disk saved)  
- Weights already local: `--skip-download`; convert done, cleanup only: `--skip-download --skip-convert --prune-hf`  

Step-by-step flags: **[docs/USAGE.md](docs/USAGE.md)**.

### After convert: start the API server

```powershell
# Build required; keep tokenizer.json + config.json beside weights (--prune-hf OK; do not --remove-hf)
cmd /c scripts\build_dev.cmd
.\build\msvc-x64\bin\llmoc_server.exe --config configs\engine.yaml
```

In another terminal (CPU reference path; first token can be slow; ~10GB+ RAM for 4B BF16):

Chat UI: `http://127.0.0.1:15085/` (see [UI preview](#ui-preview) above). Or call the API:

```powershell
# Health
Invoke-RestMethod http://127.0.0.1:15085/healthz

# Non-streaming chat
Invoke-RestMethod http://127.0.0.1:15085/v1/chat/completions -Method Post -ContentType 'application/json' -Body '{"model":"default","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'

# SSE streaming
curl.exe -N http://127.0.0.1:15085/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"max_tokens\":32}"

# Prometheus metrics
Invoke-WebRequest http://127.0.0.1:15085/metrics | Select-Object -Expand Content
```

Optional auth: set `LLMOC_API_KEY`, send `Authorization: Bearer ...`. Config: [`configs/engine.yaml`](configs/engine.yaml).

| Capability | Status |
|---|---|
| Download / convert LWC / verify / prune | ✅ |
| OpenAI-compatible `/v1/chat/completions` (non-stream + SSE) | ✅ `llmoc_server` |
| **INT4** BF16→QLWC quant + inference | ✅ `quantize_int4.py` / `.mjs` + `llmoc_server_int4` (separate path) |
| **Qwen3.5 hybrid** (linear + full attn) CPU chat | ✅ Auto-detect (no LWC expert groups) |
| **MoE** (Qwen3.8 etc.) gate top-k + ExpertPrefetcher | ✅ Auto-detect (has expert groups) |
| `/healthz` · `/metrics` · radix · schedule queue | ✅ |
| Real MTP wiring / op-level continuous batch | ⏳ |
| hybrid-gpu / pure-gpu | ✅ M5 (`mode: hybrid_gpu`/`pure_gpu`/`auto`; needs cudart+cublas) |

`llmoc_server` picks the backend from whether `.lwc` has expert groups — no code change:

```yaml
# configs/engine.yaml — Qwen3.5-4B
model:
  path: models/Qwen3.5-4B.lwc

# MoE example (after convert, point path / tokenizer at *-hf sidecar)
# model:
#   path: models/qwen3827.lwc
```

Full guide: **[docs/USAGE.md](docs/USAGE.md)** · model budget: **[docs/MODEL_QWEN3.8-27B.md](docs/MODEL_QWEN3.8-27B.md)**

## Tests & benchmarks

Binaries live in the build output (Windows: `.\build\msvc-x64\bin\`; Linux/macOS: `./build/release/bin/`).

```powershell
$BIN = ".\build\msvc-x64\bin"   # Linux/macOS: BIN=./build/release/bin

# Unit tests
& "$BIN\llmoc_unit_tests.exe"

# API server
& "$BIN\llmoc_server.exe" --config configs\engine.yaml

# INT4 server (quantize_int4.py / .mjs → .qlwc first; embeddings must pass through)
& "$BIN\llmoc_server_int4.exe" --config configs\engine_int4.yaml

# INT4/BF16 GEMM microbench (AVX2+OpenMP; decode upper bound, not e2e 30 tok/s)
& "$BIN\int4_gemm_bench.exe" --iters 30

# M0 platform profile: DRAM/NVMe bandwidth + ISA/FLOPS probe
& "$BIN\m0_bandwidth.exe" --gb 2 --out hw_profile.json
& "$BIN\m0_isa.exe"

# M2 double-buffer prefetch A/B — synthetic or real converted model
& "$BIN\m2_pipeline_bench.exe" --layers 8 --experts 32 --part-mib 2 --topk 8
& "$BIN\m2_pipeline_bench.exe" --file models\v4flash.lwc --topk 8

# M3 speculative-decode multiplier table (α × k → tokens/step → predicted throughput)
& "$BIN\m3_spec_table.exe" --base 51.67

# LWC verify (required after convert)
& "$BIN\lwc_verify.exe" models\v4flash.lwc --update
& "$BIN\lwc_verify.exe" models\v4flash.lwc
```

### Measured numbers (dev box i7-14650HX, not SPR target)

| Item | Value |
|---|---|
| M2 prefetch pipeline speedup | **1.69x** (sync 30.5 → 51.7 tok/s, io_wait 160→76ms) |
| M3 multiplier prediction | α=0.7, k=3 → **2.53x** |
| DRAM / NVMe direct read | 68 GB/s / 2339 MB/s |
| AMX | Not on consumer HX; microkernels gated by selftest on SPR CI |
| INT4 GEMM (AVX2+OpenMP, M=9216×K=2560) | ~70 GFLOP/s (after opt; was ~13) |
| Local e2e decode (INT4 4B, warm) | a few tok/s; **30 tok/s target = SPR+AMX+MTP** |

## Layout

```
src/
├─ common/    platform / logging / CPUID / aligned alloc
├─ weights/   LWC · I/O engine (thread pool + io_uring) · 3-tier residency · prefetch
├─ model/     MTP speculative decode · Qwen3.5 forward · tokenizer · generate
├─ hal/       DeviceHAL + CpuOps (GEMM / attn / DeltaNet)
├─ sched/     request queue / mode-controller
├─ kv/        radix prefix pool (thin wrapper)
└─ server/    OpenAI-compatible API / metrics → llmoc_server
tools/        convert_lwc · prepare_model · lwc_verify · m0/m2/m3 benches
tests/unit/   unit tests (tokenizer/config included)
models/       weights (gitignore): <name>-hf/ raw + <name>.lwc converted
docs/         ARCHITECTURE.md · IMPLEMENTATION.md · PLATFORM.md · USAGE.md
scripts/      Windows CMD / Linux sh / CI Dockerfile / fake-model gen
```

## Milestone status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Platform baseline + ISA probe | ✅ Windows measured; AMX selftest awaits SPR |
| M1 | LWC + I/O engine + 3-tier residency | ✅ Thread-pool; io_uring built-in, Linux field test pending |
| M2 | Double-buffer prefetch pipeline | ✅ **1.69x measured** |
| M3 | MTP verify state machine | ✅ 14/14 unit tests; weight wiring = open item O2 |
| M4 | Batching / API service | ✅ MVP: `llmoc_server` + chat + SSE + metrics |
| M5 | PlacementPlanner + modes ②③ + `hal/cuda_backend` | ✅ |

## Related docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — architecture (v0.4 frozen baseline)
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — implementation (language / modules / test plan)
- [docs/PLATFORM.md](docs/PLATFORM.md) — Windows / Linux / macOS capability matrix
- [docs/USAGE.md](docs/USAGE.md) — **usage** (model place / convert / verify / benches / limits)
