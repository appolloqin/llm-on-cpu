# P0 æ€§èƒ½åŸºçº¿

## å£å¾„ï¼ˆå†™æ­»ï¼‰

| æŒ‡æ ‡ | å®šä¹‰ | è¯´æ˜ |
|---|---|---|
| **chat_e2e_tps / e2e_tps / last_tps** | `completion_tokens / generateå¢™é’Ÿ`ï¼ˆå« prefillï¼‰ | **å¯¹è¯éªŒæ”¶ä»¥æ­¤ä¸ºå‡†**ï¼›ä¸ `llmoc_last_tps` åŒå…¬å¼ |
| **pure_decode** | prefill ååªè®¡é€æ­¥ `forward`ï¼ˆargmax åœ¨è®¡æ—¶å¤–ï¼‰ | å®éªŒå®¤ decode ä¸Šé™ï¼›**ä¸ä»¥ e2e å†’å……** |

å¤æµ‹æ¨¡æ¿ï¼ˆ**ä¸­ä½æ•° Ã—3 æ¬¡**ï¼Œwarmâ‰¥1ï¼›ä¸­é•¿å›å¤æ‘Šè–„ prefillï¼‰ï¼š

```powershell
.\build\msvc-x64\bin\bench_decode_tps.exe --config configs\engine_int4.yaml --new 256 --warm 1
# Ã—3ï¼Œå– e2e_tps ä¸­ä½æ•°ï¼›åŒæ—¶è®° pure_decode
# çŸ­èŠå¯¹ç…§ï¼šåŠ  --short --new 32
.\build\msvc-x64\bin\int4_gemm_bench.exe
```

## è®°å½•è¡¨

| æ—¥æœŸ | æœºå™¨ | é…ç½® | warm | new | e2e_tps | pure_decode | å¤‡æ³¨ |
|---|---|---|---|---|---|---|---|
| 2026-08-29 | i7-14650HX | ä¼˜åŒ–å‰ | 1 | ~16 | ~2.1 | â€” | æ—©æœŸ e2e |
| 2026-08-29 | i7-14650HX | scales/LayerPack/GDN | 1 | 32 | 2.46 | ~7.3 | æ—§ pureï¼ˆå« argmaxï¼‰ |
| 2026-08-29 | i7-14650HX | pure_decodeâ‰¥10 æ–¹æ¡ˆ | 1 | 32 | ~3.7 | **10.74** | çŸ­ EOSï¼›pure è¾¾æ ‡ |
| 2026-08-29 | i7-14650HX | **+batched GEMM prefill + last-only lm_head** | 1 | **256** | **10.32** | **~11.0** | ä¸‰æ¬¡ e2eï¼š10.35 / 10.29 / 10.32ï¼›pureâ‰ˆ11.07/10.85/11.01 |
| 2026-08-29 | i7-14650HX | åŒä¸ŠçŸ­èŠ | 1 | 32 | **~4.85** | ~10.9 | `--short` completion=9ï¼›GDN ä¸²è¡Œä¸‹ç•Œï¼Œ**ä¸**å®£ç§°çŸ­èŠ=10 |
| 2026-08-29 | i7-14650HX | åŒä¸Šä¸­é•¿ 128 | 1 | 128 | ~9.85 | ~11 | ç•¥ä½äº 10ï¼›éªŒæ”¶ç”¨ 256 |

## å†…æ ¸ä¼˜åŒ–æ‘˜è¦

| é¡¹ | æ”¹åŠ¨ | å½±å“é¢ |
|---|---|---|
| æµ‹é‡ | pure åªè®¡ forwardï¼›greedy sample OpenMP | bench / generate |
| GDN | conv_k=4ï¼›dk=dv=128 å¿«è·¯å¾„ | cpu_ops / INT4 model |
| INT4 | 4-row + **`gemm_int4_batch`**ï¼ˆæƒé‡è¡Œå¤ç”¨ n tokenï¼‰ | int4_ops |
| Prefill | layer å†… QKV/MLP/out batchï¼›`forward()` åªç®—æœ€å lm_head | qwen3_5_int4_model |
| Bench | é»˜è®¤é•¿ prompt + `--new 256`ï¼›`--short` çŸ­èŠï¼›ä¸ last_tps åŒå£å¾„ | bench_decode_tps |

## è¯šå®ç»“è®ºï¼ˆ2026-08-29ï¼‰

| å£å¾„ | æ•°å€¼ | å«ä¹‰ |
|---|---|---|
| **ä¸­é•¿èŠ e2e**ï¼ˆéªŒæ”¶ï¼‰ | **ä¸­ä½ â‰¥10**ï¼ˆ`--new 256` â†’ **10.32**ï¼‰ | å¯¹è¯å¢™é’Ÿååè¾¾æ ‡ |
| çŸ­èŠ e2eï¼ˆhiâ†’ä¸€å¥ï¼‰ | **~4.9** | prefill+GDN ä¸²è¡Œï¼›ç‰©ç†ä¸Šéš¾ç¨³ â‰¥10 |
| pure_decode | **~11** | æµå¼å‡ºå­—é—´éš”å‚è€ƒï¼›ä¿æŒä¸å›é€€ |
| GEMM-only å¾®åŸºå‡† | é e2e | å¿½ç•¥ GDN/æ•´ç½‘ |

MTP ä¿æŒ `false`ã€‚çŸ­èŠå†å†² e2e éœ€ SPR+AMX æˆ–æ›´å¼º GDNï¼Œä¸é å£å¾„æ¸¸æˆã€‚

## 2026-08-31 æé€Ÿï¼ˆpure_cpu Qwen INT4ï¼ŒHX å¼€å‘æœºï¼‰

| é¡¹ | è¯´æ˜ |
|---|---|
| OpenMP | æœªè®¾ `OMP_NUM_THREADS` æ—¶ server/bench é»˜è®¤ cap **24â†’8**ï¼ˆå‹¿ç”¨ 12ï¼‰ |
| lm_head | greedy decode èµ° `gemm_int4_argmax`ï¼Œå…å†™å…¨è¯è¡¨ + äºŒæ¬¡ argmax |
| attn | `attn_decode_one` å¤ç”¨ thread_local scores ç¼“å†² |
|  profiling | `LLMOC_PROFILE=1` åœ¨ `forward(n=1)` æ‰“å° linear/full/lm_head ms |
| æœªå®ç°ç¼ºå£ | è§ [`IMPLEMENTATION_GAP_2026-08-31.md`](IMPLEMENTATION_GAP_2026-08-31.md) |

**HX ç¬”è®°æœ¬ç°è±¡ï¼ˆ2026-08-31 å®æµ‹ï¼‰**ï¼šé•¿è·‘ decode ä¼šåœ¨ **~200msï¼ˆâ‰ˆ5 t/sï¼‰** ä¸ **~100msï¼ˆâ‰ˆ10 t/sï¼‰** é—´äº¤æ›¿ï¼Œå‡é€Ÿ **6â€“8 t/s**ï¼›ä¸»å› æ•£çƒ­/é¢‘ç‡èŠ‚æµï¼Œä¸æ˜¯ OMP å•ç‹¬èƒ½è§£ã€‚éªŒæ”¶ä»ç”¨ **bench æš–æœºå 64â€“256 token çª—å£** çœ‹ä¸­ä½æ•°ã€‚**å‹¿ç”¨ `OMP_NUM_THREADS=32`**ï¼ˆ`start_int4.cmd` æ—§ç‰ˆæ›¾é»˜è®¤ 32ï¼‰ã€‚

### MTPï¼ˆ2026-08-31ï¼ŒQwen3.5-4B INT4 @ HXï¼‰

| æ¨¡å¼ | decode t/s | mtp_alpha | ç»“è®º |
|---|---|---|---|
| greedyï¼ˆ`mtp:false`ï¼‰ | **~6.8â€“7.1** | â€” | **CPU é»˜è®¤** |
| MTP `spec_k=3`ï¼ˆ`configs/engine_int4_mtp.yaml`ï¼‰ | **~4.9â€“7.5** | **~0.28**ï¼ˆ29/105ï¼‰ | è‰ç¨¿+verify æ­¥ ~500msï¼Œæ…¢äº greedy |

- æƒé‡ï¼š`has_mtp=1`ï¼ˆqlwc å« MTP å¤´ï¼‰
- é…ç½®ï¼š`model.mtp: auto` â†’ INT4 CPU **è‡ªåŠ¨ greedy**ï¼›å¼ºåˆ¶ MTPï¼š`mtp: true` æˆ– `LLMOC_MTP=1`
- ä¼˜åŒ–ï¼š`pin_first` é”šå®š greedy é¦– tokenï¼Œå»æ‰ draft[0] å†—ä½™æ ¡éªŒ
- **â‰¥30 tok/s** ä»ä¾èµ– SPR+AMX+MTP çƒ­è·¯å¾„ä¼˜åŒ–ï¼Œéå½“å‰ HX INT4

```powershell
$env:OMP_NUM_THREADS='8'
.\build\msvc-x64\bin\bench_decode_tps.exe --config configs\engine_int4_mtp.yaml --new 64 --warm 0
```

## P0 éªŒæ”¶å¯¹ç…§

| é¡¹ | çŠ¶æ€ |
|---|---|
| MTP / cache / bench ç­‰ P0 æ¥çº¿ | âœ…ï¼ˆè§å†å²ï¼‰ |
| **çº¯ decode â‰¥10** | âœ… |
| **ä¸­é•¿èŠ chat_e2e â‰¥10** | âœ…ï¼ˆä¸­ä½ 10.32 @ new=256ï¼‰ |
| çŸ­èŠ e2e â‰¥10 | âŒ æœªè¾¾ï¼ˆ~4.9ï¼Œå·²åˆ†åˆ—è®°å½•ï¼‰ |

## FreeToken Hybrid GPU (Task 1.x) ¡ª RTX4060, CUDA 12.5

### INT4 dequant-GEMV Î¢»ù×¼ (Task 1.4)

uild/msvc-x64/bin/int4_gemm_bench.exe --K 3072 --gs 128 --iters 50

| shape (MxK)       | CPU AVX2 (ms) | CPU GFLOP/s | GPU JIT (ms) | GPU GFLOP/s | winner |
|-------------------|--------------:|------------:|-------------:|------------:|:------:|
| 896x3072 (down)   | 0.081         | 67.9        | 0.106        | 51.9        | CPU    |
| 3584x3072 (mlp)   | 0.224         | 98.3        | 0.259        | 85.0        | CPU    |
| 9216x3072         | 0.508         | 111.5       | 0.672        | 84.3        | CPU    |
| 248320x3072 (lm)  | 11.32         | 134.6       | skipped      | -           | CPU    |

**½áÂÛ**: µ±Ç° JIT GEMV kernel (1 thread/row, µ¥ warp) ÔÚ RTX4060 ÉÏÊä¸ø AVX2 INT4¡£Ô­Òò:
- K=3072 / blockDim=256 = Ã¿Ïß³Ì½ö 12 ´Îµü´ú, SM Õ¼ÓÃÂÊ²»×ã
- INT4 unpack + FP16¡úFP32 ×ª»»ÔÚ software Àï¿ªÏú´ó, FP32 GEMV ÀÛ¼ÆÖ»ÓĞ 50-85 GFLOP/s
- GPU Â·¾¶ÏÂ»¹ÓĞÃ¿ token H2D/D2H (~20us per call)

### 4B e2e (Task 1.5, hybrid_gpu, configs/engine_int4_hybrid.yaml)

`
='8'; bench_decode_tps.exe --config configs\engine_int4_hybrid.yaml --short --new 32 --warm 2
[int4] mode=hybrid_gpu hal.cuda: enabled=1 used=1.71GiB budget=4GiB warm_gpu_int4 ok=248 fail=0
[b
...[96 chars truncated]...

...
[Pure_decode n=24 forward_ms=8466.5 argmax_ms=1.8 decode_tps=2.83 ms/tok=352.8
`

**decode 2.7-2.8 tok/s** ¡ª ±È CPU baseline (7.88) Âı 2.8x¡£

**¸ùÒò**: JIT GEMV kernel ²»Èç AVX2 INT4 + Ã¿ token H2D/D2H ¿ªÏú ~0.66ms ¡Ö Õ¼ÁËÒ»°ë forward Ê±¼ä¡£
**ĞŞ¸´·½Ïò (Task 2.x)**: kernel ÏòÁ¿»¯ (uchar4¡ú8 INT4, shared mem scales), ¼¤»î×¤Éè±¸, Öğ²ãÁã¿½±´¡£

### 4B e2e ºóĞø (Task 1.6 + 2.x)

| ÓÅ»¯µã                          | decode tok/s | ms/tok |
|---------------------------------|-------------:|-------:|
| baseline (CPU AVX2 INT4)        | 7.88         | 127    |
| hybrid_gpu + µ¥Ïß³Ì/ĞĞ kernel   | 2.71         | 369    |
| hybrid_gpu + 8ĞĞ/block + vec4   | **7.52**     | 132    |

hybrid_gpu ÒÑ×·Æ½ CPU baseline; ½øÒ»²½Òª ¡İ30 tok/s Ğè¼õÉÙ H2D/D2H + ÓÃ CUDA Graphs °Ñ 33 ´Î
GEMV ºÏ³ÉÒ»´Î·¢Éä¡£

### 4B hybrid_gpu ÏêÏ¸ profiling (Task 2.3 µ÷ÑĞ)

LLMOC_PROFILE=1 Ê± forward_to_hidden Ä©´òÓ¡Ã¿ token ºÄÊ±·Ö½â(µ¥Î» ms):

| ½×¶Î                       | µäĞÍÖµ       | ËµÃ÷                                   |
|----------------------------|------------:|----------------------------------------|
| layers_linear (27 GDN)     | 100-120     | Ã¿²ã ~3.7ms£¬º¬ 5 GEMV + conv1d + gated_delta_recurrent |
| layers_full  (5 SA)        |  30-35      | Ã¿²ã ~6.7ms£¬º¬ 4 GEMV + attn_decode_one |
| lm_head (V=248320 GEMV)   |  22-28      | CPU AVX2 INT4 fast path                 |
| ×Ü                         | 150-180     | ¶ÔÓ¦ 5.5-6.5 tok/s                      |

**lm_head GPU Â·¾¶ÊµÑé**£º³¢ÊÔÓÃ gemm_view Ìæ´ú hal::gemm_int4£¬½á¹ûÍË»¯µ½ 2.6 tok/s¡£
Ô­Òò£ºench_decode_tps Î´µ÷ warm_gpu_int4_weights() ¡ú INT4 JIT Â·¾¶Ê§°Ü ¡ú ×ß
cublasSgemm »ØÍË + 248K float D2H£¬±È CPU AVX2 INT4 Âı¡£¹Ê lm_head ±£³Ö CPU Â·¾¶¡£

**µ½ ¡İ30 tok/s µÄÆ¿¾±**£º
1. Linear ²ã CPU ¿ªÏú£¨conv1d + gated_delta_recurrent + rmsnorm + reshape£©Õ¼´óÍ·
2. Ã¿ token 5 ´Î GEMV ¡Á 8 us H2D + 10 us D2H ¡Ö 80 us£¨´ÎÒª£©
3. ĞèÒª°Ñ rmsnorm / gated_delta / silu / conv1d Ò²°áµ½ GPU ²¢ÓÃ CUDA Graphs ºÏ²¢·¢Éä
