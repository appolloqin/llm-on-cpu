#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_int4.py — BF16/F16 LWC → QLWC INT4（GPTQ/AWQ 风格布局，独立新功能）

不修改原 convert_lwc / LWC 推理路径。

用法:
  python tools/quantize_int4.py --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method awq
  node   tools/quantize_int4.mjs --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method awq
  python tools/quantize_int4.py --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method gptq --group-size 128

与 convert_lwc 一样提供 Python / Node 双栈（Node 零 npm 依赖；算法与布局对齐）。

方法说明:
  awq  — 对称 INT4 absmax，AWQ 兼容存储（无 zero）；无校准集，非完整 AWQ
  gptq — 非对称 INT4 per-group scale+zero；布局对齐 GPTQ，无 Hessian 校准

embedding / lm_head 默认透传原精度（切勿量化，否则又慢又胡话）。
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

ALIGN = 4096
QLW_MAGIC = b"QLW1"
SCHEME_GPTQ = 1
SCHEME_AWQ = 2


def fnv1a64(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h


def align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def f32_to_f16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).view(np.uint16)


def bf16_bytes_to_f32(buf: bytes, n: int) -> np.ndarray:
    u16 = np.frombuffer(buf, dtype=np.uint16, count=n)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f16_bytes_to_f32(buf: bytes, n: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float16, count=n).astype(np.float32)


def read_lwc(path: Path):
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"LWC1":
            raise SystemExit(f"not LWC1: {path}")
        ver, cat_len, cat_crc = struct.unpack("<IQQ", f.read(20))
        cat = f.read(cat_len)
        if fnv1a64(cat) != cat_crc:
            raise SystemExit("LWC catalog crc mismatch")
        raw = memoryview(cat)
        o = 0

        def u32():
            nonlocal o
            v = struct.unpack_from("<I", raw, o)[0]
            o += 4
            return v

        def u64():
            nonlocal o
            v = struct.unpack_from("<Q", raw, o)[0]
            o += 8
            return v

        def s():
            nonlocal o
            n = u32()
            out = bytes(raw[o : o + n]).decode("utf-8")
            o += n
            return out

        dtype = u32()
        align = u32()
        n_t, n_g = u64(), u64()
        tensors = []
        for _ in range(n_t):
            name = s()
            nd = u64()
            shape = [u64() for _ in range(nd)]
            off, nb, ck = u64(), u64(), u64()
            tensors.append({"name": name, "shape": shape, "offset": off, "nbytes": nb})
        # skip groups
        for _ in range(n_g):
            u32()
            u32()
            nn = u64()
            for _ in range(nn):
                s()
        payloads = {}
        for t in tensors:
            f.seek(t["offset"])
            payloads[t["name"]] = f.read(t["nbytes"])
    return dtype, tensors, payloads


def pack_int4(q: np.ndarray) -> bytes:
    """q: uint8 0..15, shape [M,K] even K preferred."""
    M, K = q.shape
    if K % 2:
        q = np.pad(q, ((0, 0), (0, 1)), constant_values=0)
        K += 1
    lo = q[:, 0::2]
    hi = q[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed.tobytes()


def quant_gptq_asym(W: np.ndarray, group_size: int):
    """W [M,K] f32 → q uint8, scales/zeros f16 bits."""
    M, K = W.shape
    assert K % group_size == 0
    ng = K // group_size
    scales = np.zeros((M, ng), np.float32)
    zeros = np.zeros((M, ng), np.float32)
    q = np.zeros((M, K), np.uint8)
    for g in range(ng):
        sl = slice(g * group_size, (g + 1) * group_size)
        block = W[:, sl]
        mn = block.min(axis=1)
        mx = block.max(axis=1)
        scale = (mx - mn) / 15.0
        scale = np.maximum(scale, 1e-8)
        zero = mn
        scales[:, g] = scale
        zeros[:, g] = zero
        qq = np.round((block - zero[:, None]) / scale[:, None])
        q[:, sl] = np.clip(qq, 0, 15).astype(np.uint8)
    return pack_int4(q), f32_to_f16_bits(scales.reshape(-1)), f32_to_f16_bits(zeros.reshape(-1))


def quant_awq_sym(W: np.ndarray, group_size: int):
    """对称 INT4（AWQ 兼容存储：无 zero）。无校准集时用 per-group absmax，非完整 AWQ。"""
    M, K = W.shape
    assert K % group_size == 0
    ng = K // group_size
    scales = np.zeros((M, ng), np.float32)
    q = np.zeros((M, K), np.uint8)
    for g in range(ng):
        sl = slice(g * group_size, (g + 1) * group_size)
        block = W[:, sl]
        amax = np.max(np.abs(block), axis=1)
        scale = np.maximum(amax / 7.0, 1e-8)
        scales[:, g] = scale
        qq = np.round(block / scale[:, None]) + 7
        q[:, sl] = np.clip(qq, 0, 15).astype(np.uint8)
    # dequant: (q-7)*scale ≈ W
    return pack_int4(q), f32_to_f16_bits(scales.reshape(-1)), None


def put_u32(buf: bytearray, v: int):
    buf.extend(struct.pack("<I", v))


def put_u64(buf: bytearray, v: int):
    buf.extend(struct.pack("<Q", v))


def put_str(buf: bytearray, s: str):
    b = s.encode("utf-8")
    put_u32(buf, len(b))
    buf.extend(b)


def write_qlwc(path: Path, scheme: int, group_size: int, items: list):
    # items: dict name, kind, shape, pass_dtype?, data?, q?, scales?, zeros?
    cat = bytearray()
    put_u32(cat, scheme)
    put_u32(cat, group_size)
    put_u32(cat, ALIGN)
    put_u64(cat, len(items))
    # placeholder offsets — build twice
    def build_cat(with_off: bool, offsets: list | None = None):
        c = bytearray()
        put_u32(c, scheme)
        put_u32(c, group_size)
        put_u32(c, ALIGN)
        put_u64(c, len(items))
        for i, it in enumerate(items):
            put_str(c, it["name"])
            put_u32(c, it["kind"])
            put_u64(c, len(it["shape"]))
            for d in it["shape"]:
                put_u64(c, int(d))
            if it["kind"] == 0:
                put_u32(c, it["pass_dtype"])
                off = offsets[i]["data"] if with_off else 0
                put_u64(c, off)
                put_u64(c, len(it["data"]))
            else:
                put_u32(c, group_size)
                o = offsets[i] if with_off else {"q": 0, "s": 0, "z": 0}
                put_u64(c, o["q"])
                put_u64(c, len(it["q"]))
                put_u64(c, o["s"])
                put_u64(c, len(it["scales"]) * 2)
                put_u64(c, o["z"])
                zlen = 0 if it["zeros"] is None else len(it["zeros"]) * 2
                put_u64(c, zlen)
        return c

    probe = build_cat(False)
    off = align_up(24 + len(probe), ALIGN)
    offsets = []
    for it in items:
        if it["kind"] == 0:
            offsets.append({"data": off})
            off = align_up(off + len(it["data"]), ALIGN)
        else:
            q_off = off
            off = align_up(off + len(it["q"]), 64)
            s_off = off
            off = align_up(off + len(it["scales"]) * 2, 64)
            z_off = off
            zlen = 0 if it["zeros"] is None else len(it["zeros"]) * 2
            off = align_up(off + zlen, ALIGN)
            offsets.append({"q": q_off, "s": s_off, "z": z_off})

    catalog = build_cat(True, offsets)
    preface = QLW_MAGIC + struct.pack("<IQQ", 1, len(catalog), fnv1a64(bytes(catalog)))
    data0 = align_up(len(preface) + len(catalog), ALIGN)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 流式按偏移写盘，避免整文件 >2GiB 时双倍内存
    with path.open("wb") as f:
        f.write(preface)
        f.write(catalog)
        if f.tell() < data0:
            f.write(b"\0" * (data0 - f.tell()))
        f.truncate(off)
        for i, it in enumerate(items):
            if it["kind"] == 0:
                f.seek(offsets[i]["data"])
                f.write(it["data"])
            else:
                o = offsets[i]
                f.seek(o["q"])
                f.write(it["q"])
                f.seek(o["s"])
                f.write(it["scales"].astype(np.uint16).tobytes())
                if it["zeros"] is not None:
                    f.seek(o["z"])
                    f.write(it["zeros"].astype(np.uint16).tobytes())


def should_quantize(name: str, shape: list, min_cols: int, gs: int) -> bool:
    if len(shape) != 2 or shape[1] < min_cols or shape[1] % gs != 0:
        return False
    n = name.lower()
    # embedding / lm_head 保持原精度（查表 + 大 vocab GEMM）
    if n.endswith("embed_tokens.weight") or n.endswith("embedding.weight"):
        return False
    if n.endswith("lm_head.weight") or n == "lm_head.weight":
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="BF16/F16 LWC → QLWC INT4 (GPTQ/AWQ layout)")
    ap.add_argument("--src", required=True, help="输入 .lwc (BF16/F16)")
    ap.add_argument("--out", required=True, help="输出 .qlwc")
    ap.add_argument("--method", choices=("awq", "gptq"), default="awq")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--min-cols", type=int, default=128, help="列数低于此不量化(透传)")
    args = ap.parse_args()

    dtype, tensors, payloads = read_lwc(Path(args.src))
    to_f32 = bf16_bytes_to_f32 if dtype == 1 else f16_bytes_to_f32
    scheme = SCHEME_AWQ if args.method == "awq" else SCHEME_GPTQ
    gs = args.group_size
    items = []
    n_q, n_pass = 0, 0
    for t in tensors:
        name, shape, raw = t["name"], t["shape"], payloads[t["name"]]
        ne = int(np.prod(shape)) if shape else 0
        # 仅 2D 权重且 K 对齐 group；embedding/lm_head 透传
        if should_quantize(name, shape, args.min_cols, gs):
            W = to_f32(raw, ne).reshape(int(shape[0]), int(shape[1]))
            if args.method == "awq":
                q, scales, zeros = quant_awq_sym(W, gs)
            else:
                q, scales, zeros = quant_gptq_asym(W, gs)
            items.append(
                {
                    "name": name,
                    "kind": 1,
                    "shape": shape,
                    "q": q,
                    "scales": scales,
                    "zeros": zeros,
                }
            )
            n_q += 1
        else:
            items.append(
                {
                    "name": name,
                    "kind": 0,
                    "shape": shape,
                    "pass_dtype": dtype,
                    "data": raw,
                }
            )
            n_pass += 1

    out = Path(args.out)
    write_qlwc(out, scheme, gs, items)
    size_g = out.stat().st_size / (1024**3)
    print(f"[quantize_int4] method={args.method} group={gs}")
    print(f"[quantize_int4] quantized={n_q} passthrough={n_pass}")
    print(f"[quantize_int4] wrote {out} ({size_g:.2f} GiB)")
    print(f"[next] .\\build\\msvc-x64\\bin\\llmoc_server_int4.exe --config configs/engine_int4.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
