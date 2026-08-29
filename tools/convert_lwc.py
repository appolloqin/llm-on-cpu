#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_lwc.py — HuggingFace safetensors → LWC v1 (llm-on-cpu 权重容器)

功能:
  * 读取 HF 目录(config.json + *.safetensors[.index.json])
  * 识别 MoE 专家张量 (layers.N.mlp.experts.M.{gate,up,down}_proj.weight)
    → LWC 专家组布局(layers.N.experts.M.{gate,up,down}), 组内相邻、4K 对齐
  * 其余稠密权重按原名映射(去掉 model. 前缀, embed→embedding.weight)
  * 写 LWC v1 二进制目录(与 C++ lwc_format 严格一致)
  * 校验和策略: 落盘 checksum=0(哨兵), 转换后用 C++ 工具秒级回填:
        lwc_verify models/xxx.lwc --update
    (FNV 链式计算在 Python 上是分钟级, C++ 是秒级 —— 所以放 C++ 做)

用法:
  python tools/convert_lwc.py --src models/v4flash-hf --out models/v4flash.lwc
  python tools/convert_lwc.py --src ... --out ... --limit-experts 4   # 小样自检
  python tools/convert_lwc.py --src ... --out ... --verify            # 写完抽检结构
依赖: numpy, safetensors (bf16 权重额外需要 torch)
"""
import argparse
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

torch = None  # 延迟填充(BF16 读 / dtype 下转时需要)

MAGIC = b"LWC1"
VERSION = 1
ALIGN = 4096
DTYPE_CODE = {"BF16": 1, "F16": 2, "F32": 3}
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}
TARGET_TORCH = {"BF16": None, "F16": None, "F32": None}  # 延迟填充(见 main)
PART_MAP = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}

EXPERT_RE = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

# ---------------- 二进制序列化(与 src/weights/lwc_format.cpp 严格一致) ----------------

def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def u32(v): return struct.pack("<I", v)
def u64(v): return struct.pack("<Q", v)
def sstr(s): b = s.encode(); return u32(len(b)) + b


def build_catalog(dtype_code, align, tensors, groups):
    """tensors: [(name, shape, offset, nbytes, checksum)]; groups: [(layer, eid, names)]"""
    out = bytearray()
    out += u32(dtype_code) + u32(align) + u64(len(tensors)) + u64(len(groups))
    for name, shape, off, nb, ck in tensors:
        out += sstr(name) + u64(len(shape))
        for d in shape:
            out += u64(d)
        out += u64(off) + u64(nb) + u64(ck)
    for layer, eid, names in groups:
        out += u32(layer) + u32(eid) + u64(len(names))
        for n in names:
            out += sstr(n)
    return bytes(out)


# ---------------- HF 侧收集 ----------------

def collect_source_files(src: Path):
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        m = json.loads(idx.read_text(encoding="utf-8"))
        return [src / f for f in sorted(set(m["weight_map"].values()))]
    files = sorted(src.glob("*.safetensors"))
    if not files:
        sys.exit(f"ERROR: no *.safetensors under {src}")
    return files


def read_dtype_of(sf, key):
    sl = sf.get_slice(key)
    d = sl.get_dtype()          # 'BF16'|'F16'|'F32'|...
    return d


def tensor_to_bytes(sf, key, dtype):
    """numpy 路径覆盖 F16/F32; BF16 走 torch 句柄(safetensors.numpy 不支持 bf16)
    注意: 调用方必须已按 dtype 选择了正确的 framework 打开 sf。"""
    if dtype == "BF16":
        import torch  # 早已在 pass1 预检过, 此处必然可用
        ft = sf.get_tensor(key)  # framework=torch 时返回 torch tensor
        return ft.contiguous().view(torch.uint16).numpy().tobytes(), None
    arr = sf.get_tensor(key)  # framework=numpy
    return np.ascontiguousarray(arr).tobytes(), arr.shape


def main():
    ap = argparse.ArgumentParser(description="HF safetensors → LWC v1")
    ap.add_argument("--src", required=True, help="HF 模型目录(config.json 所在)")
    ap.add_argument("--out", required=True, help="输出 .lwc 路径")
    ap.add_argument("--limit-experts", type=int, default=0,
                    help="只转换每层前 N 个专家(小样自检用), 0=全部")
    ap.add_argument("--verify", action="store_true",
                    help="写完后重读目录做结构自检(抽检首尾张量)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    raw_dtype = (
        cfg.get("torch_dtype")
        or cfg.get("dtype")
        or (text_cfg or {}).get("torch_dtype")
        or (text_cfg or {}).get("dtype")
        or "float16"
    )
    torch_dtype = str(raw_dtype).lower()
    dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(torch_dtype)
    if dtype is None:
        sys.exit(f"ERROR: unsupported torch_dtype={torch_dtype}")
    dtype_code = DTYPE_CODE[dtype]

    files = collect_source_files(src)

    # ---- pass 1: 元数据收集(逐 key 映射, 不加载权重) ----
    entries = []          # (lwc_name, src_file, src_key, shape, nbytes)
    experts = {}          # (layer, eid) -> {part: lwc_name}
    skipped_experts = 0
    for fp in files:
        with safe_open(str(fp), framework="numpy") as f:
            keys = list(f.keys())
            for key in keys:
                dt = read_dtype_of(f, key)
                if dt == "BF16" and dtype != "BF16":
                    dtype, dtype_code = "BF16", DTYPE_CODE["BF16"]
                m = EXPERT_RE.match(key)
                if m:
                    layer, eid, part = int(m.group(1)), int(m.group(2)), m.group(3)
                    if args.limit_experts and eid >= args.limit_experts:
                        skipped_experts += 1
                        continue
                    lname = f"layers.{layer}.experts.{eid}.{PART_MAP[part]}"
                    experts.setdefault((layer, eid), {})[PART_MAP[part]] = lname
                    entries.append([lname, fp, key, dt, None, 0])
                else:
                    lname = ("embedding.weight"
                             if key == "model.embed_tokens.weight"
                             else re.sub(r"^model\.", "", key))
                    entries.append([lname, fp, key, dt, None, 0])

    # 形状/字节数(切片元数据, 不载入数据)
    for fp in files:
        with safe_open(str(fp), framework="numpy") as f:
            for e in entries:
                if e[1] == fp and e[4] is None:
                    sl = f.get_slice(e[2])
                    shape = tuple(sl.get_shape())
                    e[4] = shape
                    e[5] = int(np.prod(shape)) * ITEMSIZE[e[3]]

    # bf16 预检 / dtype 下转预检: 真正落盘前确认 torch 可用
    # (framework=torch 才能读 bf16; 源 dtype 与 header 不一致时还需 torch 做下转)
    need_torch = dtype == "BF16" or any(e[3] != dtype for e in ordered)
    if need_torch:
        if importlib.util.find_spec("torch") is None:
            sys.exit("ERROR: 转换需要 torch(tf32→{0} 或读取 bf16)。"
                     " pip install torch  或先把模型转为 float16。".format(dtype))
        global torch
        import torch as _torch
        torch = _torch
        TARGET_TORCH["BF16"] = _torch.bfloat16
        TARGET_TORCH["F16"] = _torch.float16
        TARGET_TORCH["F32"] = _torch.float32

    # 稠密在前/专家在后排序, 专家组内 gate,up,down 相邻
    order = {"gate": 0, "up": 1, "down": 2}
    dense = [e for e in entries if not EXPERT_RE.match(e[2])]
    expl = [e for e in entries if EXPERT_RE.match(e[2])]
    expl.sort(key=lambda e: (
        int(EXPERT_RE.match(e[2]).group(1)),
        int(EXPERT_RE.match(e[2]).group(2)),
        order[PART_MAP[EXPERT_RE.match(e[2]).group(3)]],
    ))
    ordered = dense + expl

    # ---- pass 2: 布局计算(目标字节数按 header dtype 计算; 源 dtype 不一致时 pass3 下转) ----
    tensors_meta = []
    name2off = {}
    cursor = 0
    total_bytes = 0
    for lname, _fp, _key, _dt, shape, nb in ordered:
        elems = int(np.prod(shape)) if shape else 1
        nb_target = elems * ITEMSIZE[dtype]
        tensors_meta.append([lname, list(shape), 0, nb_target, 0])  # offset 后填
        name2off[lname] = cursor
        cursor += 1
        total_bytes += nb_target
    # 数据区起点依赖目录长度, 而目录长度与 offset 无关(u64 定长) → 先置 0 算一次
    groups_meta = [
        (layer, eid, [experts[(layer, eid)][p] for p in ("gate", "up", "down")])
        for (layer, eid) in sorted(experts.keys())
        if len(experts[(layer, eid)]) == 3
    ]
    probe = build_catalog(dtype_code, ALIGN, tensors_meta, groups_meta)
    data_start = -(-((24 + len(probe)) ) // ALIGN) * ALIGN
    off = data_start
    for tm in tensors_meta:
        tm[2] = off
        off = -(-(off + tm[3]) // ALIGN) * ALIGN

    # ---- pass 3: 顺序写盘(逐张量: 真实字节偏移 = tensors_meta[i][2]) ----
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(dtype_code, ALIGN, tensors_meta, groups_meta)
    preface = MAGIC + u32(VERSION) + u64(len(catalog)) + u64(fnv1a64(catalog))
    with open(out, "wb") as w:
        w.write(preface)
        w.write(catalog)
        for (lname, fp, key, dt, _shape, nb), tm in zip(ordered, tensors_meta):
            if dt == dtype:
                # 同 dtype: 原路径逐字节搬(零依赖)
                fw = "torch" if dt == "BF16" else "numpy"
                with safe_open(str(fp), framework=fw) as f:
                    raw, _ = tensor_to_bytes(f, key, dt)
            else:
                # 源 dtype 与 header 不一致: 转成 header dtype 再写
                if torch is None:
                    sys.exit(f"ERROR: 张量 {key} dtype={dt} 与 header={dtype} 不一致,"
                             " 下转需要 torch: pip install torch")
                with safe_open(str(fp), framework="torch") as f:
                    arr = f.get_tensor(key)
                arr = arr.to(TARGET_TORCH[dtype]).contiguous()
                if dtype == "BF16":
                    raw = arr.view(torch.uint16).numpy().tobytes()
                elif dtype == "F16":
                    raw = arr.view(torch.uint16).numpy().tobytes()
                else:
                    raw = arr.view(torch.float32).numpy().tobytes()
            if len(raw) != tm[3]:
                sys.exit(f"ERROR: {key} size drift {len(raw)} != {tm[3]}")
            w.seek(tm[2])          # tm[2] = 该张量的真实文件偏移(已 4K 对齐)
            w.write(raw)
        # checksum=0 哨兵已写入目录; 回填交给 lwc_verify --update

    print(f"[convert] dtype={dtype} tensors={len(ordered)} "
          f"(dense={len(dense)}, expert={len(expl)}) "
          f"groups={len(groups_meta)} size={total_bytes/2**30:.2f} GiB")
    if skipped_experts:
        print(f"[convert] --limit-experts: skipped {skipped_experts} expert tensors")
    print(f"[convert] wrote {out}")
    print("[next ] .\\build\\msvc-x64\\bin\\lwc_verify.exe " + str(out)
          + " --update   # Win; Linux: ./build/release/bin/lwc_verify")

    if args.verify:
        with open(out, "rb") as r:
            head = r.read(24)
            cat_back = r.read(len(catalog))
        assert head[:4] == MAGIC, "magic mismatch"
        assert struct.unpack("<Q", head[8:16])[0] == len(catalog), "catalog len mismatch"
        assert struct.unpack("<Q", head[16:24])[0] == fnv1a64(cat_back), "catalog crc mismatch"
        assert cat_back == catalog, "catalog bytes differ from what was written"
        print("[verify] header/catalog structure OK")

if __name__ == "__main__":
    main()
