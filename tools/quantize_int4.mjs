#!/usr/bin/env node
// llm-on-cpu :: tools/quantize_int4.mjs
// BF16/F16 LWC → QLWC INT4 —— quantize_int4.py 的 Node 对等实现（零 npm 依赖）。
// 布局与 C++ src/weights/qlwc_format.* / Python 版字节级一致。
//
// 用法:
//   node tools/quantize_int4.mjs --src models/Qwen3.5-4B.lwc --out models/Qwen3.5-4B.int4.qlwc --method awq
//   node tools/quantize_int4.mjs --src ... --out ... --method gptq --group-size 128

import fs from "node:fs";
import path from "node:path";

const ALIGN = 4096;
const QLW_MAGIC = Buffer.from("QLW1");
const SCHEME_GPTQ = 1;
const SCHEME_AWQ = 2;

function fnv1a64(buf) {
  let h = 0xcbf29ce484222325n;
  const P = 0x100000001b3n;
  const M = (1n << 64n) - 1n;
  for (const b of buf) h = ((h ^ BigInt(b)) * P) & M;
  return h;
}

function alignUp(v, a) {
  return Math.floor((v + a - 1) / a) * a;
}

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--src") || !a.includes("--out")) {
    console.error(
      "usage: node quantize_int4.mjs --src <file.lwc> --out <file.qlwc> [--method awq|gptq] [--group-size 128] [--min-cols 128]",
    );
    process.exit(2);
  }
  const method = get("--method") ?? "awq";
  if (method !== "awq" && method !== "gptq") {
    console.error("--method must be awq or gptq");
    process.exit(2);
  }
  return {
    src: get("--src"),
    out: get("--out"),
    method,
    groupSize: Number(get("--group-size") ?? 128),
    minCols: Number(get("--min-cols") ?? 128),
  };
}

function u32(v) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v >>> 0);
  return b;
}
function u64(v) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(v));
  return b;
}
function str(s) {
  const b = Buffer.from(s, "utf8");
  return Buffer.concat([u32(b.length), b]);
}

/** IEEE754 f32 → f16 bits (uint16) */
function f32ToF16Bits(x) {
  const f32 = new Float32Array(1);
  const u32v = new Uint32Array(f32.buffer);
  f32[0] = x;
  const bits = u32v[0];
  const sign = (bits >>> 16) & 0x8000;
  let exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;
  if (exp === 255) {
    // Inf/NaN
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }
  if (exp === 0) {
    // subnormal/zero → flush to 0 in f16
    return sign;
  }
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00; // overflow → Inf
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >>> (shift + 13);
    // round
    if ((mant >>> (shift + 12)) & 1) half += 1;
    return sign | half;
  }
  let half = (exp << 10) | (mant >>> 13);
  if (mant & 0x1000) half += 1; // round
  return sign | (half & 0x7fff);
}

function f16BitsToF32(h) {
  const sign = (h & 0x8000) << 16;
  let exp = (h >>> 10) & 0x1f;
  let mant = h & 0x3ff;
  let bits;
  if (exp === 0) {
    if (mant === 0) bits = sign;
    else {
      exp = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp === 31) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  const u = new Uint32Array([bits]);
  return new Float32Array(u.buffer)[0];
}

function bf16BytesToF32(buf, n) {
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, n);
  const out = new Float32Array(n);
  const tmp = new Uint32Array(1);
  const fview = new Float32Array(tmp.buffer);
  for (let i = 0; i < n; ++i) {
    tmp[0] = u16[i] << 16;
    out[i] = fview[0];
  }
  return out;
}

function f16BytesToF32(buf, n) {
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; ++i) out[i] = f16BitsToF32(u16[i]);
  return out;
}

function f32ArrToF16Bits(arr) {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; ++i) out[i] = f32ToF16Bits(arr[i]);
  return out;
}

function readLwc(filePath) {
  const fd = fs.openSync(filePath, "r");
  try {
    const magic = Buffer.alloc(4);
    fs.readSync(fd, magic, 0, 4, 0);
    if (magic.toString("ascii") !== "LWC1") throw new Error(`not LWC1: ${filePath}`);
    const pref = Buffer.alloc(20);
    fs.readSync(fd, pref, 0, 20, 4);
    const ver = pref.readUInt32LE(0);
    const catLen = Number(pref.readBigUInt64LE(4));
    const catCrc = pref.readBigUInt64LE(12);
    void ver;
    const cat = Buffer.alloc(catLen);
    fs.readSync(fd, cat, 0, catLen, 24);
    if (fnv1a64(cat) !== catCrc) throw new Error("LWC catalog crc mismatch");

    let o = 0;
    const u32r = () => {
      const v = cat.readUInt32LE(o);
      o += 4;
      return v;
    };
    const u64r = () => {
      const v = Number(cat.readBigUInt64LE(o));
      o += 8;
      return v;
    };
    const sr = () => {
      const n = u32r();
      const s = cat.subarray(o, o + n).toString("utf8");
      o += n;
      return s;
    };

    const dtype = u32r();
    u32r(); // align
    const nT = u64r();
    const nG = u64r();
    const tensors = [];
    for (let i = 0; i < nT; ++i) {
      const name = sr();
      const nd = u64r();
      const shape = [];
      for (let d = 0; d < nd; ++d) shape.push(u64r());
      const offset = u64r();
      const nbytes = u64r();
      u64r(); // checksum
      tensors.push({ name, shape, offset, nbytes });
    }
    for (let g = 0; g < nG; ++g) {
      u32r();
      u32r();
      const nn = u64r();
      for (let i = 0; i < nn; ++i) sr();
    }

    const payloads = {};
    for (const t of tensors) {
      const buf = Buffer.alloc(t.nbytes);
      fs.readSync(fd, buf, 0, t.nbytes, t.offset);
      payloads[t.name] = buf;
    }
    return { dtype, tensors, payloads };
  } finally {
    fs.closeSync(fd);
  }
}

function packInt4(q /* Uint8Array flat M*K */, M, K) {
  let Kp = K;
  let src = q;
  if (K % 2) {
    Kp = K + 1;
    src = new Uint8Array(M * Kp);
    for (let m = 0; m < M; ++m) {
      src.set(q.subarray(m * K, m * K + K), m * Kp);
    }
  }
  const packed = new Uint8Array(M * (Kp / 2));
  for (let m = 0; m < M; ++m) {
    const row = m * Kp;
    const prow = m * (Kp / 2);
    for (let k = 0; k < Kp; k += 2) {
      packed[prow + k / 2] = (src[row + k] & 0xf) | ((src[row + k + 1] & 0xf) << 4);
    }
  }
  return Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength);
}

function quantGptqAsym(W /* Float32Array */, M, K, groupSize) {
  if (K % groupSize !== 0) throw new Error("K % group_size != 0");
  const ng = K / groupSize;
  const scales = new Float32Array(M * ng);
  const zeros = new Float32Array(M * ng);
  const q = new Uint8Array(M * K);
  for (let g = 0; g < ng; ++g) {
    const k0 = g * groupSize;
    for (let m = 0; m < M; ++m) {
      let mn = Infinity;
      let mx = -Infinity;
      const base = m * K + k0;
      for (let k = 0; k < groupSize; ++k) {
        const v = W[base + k];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      let scale = (mx - mn) / 15.0;
      if (scale < 1e-8) scale = 1e-8;
      scales[m * ng + g] = scale;
      zeros[m * ng + g] = mn;
      for (let k = 0; k < groupSize; ++k) {
        let qq = Math.round((W[base + k] - mn) / scale);
        if (qq < 0) qq = 0;
        if (qq > 15) qq = 15;
        q[base + k] = qq;
      }
    }
  }
  return {
    q: packInt4(q, M, K),
    scales: f32ArrToF16Bits(scales),
    zeros: f32ArrToF16Bits(zeros),
  };
}

function quantAwqSym(W, M, K, groupSize) {
  if (K % groupSize !== 0) throw new Error("K % group_size != 0");
  const ng = K / groupSize;
  // 对称 absmax（AWQ 兼容无-zero 布局；非完整 AWQ 校准）
  const scales = new Float32Array(M * ng);
  const q = new Uint8Array(M * K);
  for (let g = 0; g < ng; ++g) {
    const k0 = g * groupSize;
    for (let m = 0; m < M; ++m) {
      const base = m * K + k0;
      let amax = 0;
      for (let k = 0; k < groupSize; ++k) {
        const a = Math.abs(W[base + k]);
        if (a > amax) amax = a;
      }
      let scale = amax / 7.0;
      if (scale < 1e-8) scale = 1e-8;
      scales[m * ng + g] = scale;
      for (let k = 0; k < groupSize; ++k) {
        let qq = Math.round(W[base + k] / scale) + 7;
        if (qq < 0) qq = 0;
        if (qq > 15) qq = 15;
        q[base + k] = qq;
      }
    }
  }
  return { q: packInt4(q, M, K), scales: f32ArrToF16Bits(scales), zeros: null };
}

function writeAt(fd, buf, filePos) {
  // Node writeSync length 上限为 2^31-1；大文件必须分块写
  const CHUNK = 1 << 30; // 1 GiB
  let off = 0;
  let pos = filePos;
  while (off < buf.length) {
    const n = Math.min(CHUNK, buf.length - off);
    const w = fs.writeSync(fd, buf, off, n, pos);
    off += w;
    pos += w;
  }
}

function writeQlwc(filePath, scheme, groupSize, items) {
  const buildCat = (withOff, offsets) => {
    const parts = [u32(scheme), u32(groupSize), u32(ALIGN), u64(items.length)];
    for (let i = 0; i < items.length; ++i) {
      const it = items[i];
      parts.push(str(it.name), u32(it.kind), u64(it.shape.length));
      for (const d of it.shape) parts.push(u64(d));
      if (it.kind === 0) {
        parts.push(u32(it.pass_dtype));
        const off = withOff ? offsets[i].data : 0;
        parts.push(u64(off), u64(it.data.length));
      } else {
        const o = withOff ? offsets[i] : { q: 0, s: 0, z: 0 };
        const zlen = it.zeros ? it.zeros.length * 2 : 0;
        parts.push(
          u32(groupSize),
          u64(o.q),
          u64(it.q.length),
          u64(o.s),
          u64(it.scales.length * 2),
          u64(o.z),
          u64(zlen),
        );
      }
    }
    return Buffer.concat(parts);
  };

  const probe = buildCat(false, null);
  let off = alignUp(24 + probe.length, ALIGN);
  const offsets = [];
  for (const it of items) {
    if (it.kind === 0) {
      offsets.push({ data: off });
      off = alignUp(off + it.data.length, ALIGN);
    } else {
      const qOff = off;
      off = alignUp(off + it.q.length, 64);
      const sOff = off;
      off = alignUp(off + it.scales.length * 2, 64);
      const zOff = off;
      const zlen = it.zeros ? it.zeros.length * 2 : 0;
      off = alignUp(off + zlen, ALIGN);
      offsets.push({ q: qOff, s: sOff, z: zOff });
    }
  }

  const catalog = buildCat(true, offsets);
  const crc = fnv1a64(catalog);
  const preface = Buffer.concat([
    QLW_MAGIC,
    u32(1),
    u64(catalog.length),
    (() => {
      const b = Buffer.alloc(8);
      b.writeBigUInt64LE(crc);
      return b;
    })(),
  ]);
  const data0 = alignUp(preface.length + catalog.length, ALIGN);
  const fileSize = Math.max(off, data0);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const fd = fs.openSync(filePath, "w");
  try {
    // 不整文件 Buffer.alloc：>2GiB 会撞 Node write 长度上限
    writeAt(fd, preface, 0);
    writeAt(fd, catalog, preface.length);
    fs.ftruncateSync(fd, fileSize); // 对齐空洞填 0
    for (let i = 0; i < items.length; ++i) {
      const it = items[i];
      if (it.kind === 0) {
        writeAt(fd, it.data, offsets[i].data);
      } else {
        const o = offsets[i];
        writeAt(fd, it.q, o.q);
        writeAt(
          fd,
          Buffer.from(it.scales.buffer, it.scales.byteOffset, it.scales.byteLength),
          o.s,
        );
        if (it.zeros) {
          writeAt(
            fd,
            Buffer.from(it.zeros.buffer, it.zeros.byteOffset, it.zeros.byteLength),
            o.z,
          );
        }
      }
      if ((i + 1) % 50 === 0)
        process.stderr.write(`\r[quantize_int4] writing ${i + 1}/${items.length}...`);
    }
    if (items.length >= 50) process.stderr.write("\n");
  } finally {
    fs.closeSync(fd);
  }
}

function shouldQuantize(name, shape, minCols, gs) {
  if (shape.length !== 2 || shape[1] < minCols || shape[1] % gs !== 0) return false;
  const n = name.toLowerCase();
  if (n.endsWith("embed_tokens.weight") || n.endsWith("embedding.weight")) return false;
  if (n.endsWith("lm_head.weight") || n === "lm_head.weight") return false;
  return true;
}

function main() {
  const opt = parseArgs();
  const src = path.resolve(opt.src);
  const out = path.resolve(opt.out);
  const { dtype, tensors, payloads } = readLwc(src);
  const toF32 = dtype === 1 ? bf16BytesToF32 : f16BytesToF32;
  const scheme = opt.method === "awq" ? SCHEME_AWQ : SCHEME_GPTQ;
  const gs = opt.groupSize;
  const items = [];
  let nQ = 0;
  let nPass = 0;

  for (const t of tensors) {
    const { name, shape } = t;
    const raw = payloads[name];
    const ne = shape.reduce((a, b) => a * b, 1);
    if (shouldQuantize(name, shape, opt.minCols, gs)) {
      const M = shape[0];
      const K = shape[1];
      const W = toF32(raw, ne);
      const qres =
        opt.method === "awq" ? quantAwqSym(W, M, K, gs) : quantGptqAsym(W, M, K, gs);
      items.push({
        name,
        kind: 1,
        shape,
        q: qres.q,
        scales: qres.scales,
        zeros: qres.zeros,
      });
      nQ += 1;
      if (nQ % 20 === 0) process.stderr.write(`\r[quantize_int4] quantized ${nQ} tensors...`);
    } else {
      items.push({ name, kind: 0, shape, pass_dtype: dtype, data: raw });
      nPass += 1;
    }
  }
  if (nQ >= 20) process.stderr.write("\n");

  writeQlwc(out, scheme, gs, items);
  const sizeG = fs.statSync(out).size / 1024 ** 3;
  console.log(`[quantize_int4] method=${opt.method} group=${gs}`);
  console.log(`[quantize_int4] quantized=${nQ} passthrough=${nPass}`);
  console.log(`[quantize_int4] wrote ${out} (${sizeG.toFixed(2)} GiB)`);
  console.log(`[next] .\\build\\msvc-x64\\bin\\llmoc_server_int4.exe --config configs/engine_int4.yaml`);
}

main();
