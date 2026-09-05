#!/usr/bin/env node
// llm-on-cpu :: tools/import_awq_hf_qlwc.mjs
// HuggingFace 预量化 INT4（AutoAWQ / compressed-tensors pack-quantized）→ QLWC 重打包（不二次量化 BF16）。
//
// 用法:
//   node tools/import_awq_hf_qlwc.mjs --src models/Qwen3.8-27B-AWQ-INT4-hf --out models/Qwen3.8-27B-AWQ-INT4.int4.qlwc
//   node tools/import_awq_hf_qlwc.mjs --src ... --out ... --group-size 32

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const QLW_MAGIC = Buffer.from("QLW1");
const SCHEME_GPTQ = 1;
const SCHEME_AWQ = 2;
const ALIGN = 4096;

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--src") || !a.includes("--out")) {
    console.error(
      "usage: node tools/import_awq_hf_qlwc.mjs --src <hf_dir> --out <file.qlwc>\n" +
        "       [--group-size N] [--min-cols 128] [--format auto|compressed-tensors|autoawq]",
    );
    process.exit(2);
  }
  return {
    src: path.resolve(get("--src")),
    out: path.resolve(get("--out")),
    groupSize: Number(get("--group-size") ?? 0),
    minCols: Number(get("--min-cols") ?? 128),
    format: get("--format") ?? "auto",
  };
}

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

function u32(v) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v >>> 0);
  return b;
}
function u64(v) {
  const n = typeof v === "bigint" ? v : BigInt(Math.trunc(Number(v)));
  if (n < 0n) throw new Error(`u64 got negative ${v} (often int32 overflow on ≥2GiB size)`);
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(n);
  return b;
}
function str(s) {
  const b = Buffer.from(s, "utf8");
  return Buffer.concat([u32(b.length), b]);
}

function f32ToF16Bits(x) {
  const f32 = new Float32Array(1);
  const u32v = new Uint32Array(f32.buffer);
  f32[0] = x;
  const bits = u32v[0];
  const sign = (bits >>> 16) & 0x8000;
  let exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;
  if (exp === 255) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp === 0) return sign;
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >>> (shift + 13);
    if ((mant >>> (shift + 12)) & 1) half += 1;
    return sign | half;
  }
  let half = (exp << 10) | (mant >>> 13);
  if (mant & 0x1000) half += 1;
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
  const u = new Uint32Array([bits >>> 0]);
  return new Float32Array(u.buffer)[0];
}

function f32ArrToF16Bits(arr) {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; ++i) out[i] = f32ToF16Bits(arr[i]);
  return out;
}

function packInt4(q, M, K) {
  let Kp = K;
  let src = q;
  if (K % 2) {
    Kp = K + 1;
    src = new Uint8Array(M * Kp);
    for (let m = 0; m < M; ++m) src.set(q.subarray(m * K, m * K + K), m * Kp);
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

function quantGptqAsym(W, M, K, groupSize) {
  if (K % groupSize !== 0) throw new Error(`K=${K} % group_size=${groupSize} != 0`);
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
  return { q: packInt4(q, M, K), scales: f32ArrToF16Bits(scales), zeros: f32ArrToF16Bits(zeros) };
}

function quantAwqSym(W, M, K, groupSize) {
  if (K % groupSize !== 0) throw new Error(`K=${K} % group_size=${groupSize} != 0`);
  const ng = K / groupSize;
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

// Node Buffer 长度走有符号 32-bit；≥2GiB 会变成负数并抛 length out of range。
const MAX_BUF = 0x7ffff000;
const COPY_CHUNK = 1 << 20;

function blobFromBuffer(buf) {
  return { kind: "buffer", buf, length: buf.length };
}
function blobFromFile(filePath, length) {
  return { kind: "file", path: filePath, length };
}
function blobFromSlice(entry, length) {
  return { kind: "slice", entry, length: length ?? entry.end - entry.start };
}
function blobLen(b) {
  if (!b) return 0;
  if (Buffer.isBuffer(b) || ArrayBuffer.isView(b)) return b.byteLength ?? b.length;
  // 禁止 `| 0`：≥2GiB 会变成有符号负数，随后 u64()/writeBigUInt64LE 报错
  const n = Number(b.length);
  if (!Number.isFinite(n) || n < 0) throw new Error(`bad blob length ${b.length}`);
  return n;
}

function writeAt(fd, buf, filePos) {
  const CHUNK = 1 << 30;
  let off = 0;
  let pos = filePos;
  while (off < buf.length) {
    const n = Math.min(CHUNK, buf.length - off);
    const w = fs.writeSync(fd, buf, off, n, pos);
    off += w;
    pos += w;
  }
}

function writeBlob(fd, blob, filePos) {
  if (!blob) return;
  if (Buffer.isBuffer(blob) || ArrayBuffer.isView(blob)) {
    const buf = Buffer.isBuffer(blob)
      ? blob
      : Buffer.from(blob.buffer, blob.byteOffset, blob.byteLength);
    writeAt(fd, buf, filePos);
    return;
  }
  if (blob.kind === "buffer") {
    writeAt(fd, blob.buf, filePos);
    return;
  }
  const tmp = Buffer.alloc(COPY_CHUNK);
  if (blob.kind === "file") {
    const src = fs.openSync(blob.path, "r");
    try {
      let left = blob.length;
      let pos = filePos;
      let soff = 0;
      while (left > 0) {
        const n = Math.min(COPY_CHUNK, left);
        const r = fs.readSync(src, tmp, 0, n, soff);
        if (r <= 0) throw new Error(`short read temp blob ${blob.path}`);
        writeAt(fd, tmp.subarray(0, r), pos);
        soff += r;
        pos += r;
        left -= r;
      }
    } finally {
      fs.closeSync(src);
    }
    return;
  }
  if (blob.kind === "slice") {
    const e = blob.entry;
    const src = fs.openSync(e.file, "r");
    try {
      let left = blob.length;
      let pos = filePos;
      let soff = e.start;
      while (left > 0) {
        const n = Math.min(COPY_CHUNK, left);
        const r = fs.readSync(src, tmp, 0, n, soff);
        if (r <= 0) throw new Error(`short read slice ${e.file}`);
        writeAt(fd, tmp.subarray(0, r), pos);
        soff += r;
        pos += r;
        left -= r;
      }
    } finally {
      fs.closeSync(src);
    }
    return;
  }
  throw new Error(`unknown blob kind ${blob.kind}`);
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
        parts.push(u64(withOff ? offsets[i].data : 0), u64(blobLen(it.data)));
      } else {
        const o = withOff ? offsets[i] : { q: 0, s: 0, z: 0 };
        const zlen = it.zeros ? blobLen(it.zeros) : 0;
        parts.push(
          u32(groupSize),
          u64(o.q),
          u64(blobLen(it.q)),
          u64(o.s),
          u64(blobLen(it.scales)),
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
      off = alignUp(off + blobLen(it.data), ALIGN);
    } else {
      const qOff = off;
      off = alignUp(off + blobLen(it.q), 64);
      const sOff = off;
      off = alignUp(off + blobLen(it.scales), 64);
      const zOff = off;
      const zlen = it.zeros ? blobLen(it.zeros) : 0;
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
  const fileSize = Math.max(off, alignUp(preface.length + catalog.length, ALIGN));

  const outDir = path.dirname(filePath);
  // Windows 盘符根目录（如 E:\）mkdir 会 EPERM，跳过即可
  const root = path.parse(outDir).root;
  if (outDir && outDir !== root) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  const fd = fs.openSync(filePath, "w");
  try {
    writeAt(fd, preface, 0);
    writeAt(fd, catalog, preface.length);
    fs.ftruncateSync(fd, fileSize);
    for (let i = 0; i < items.length; ++i) {
      const it = items[i];
      if (it.kind === 0) writeBlob(fd, it.data, offsets[i].data);
      else {
        writeBlob(fd, it.q, offsets[i].q);
        writeBlob(fd, it.scales, offsets[i].s);
        if (it.zeros) writeBlob(fd, it.zeros, offsets[i].z);
      }
      if ((i + 1) % 50 === 0) process.stderr.write(`\r[import-awq] writing ${i + 1}/${items.length}...`);
    }
    if (items.length >= 50) process.stderr.write("\n");
  } finally {
    fs.closeSync(fd);
  }
}

function listSafetensors(dir) {
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".safetensors"))
    .sort()
    .map((f) => path.join(dir, f));
}

async function readSafetensorsHeader(file) {
  const fh = await fsp.open(file, "r");
  try {
    const lenBuf = Buffer.alloc(8);
    await fh.read(lenBuf, 0, 8, 0);
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    const hdrBuf = Buffer.alloc(headerLen);
    await fh.read(hdrBuf, 0, headerLen, 8);
    return {
      fh,
      header: JSON.parse(hdrBuf.toString("utf8")),
      dataBase: 8 + headerLen,
    };
  } catch (e) {
    await fh.close();
    throw e;
  }
}

async function buildIndex(hfDir) {
  const index = {};
  for (const file of listSafetensors(hfDir)) {
    const { fh, header, dataBase } = await readSafetensorsHeader(file);
    for (const [key, meta] of Object.entries(header)) {
      if (key === "__metadata__") continue;
      const [s, e] = meta.data_offsets;
      index[key] = { file, dataBase, start: dataBase + s, end: dataBase + e, shape: meta.shape, dtype: meta.dtype };
    }
    await fh.close();
  }
  return index;
}

function readTensor(entry) {
  const n = Number(entry.end) - Number(entry.start);
  if (!Number.isFinite(n) || n < 0) {
    throw new Error(`bad tensor span start=${entry.start} end=${entry.end}`);
  }
  if (n > MAX_BUF) {
    throw new Error(
      `tensor blob too large for single Buffer (${n} bytes ≈ ${(n / 2 ** 30).toFixed(2)} GiB); use slice/stream path`,
    );
  }
  const buf = Buffer.alloc(n);
  const fd = fs.openSync(entry.file, "r");
  try {
    let off = 0;
    while (off < n) {
      const w = fs.readSync(fd, buf, off, n - off, entry.start + off);
      if (w <= 0) throw new Error(`short read at ${entry.file}`);
      off += w;
    }
  } finally {
    fs.closeSync(fd);
  }
  return buf;
}

function gptqZerosFromZp(scaleF, zpF, M, ng, scaleLayout, zpLayout) {
  // CT/GPTQ: w=(q-zp)*scale  →  QLWC GPTQ: w=q*scale+zero  ⇒  zero = -zp*scale
  const out = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m) {
    for (let g = 0; g < ng; ++g) {
      const sc = pickScale(scaleF, M, ng, m, g, scaleLayout);
      const zp = pickScale(zpF, M, ng, m, g, zpLayout);
      out[m * ng + g] = -zp * sc;
    }
  }
  return f32ArrToF16Bits(out);
}

/**
 * compressed-tensors weight_zero_point: I32 [ceil(M/8), ng]，每 int32 沿 M 打包 8 个 4bit zp。
 * 也兼容已展开的 [M,ng] / [ng,M] / I8 等。
 */
function loadCtZeroPoints(zerosE, M, ng) {
  const zeros = readTensor(zerosE);
  const dt = String(zerosE.dtype || "").toUpperCase();
  const shape = zerosE.shape || [];
  const packedRows = Math.ceil(M / 8);
  const elems = tensorElemCount(zerosE, zeros);

  const unpackPackedI32 = (layoutPmNg) => {
    if (zeros.length < packedRows * ng * 4) {
      throw new Error(`packed zp too short bytes=${zeros.length} need=${packedRows * ng * 4}`);
    }
    const view = new Int32Array(zeros.buffer, zeros.byteOffset, packedRows * ng);
    const out = new Float32Array(M * ng);
    for (let m = 0; m < M; ++m) {
      const pm = (m / 8) | 0;
      const shift = (m % 8) * 4;
      for (let g = 0; g < ng; ++g) {
        const idx = layoutPmNg ? pm * ng + g : g * packedRows + pm;
        out[m * ng + g] = (view[idx] >>> shift) & 0xf;
      }
    }
    return out;
  };

  // 典型 CT: shape [M/8, ng] I32
  if (
    (dt === "I32" || dt === "U32") &&
    ((shape.length === 2 && shape[0] === packedRows && shape[1] === ng) ||
      elems === packedRows * ng)
  ) {
    return { zpF: unpackPackedI32(true), layout: "m_g" };
  }
  // 偶发 [ng, M/8]
  if (
    (dt === "I32" || dt === "U32") &&
    shape.length === 2 &&
    shape[0] === ng &&
    shape[1] === packedRows
  ) {
    return { zpF: unpackPackedI32(false), layout: "m_g" };
  }

  const zpF = tensorToF32(zeros, zerosE.dtype, elems);
  const layout = inferLayout(zpF.length, M, ng);
  return { zpF, layout };
}

function scalesToF16Mg(scaleF, M, ng, layout) {
  const out = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m) {
    for (let g = 0; g < ng; ++g) out[m * ng + g] = pickScale(scaleF, M, ng, m, g, layout);
  }
  return f32ArrToF16Bits(out);
}

function allocQPayload(qBytes, tmpDir, tag) {
  if (qBytes <= MAX_BUF) return { mode: "buffer", q: Buffer.alloc(qBytes), length: qBytes };
  const filePath = path.join(tmpDir, `${tag}.q.bin`);
  const fd = fs.openSync(filePath, "w");
  fs.ftruncateSync(fd, qBytes);
  return { mode: "file", fd, path: filePath, length: qBytes };
}

function finishQPayload(payload) {
  if (payload.mode === "buffer") return blobFromBuffer(payload.q);
  fs.closeSync(payload.fd);
  return blobFromFile(payload.path, payload.length);
}

/** compressed-tensors pack-quantized → QLWC：按行流式重打包，不解 FP32、不二次量化 */
function importCtToQlwc(packedE, scalesE, zerosE, M, K, gsHint, symmetric, tmpDir, tag) {
  if (K % 2 !== 0) throw new Error(`QLWC pack needs even K, got ${K}`);
  const colsPacked = Math.ceil(K / 8);
  const rowBytes = colsPacked * 4;
  const expected = M * rowBytes;
  const span = packedE.end - packedE.start;
  if (span < expected) {
    throw new Error(`packed size ${span} < expected ${expected} for M=${M} K=${K}`);
  }

  const scales = readTensor(scalesE);
  const scaleCount = tensorElemCount(scalesE, scales);
  const scaleF = tensorToF32(scales, scalesE.dtype, scaleCount);
  const resolved = resolveScaleGroup(scaleF.length, M, K, gsHint);
  const { ng, gs, layout: scaleLayout } = resolved;
  if (gsHint > 0 && gs !== gsHint) {
    // 仅首次由调用方汇总打印；此处保持安静以免刷屏
  }
  const qlwcScales = scalesToF16Mg(scaleF, M, ng, scaleLayout);

  let qlwcZeros = null;
  if (!symmetric) {
    if (!zerosE) throw new Error("asymmetric CT weights need weight_zero_point");
    const { zpF, layout: zpLayout } = loadCtZeroPoints(zerosE, M, ng);
    qlwcZeros = gptqZerosFromZp(scaleF, zpF, M, ng, scaleLayout, zpLayout);
  }

  const qBytes = M * (K / 2);
  const payload = allocQPayload(qBytes, tmpDir, tag);
  const rowOut = Buffer.alloc(K / 2);
  const rowBuf = Buffer.alloc(rowBytes);
  const fd = fs.openSync(packedE.file, "r");
  try {
    for (let m = 0; m < M; ++m) {
      let off = 0;
      const pos0 = packedE.start + m * rowBytes;
      while (off < rowBytes) {
        const w = fs.readSync(fd, rowBuf, off, rowBytes - off, pos0 + off);
        if (w <= 0) throw new Error(`short packed row read m=${m}`);
        off += w;
      }
      const view = new Int32Array(rowBuf.buffer, rowBuf.byteOffset, colsPacked);
      for (let k = 0; k < K; k += 2) {
        const pc0 = (k / 8) | 0;
        const pc1 = ((k + 1) / 8) | 0;
        const q0 = (view[pc0] >>> ((k % 8) * 4)) & 0xf;
        const q1 = (view[pc1] >>> (((k + 1) % 8) * 4)) & 0xf;
        rowOut[k / 2] = (q0 & 0xf) | ((q1 & 0xf) << 4);
      }
      if (payload.mode === "buffer") {
        rowOut.copy(payload.q, m * (K / 2));
      } else {
        writeAt(payload.fd, rowOut, m * (K / 2));
      }
    }
    return { q: finishQPayload(payload), scales: qlwcScales, zeros: qlwcZeros, gs, ng };
  } catch (err) {
    if (payload.mode === "file") {
      try {
        fs.closeSync(payload.fd);
      } catch {
        /* ignore */
      }
    }
    throw err;
  } finally {
    fs.closeSync(fd);
  }
}

/** AutoAWQ qweight [K, ceil(M/8)] int32 → QLWC row-nibble */
function importAwqToQlwc(qweightE, scalesE, qzerosE, M, K, gs, symmetric, tmpDir, tag) {
  if (K % gs !== 0) throw new Error(`K=${K} % group_size=${gs} != 0`);
  if (K % 2 !== 0) throw new Error(`QLWC pack needs even K, got ${K}`);
  const ng = K / gs;
  const colsPacked = Math.ceil(M / 8);
  const colBytes = colsPacked * 4;

  const scales = readTensor(scalesE);
  const scaleCount =
    scalesE.dtype === "F16" || scalesE.dtype === "BF16" ? scales.length / 2 : scales.length / 4;
  const scaleF = tensorToF32(scales, scalesE.dtype, scaleCount);
  const scaleLayout = inferLayout(scaleF.length, M, ng);
  const qlwcScales = scalesToF16Mg(scaleF, M, ng, scaleLayout);

  let qlwcZeros = null;
  if (!symmetric) {
    if (!qzerosE) throw new Error("asymmetric AutoAWQ needs qzeros");
    const qzeros = readTensor(qzerosE);
    const zeroF = tensorToF32(qzeros, "I32", qzeros.length / 4);
    const zpF = new Float32Array(M * ng);
    for (let m = 0; m < M; ++m) {
      for (let g = 0; g < ng; ++g) {
        const zpIdx = g * Math.ceil(M / 8) + Math.floor(m / 8);
        const word = zeroF[zpIdx] | 0;
        zpF[m * ng + g] = (word >>> ((m % 8) * 4)) & 0xf;
      }
    }
    qlwcZeros = gptqZerosFromZp(scaleF, zpF, M, ng, scaleLayout, "m_g");
  }

  const qBytes = M * (K / 2);
  if (qBytes > MAX_BUF) {
    throw new Error(
      `AutoAWQ packed weight too large (${(qBytes / 2 ** 30).toFixed(2)} GiB); ` +
        `use compressed-tensors HF or a smaller model for streaming import`,
    );
  }
  const outQ = Buffer.alloc(qBytes);
  const colBuf = Buffer.alloc(colBytes);
  const fd = fs.openSync(qweightE.file, "r");
  try {
    for (let k = 0; k < K; k += 2) {
      for (let kk = 0; kk < 2; ++kk) {
        const kkAbs = k + kk;
        let off = 0;
        const pos0 = qweightE.start + kkAbs * colBytes;
        while (off < colBytes) {
          const w = fs.readSync(fd, colBuf, off, colBytes - off, pos0 + off);
          if (w <= 0) throw new Error(`short awq col read k=${kkAbs}`);
          off += w;
        }
        const view = new Int32Array(colBuf.buffer, colBuf.byteOffset, colsPacked);
        for (let m = 0; m < M; ++m) {
          const pm = (m / 8) | 0;
          const q = (view[pm] >>> ((m % 8) * 4)) & 0xf;
          const byteIndex = m * (K / 2) + (k / 2);
          if (kk === 0) outQ[byteIndex] = (outQ[byteIndex] & 0xf0) | (q & 0xf);
          else outQ[byteIndex] = (outQ[byteIndex] & 0x0f) | ((q & 0xf) << 4);
        }
      }
    }
  } finally {
    fs.closeSync(fd);
  }
  return { q: blobFromBuffer(outQ), scales: qlwcScales, zeros: qlwcZeros, gs, ng };
}

function loadConfig(hfDir) {
  const p = path.join(hfDir, "config.json");
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function quantOptsFromConfig(cfg) {
  const qc = cfg.quantization_config || cfg.compression_config || cfg.text_config?.quantization_config || {};
  const groups = qc.config_groups || {};
  const g0 = groups.group_0 || groups[Object.keys(groups)[0]] || {};
  const wg = g0.weights || {};
  return {
    format: String(qc.format || qc.quant_method || "").toLowerCase(),
    method: String(qc.quant_method || qc.quantization_method || "").toLowerCase(),
    groupSize: wg.group_size ?? qc.group_size ?? 128,
    symmetric: wg.symmetric ?? true,
    numBits: wg.num_bits ?? qc.bits ?? 4,
  };
}

function detectFormat(index, cfgOpts, forced) {
  if (forced !== "auto") return forced;
  const names = Object.keys(index);
  if (cfgOpts.format.includes("pack-quantized") || cfgOpts.method === "compressed-tensors") {
    return "compressed-tensors";
  }
  if (cfgOpts.method === "awq" || names.some((n) => n.endsWith(".qweight"))) return "autoawq";
  if (names.some((n) => n.endsWith(".weight_packed"))) return "compressed-tensors";
  throw new Error("cannot detect HF quant format (need .weight_packed or .qweight tensors, or config.quantization_config)");
}

function readWeightShape(entry) {
  const buf = readTensor(entry);
  const dt = String(entry.dtype || "").toUpperCase();
  let M;
  let K;
  if (dt === "I64" || dt === "U64") {
    if (buf.length < 16) throw new Error(`weight_shape I64 too short (${buf.length})`);
    M = Number(buf.readBigInt64LE(0));
    K = Number(buf.readBigInt64LE(8));
  } else if (dt === "I32" || dt === "U32") {
    if (buf.length < 8) throw new Error(`weight_shape I32 too short (${buf.length})`);
    M = buf.readInt32LE(0);
    K = buf.readInt32LE(4);
  } else if (buf.length >= 16) {
    // 缺 dtype 时：16B 优先按 I64（compressed-tensors 常见）
    M = Number(buf.readBigInt64LE(0));
    K = Number(buf.readBigInt64LE(8));
  } else if (buf.length >= 8) {
    M = buf.readInt32LE(0);
    K = buf.readInt32LE(4);
  } else {
    throw new Error(`weight_shape bad size ${buf.length} dtype=${entry.dtype}`);
  }
  if (!Number.isFinite(M) || !Number.isFinite(K) || M <= 0 || K <= 0) {
    throw new Error(`weight_shape invalid M=${M} K=${K} dtype=${entry.dtype}`);
  }
  return [M, K];
}

function inferCtMk(packedE, shapeE) {
  if (shapeE) {
    const [M, K] = readWeightShape(shapeE);
    const colsPacked = Math.ceil(K / 8);
    const expect = M * colsPacked * 4;
    const span = packedE.end - packedE.start;
    if (span < expect) {
      throw new Error(
        `weight_shape [M=${M},K=${K}] expects packed ${expect}B but tensor has ${span}B`,
      );
    }
    return [M, K];
  }
  // packed layout [M, ceil(K/8)] int32
  const M = packedE.shape[0];
  const K = packedE.shape[1] * 8;
  return [M, K];
}

function mapQlwcName(hfKey) {
  let n = hfKey;
  const wasWeight = n.endsWith(".weight");
  if (wasWeight) n = n.slice(0, -".weight".length);
  n = n.replace(
    /\.(weight_packed|weight_scale|weight_zero_point|weight_shape|qweight|scales|qzeros|qscale|wzeros)$/i,
    "",
  );
  if (n.startsWith("model.")) n = n.slice("model.".length);
  if (n === "embed_tokens") return "embedding.weight";
  // A_log / dt_bias / bias 等非 .weight 张量：保持原名（勿强行加 .weight）
  if (!wasWeight) return n;
  return `${n}.weight`;
}

function shouldQuantize(name, shape, minCols, gs) {
  if (shape.length !== 2 || shape[1] < minCols || shape[1] % gs !== 0) return false;
  const n = name.toLowerCase();
  if (n.endsWith("embed_tokens.weight") || n.endsWith("embedding.weight")) return false;
  if (n.endsWith("lm_head.weight")) return false;
  return true;
}

function tensorToF32(buf, dtype, count) {
  if (dtype === "F32") {
    const out = new Float32Array(count);
    for (let i = 0; i < count; ++i) out[i] = buf.readFloatLE(i * 4);
    return out;
  }
  if (dtype === "F16") {
    const out = new Float32Array(count);
    for (let i = 0; i < count; ++i) out[i] = f16BitsToF32(buf.readUInt16LE(i * 2));
    return out;
  }
  if (dtype === "BF16") {
    const out = new Float32Array(count);
    for (let i = 0; i < count; ++i) {
      const u = buf.readUInt16LE(i * 2) << 16;
      const tmp = new Uint32Array([u]);
      out[i] = new Float32Array(tmp.buffer)[0];
    }
    return out;
  }
  if (dtype === "I32" || dtype === "U32") {
    const out = new Float32Array(count);
    for (let i = 0; i < count; ++i) out[i] = buf.readInt32LE(i * 4);
    return out;
  }
  if (dtype === "I8") {
    const out = new Float32Array(count);
    for (let i = 0; i < count; ++i) out[i] = buf.readInt8(i);
    return out;
  }
  if (dtype === "U8") {
    const out = new Float32Array(count);
    for (let i = 0; i < count; ++i) out[i] = buf.readUInt8(i);
    return out;
  }
  throw new Error(`unsupported safetensors dtype ${dtype}`);
}

function inferLayout(len, M, ng) {
  if (len === M * ng) return "m_g";
  if (len === ng * M) return "g_m";
  if (len === ng) return "ng";
  if (len === M) return "m";
  if (len === 1) return "1";
  throw new Error(`unexpected tensor length ${len} for M=${M} ng=${ng}`);
}

/** config/CLI 的 group_size 可能与真实 weight_scale 不一致；以 scale 元素数为准。 */
function resolveScaleGroup(scaleCount, M, K, gsHint) {
  const match = (ng, layout) => {
    if (ng <= 0 || K % ng !== 0) return null;
    if (layout === "m_g" && scaleCount === M * ng) return { ng, gs: K / ng, layout };
    if (layout === "g_m" && scaleCount === ng * M) return { ng, gs: K / ng, layout };
    if (layout === "ng" && scaleCount === ng) return { ng, gs: K / ng, layout };
    if (layout === "m" && scaleCount === M && ng === 1) return { ng: 1, gs: K, layout: "m" };
    if (layout === "1" && scaleCount === 1 && ng === 1) return { ng: 1, gs: K, layout: "1" };
    return null;
  };
  if (gsHint > 0 && K % gsHint === 0) {
    const ng = K / gsHint;
    for (const layout of ["m_g", "g_m", "ng", "m", "1"]) {
      const r = match(ng, layout);
      if (r) return r;
    }
  }
  if (M > 0 && scaleCount % M === 0) {
    const r = match(scaleCount / M, "m_g");
    if (r) return r;
  }
  // 穷举能整除 K 的 ng（K 通常 ≤ 数万）
  for (let ng = 1; ng <= K; ++ng) {
    if (K % ng !== 0) continue;
    for (const layout of ["m_g", "g_m", "ng"]) {
      const r = match(ng, layout);
      if (r) return r;
    }
  }
  throw new Error(
    `cannot resolve group from scale length=${scaleCount} M=${M} K=${K} gsHint=${gsHint}`,
  );
}

function tensorElemCount(entry, buf) {
  if (entry.shape && entry.shape.length) {
    return entry.shape.reduce((a, b) => a * b, 1);
  }
  const dt = entry.dtype;
  if (dt === "F16" || dt === "BF16") return buf.length / 2;
  if (dt === "I8" || dt === "U8") return buf.length;
  return buf.length / 4;
}

function pickScale(values, M, ng, m, g, layout) {
  if (layout === "m_g") return values[m * ng + g];
  if (layout === "g_m") return values[g * M + m];
  if (layout === "ng") return values[g];
  if (layout === "m") return values[m];
  if (layout === "1") return values[0];
  throw new Error(`bad scale layout ${layout}`);
}

function unpackCtNibble(packed, M, K, m, k) {
  const packFactor = 8;
  const colsPacked = Math.ceil(K / packFactor);
  const view = new Int32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const pc = Math.floor(k / packFactor);
  const val = view[m * colsPacked + pc];
  return (val >>> ((k % packFactor) * 4)) & 0xf;
}

function unpackAwqNibble(qweight, M, K, m, k) {
  const packFactor = 8;
  const colsPacked = Math.ceil(M / packFactor);
  const view = new Int32Array(qweight.buffer, qweight.byteOffset, qweight.byteLength / 4);
  const pm = Math.floor(m / packFactor);
  const val = view[k * colsPacked + pm];
  return (val >>> ((m % packFactor) * 4)) & 0xf;
}

function dequantCt(packed, scales, zeros, M, K, gs, scaleDt, zpDt, symmetric) {
  const ng = K / gs;
  const scaleCount =
    scaleDt === "F16" || scaleDt === "BF16" ? scales.length / 2 : scales.length / 4;
  const scaleF = tensorToF32(scales, scaleDt, scaleCount);
  const scaleLayout = inferLayout(scaleF.length, M, ng);
  const zpCount = zeros ? (zpDt === "I8" ? zeros.length : zeros.length / 4) : 0;
  const zpF = zeros ? tensorToF32(zeros, zpDt, zpCount) : null;
  const zpLayout = zpF ? inferLayout(zpF.length, M, ng) : null;
  const W = new Float32Array(M * K);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) {
      const g = Math.floor(k / gs);
      const q = unpackCtNibble(packed, M, K, m, k);
      const sc = pickScale(scaleF, M, ng, m, g, scaleLayout);
      if (symmetric) W[m * K + k] = (q - 7) * sc;
      else {
        const zp = zpF ? pickScale(zpF, M, ng, m, g, zpLayout) : 0;
        W[m * K + k] = (q - zp) * sc;
      }
    }
  }
  return W;
}

function dequantAwq(qweight, scales, qzeros, M, K, gs, scaleDt, symmetric) {
  const ng = K / gs;
  const scaleCount =
    scaleDt === "F16" || scaleDt === "BF16" ? scales.length / 2 : scales.length / 4;
  const scaleF = tensorToF32(scales, scaleDt, scaleCount);
  const scaleLayout = inferLayout(scaleF.length, M, ng);
  const zeroF = qzeros ? tensorToF32(qzeros, "I32", qzeros.length / 4) : null;
  const W = new Float32Array(M * K);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) {
      const g = Math.floor(k / gs);
      const q = unpackAwqNibble(qweight, M, K, m, k);
      const sc = pickScale(scaleF, M, ng, m, g, scaleLayout);
      if (symmetric) W[m * K + k] = (q - 8) * sc;
      else {
        const zpIdx = Math.floor(k / gs) * Math.ceil(M / 8) + Math.floor(m / 8);
        const zp = zeroF ? zeroF[zpIdx] & 0xf : 0;
        W[m * K + k] = (q - zp) * sc;
      }
    }
  }
  return W;
}

function groupQuantBases(index, format) {
  const bases = new Map();
  if (format === "compressed-tensors") {
    for (const key of Object.keys(index)) {
      if (!key.endsWith(".weight_packed")) continue;
      const base = key.slice(0, -".weight_packed".length);
      if (!bases.has(base)) bases.set(base, {});
      bases.get(base).packed = key;
    }
    for (const [base, g] of bases) {
      for (const suf of ["weight_scale", "weight_zero_point", "weight_shape"]) {
        const k = `${base}.${suf}`;
        if (index[k]) g[suf] = k;
      }
    }
  } else {
    for (const key of Object.keys(index)) {
      if (!key.endsWith(".qweight")) continue;
      const base = key.slice(0, -".qweight".length);
      if (!bases.has(base)) bases.set(base, {});
      bases.get(base).packed = key;
    }
    for (const [base, g] of bases) {
      for (const [suf, hk] of [
        ["scales", ".scales"],
        ["qzeros", ".qzeros"],
      ]) {
        const k = `${base}${hk}`;
        if (index[k]) g[suf] = k;
      }
    }
  }
  return bases;
}

function passDtypeCode(stDt) {
  if (stDt === "BF16") return 1;
  if (stDt === "F16") return 2;
  return 1;
}

async function main() {
  const opt = parseArgs();
  if (!fs.existsSync(opt.src)) throw new Error(`HF dir not found: ${opt.src}`);
  const cfg = loadConfig(opt.src);
  const cfgOpts = quantOptsFromConfig(cfg);
  const gs = opt.groupSize > 0 ? opt.groupSize : cfgOpts.groupSize;
  const index = await buildIndex(opt.src);
  const format = detectFormat(index, cfgOpts, opt.format);
  const scheme = cfgOpts.symmetric ? SCHEME_AWQ : SCHEME_GPTQ;

  console.log(`[import-awq] format=${format} scheme=${cfgOpts.symmetric ? "awq" : "gptq"} group=${gs}`);
  console.log(`[import-awq] indexing ${Object.keys(index).length} tensors`);

  const tmpDir = path.join(path.dirname(opt.out), `.import-awq-tmp-${process.pid}`);
  fs.mkdirSync(tmpDir, { recursive: true });
  const items = [];
  const seen = new Set();
  let nQ = 0;
  let nPass = 0;
  let nSkip = 0;
  let fileGs = gs;
  let gsWarned = false;

  try {
    const qBases = groupQuantBases(index, format);
    for (const [base, g] of [...qBases.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
      const qlwcName = mapQlwcName(base);
      if (seen.has(qlwcName)) continue;
      const packedE = index[g.packed];
      let M;
      let K;
      try {
        if (format === "compressed-tensors") {
          [M, K] = inferCtMk(packedE, g.weight_shape ? index[g.weight_shape] : null);
        } else if (g.weight_shape && index[g.weight_shape]) {
          [M, K] = readWeightShape(index[g.weight_shape]);
        } else {
          K = packedE.shape[0];
          M = packedE.shape[1] * 8;
        }
      } catch (e) {
        throw new Error(`${base}: ${e.message || e}`);
      }
      if (!shouldQuantize(qlwcName, [M, K], opt.minCols, gs)) {
        if (nSkip < 5) {
          console.warn(
            `[import-awq] skip ${qlwcName} shape=[${M},${K}] (minCols=${opt.minCols} gs=${gs})`,
          );
        }
        nSkip += 1;
        continue;
      }

      const scaleKey = g.weight_scale || g.scales;
      if (!scaleKey || !index[scaleKey]) throw new Error(`missing scales for ${base}`);
      const scaleE = index[scaleKey];
      const zpKey = g.weight_zero_point || g.qzeros;
      const zpE = zpKey && index[zpKey] ? index[zpKey] : null;
      const tag = `t${nQ}`;
      const qres =
        format === "compressed-tensors"
          ? importCtToQlwc(packedE, scaleE, zpE, M, K, gs, cfgOpts.symmetric, tmpDir, tag)
          : importAwqToQlwc(packedE, scaleE, zpE, M, K, gs, cfgOpts.symmetric, tmpDir, tag);
      const usedGs = qres.gs ?? gs;
      if (nQ === 0) {
        fileGs = usedGs;
        if (usedGs !== gs) {
          console.warn(
            `[import-awq] weight_scale implies group_size=${usedGs} (config/CLI=${gs}); using ${usedGs}`,
          );
          gsWarned = true;
        }
      } else if (usedGs !== fileGs) {
        throw new Error(`mixed group sizes in checkpoint: ${fileGs} vs ${usedGs} at ${qlwcName}`);
      }
      items.push({
        name: qlwcName,
        kind: 1,
        shape: [M, K],
        q: qres.q,
        scales: qres.scales,
        zeros: qres.zeros,
      });
      seen.add(qlwcName);
      nQ += 1;
      if (nQ % 10 === 0) process.stderr.write(`\r[import-awq] packed ${nQ} int4 tensors...`);
    }
    if (nQ >= 10) process.stderr.write("\n");

    for (const key of Object.keys(index).sort()) {
      if (key.endsWith(".weight_packed") || key.endsWith(".qweight")) continue;
      if (/\.(weight_scale|weight_zero_point|weight_shape|scales|qzeros|qscale|wzeros)$/.test(key)) continue;
      const e = index[key];
      const stDt = e.dtype;
      // 透传：.weight / A_log / dt_bias / bias 等 BF16|F16|F32
      if (stDt !== "BF16" && stDt !== "F16" && stDt !== "F32") continue;
      const qlwcName = mapQlwcName(key);
      if (seen.has(qlwcName)) continue;
      const shape = e.shape;
      const nbytes = e.end - e.start;
      let data;
      if (stDt === "BF16" || stDt === "F16") {
        data = blobFromSlice(e, nbytes);
      } else if (nbytes <= MAX_BUF) {
        data = blobFromBuffer(convertF32toBf16(readTensor(e), shape.reduce((a, b) => a * b, 1)));
      } else {
        throw new Error(`F32 passthrough too large (${(nbytes / 2 ** 30).toFixed(2)} GiB): ${key}`);
      }
      items.push({
        name: qlwcName,
        kind: 0,
        shape,
        pass_dtype: passDtypeCode(stDt === "F32" ? "BF16" : stDt),
        data,
      });
      seen.add(qlwcName);
      nPass += 1;
    }

    if (items.length === 0) throw new Error("no tensors imported — check HF dir and format");
    if (format === "compressed-tensors" || format === "autoawq") {
      const nPacked = [...qBases.keys()].length;
      if (nQ === 0 && nPacked > 0) {
        throw new Error(
          `imported int4=0 but found ${nPacked} packed weight bases — ` +
            `likely bad weight_shape parse or group_size=${gs} mismatch; refusing passthrough-only QLWC`,
        );
      }
    }
    writeQlwc(opt.out, scheme, fileGs, items);
    const sizeG = fs.statSync(opt.out).size / 1024 ** 3;
    console.log(`[import-awq] int4=${nQ} passthrough=${nPass} group_size=${fileGs}${gsWarned ? " (from scales)" : ""}`);
    console.log(`[import-awq] wrote ${opt.out} (${sizeG.toFixed(2)} GiB)`);
  } finally {
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  }
}

function convertF32toBf16(buf, n) {
  const out = Buffer.alloc(n * 2);
  for (let i = 0; i < n; ++i) {
    const f = buf.readFloatLE(i * 4);
    const tmp = new Float32Array([f]);
    const u = new Uint32Array(tmp.buffer)[0];
    out.writeUInt16LE((u + 0x7fff + ((u >> 16) & 1)) >> 16, i * 2);
  }
  return out;
}

try {
  await main();
} catch (e) {
  console.error("\nERROR:", e.message || e);
  process.exit(1);
}
