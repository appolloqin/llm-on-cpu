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
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(v));
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

function writeQlwc(filePath, scheme, groupSize, items) {
  const buildCat = (withOff, offsets) => {
    const parts = [u32(scheme), u32(groupSize), u32(ALIGN), u64(items.length)];
    for (let i = 0; i < items.length; ++i) {
      const it = items[i];
      parts.push(str(it.name), u32(it.kind), u64(it.shape.length));
      for (const d of it.shape) parts.push(u64(d));
      if (it.kind === 0) {
        parts.push(u32(it.pass_dtype));
        parts.push(u64(withOff ? offsets[i].data : 0), u64(it.data.length));
      } else {
        const o = withOff ? offsets[i] : { q: 0, s: 0, z: 0 };
        const zlen = it.zeros ? it.zeros.length * 2 : 0;
        parts.push(u32(groupSize), u64(o.q), u64(it.q.length), u64(o.s), u64(it.scales.length * 2), u64(o.z), u64(zlen));
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
  const preface = Buffer.concat([QLW_MAGIC, u32(1), u64(catalog.length), (() => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(crc);
    return b;
  })()]);
  const fileSize = Math.max(off, alignUp(preface.length + catalog.length, ALIGN));

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const fd = fs.openSync(filePath, "w");
  try {
    writeAt(fd, preface, 0);
    writeAt(fd, catalog, preface.length);
    fs.ftruncateSync(fd, fileSize);
    for (let i = 0; i < items.length; ++i) {
      const it = items[i];
      if (it.kind === 0) writeAt(fd, it.data, offsets[i].data);
      else {
        writeAt(fd, it.q, offsets[i].q);
        writeAt(fd, Buffer.from(it.scales.buffer, it.scales.byteOffset, it.scales.byteLength), offsets[i].s);
        if (it.zeros) {
          writeAt(fd, Buffer.from(it.zeros.buffer, it.zeros.byteOffset, it.zeros.byteLength), offsets[i].z);
        }
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
  const n = entry.end - entry.start;
  const buf = Buffer.alloc(n);
  const fd = fs.openSync(entry.file, "r");
  try {
    fs.readSync(fd, buf, 0, n, entry.start);
  } finally {
    fs.closeSync(fd);
  }
  return buf;
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

function mapQlwcName(hfKey) {
  let n = hfKey;
  if (n.endsWith(".weight")) n = n.slice(0, -".weight".length);
  n = n.replace(/\.(weight_packed|weight_scale|weight_zero_point|weight_shape|qweight|scales|qzeros|qscale|wzeros)$/i, "");
  if (n.startsWith("model.")) n = n.slice("model.".length);
  if (n === "embed_tokens") return "embedding.weight";
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
  const quantFn = cfgOpts.symmetric ? quantAwqSym : quantGptqAsym;

  console.log(`[import-awq] format=${format} scheme=${cfgOpts.symmetric ? "awq" : "gptq"} group=${gs}`);
  console.log(`[import-awq] indexing ${Object.keys(index).length} tensors`);

  const items = [];
  const seen = new Set();
  let nQ = 0;
  let nPass = 0;

  const qBases = groupQuantBases(index, format);
  for (const [base, g] of [...qBases.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const qlwcName = mapQlwcName(base);
    if (seen.has(qlwcName)) continue;
    const packedE = index[g.packed];
    let M, K;
    if (g.weight_shape && index[g.weight_shape]) {
      const shapeBuf = readTensor(index[g.weight_shape]);
      M = shapeBuf.readInt32LE(0);
      K = shapeBuf.readInt32LE(4);
    } else if (format === "compressed-tensors") {
      M = packedE.shape[0];
      K = packedE.shape[1] * 8;
    } else {
      K = packedE.shape[0];
      M = packedE.shape[1] * 8;
    }
    if (!shouldQuantize(qlwcName, [M, K], opt.minCols, gs)) {
      continue;
    }
    const packed = readTensor(packedE);
    const scales = g.weight_scale || g.scales ? readTensor(index[g.weight_scale || g.scales]) : null;
    const zeros = g.weight_zero_point || g.qzeros ? readTensor(index[g.weight_zero_point || g.qzeros]) : null;
    if (!scales) throw new Error(`missing scales for ${base}`);
    const scaleE = index[g.weight_scale || g.scales];
    const zpE = g.weight_zero_point || g.qzeros ? index[g.weight_zero_point || g.qzeros] : null;
    let W;
    if (format === "compressed-tensors") {
      W = dequantCt(packed, scales, zeros, M, K, gs, scaleE.dtype, zpE?.dtype ?? "I8", cfgOpts.symmetric);
    } else {
      W = dequantAwq(packed, scales, zeros, M, K, gs, scaleE.dtype, cfgOpts.symmetric);
    }
    const qres = quantFn(W, M, K, gs);
    items.push({ name: qlwcName, kind: 1, shape: [M, K], q: qres.q, scales: qres.scales, zeros: qres.zeros });
    seen.add(qlwcName);
    nQ += 1;
    if (nQ % 10 === 0) process.stderr.write(`\r[import-awq] packed ${nQ} int4 tensors...`);
  }
  if (nQ >= 10) process.stderr.write("\n");

  for (const key of Object.keys(index).sort()) {
    if (key.endsWith(".weight_packed") || key.endsWith(".qweight")) continue;
    if (/\.(weight_scale|weight_zero_point|weight_shape|scales|qzeros|qscale|wzeros)$/.test(key)) continue;
    if (!key.endsWith(".weight")) continue;
    const qlwcName = mapQlwcName(key);
    if (seen.has(qlwcName)) continue;
    const e = index[key];
    const stDt = e.dtype;
    if (stDt !== "BF16" && stDt !== "F16" && stDt !== "F32") continue;
    const shape = e.shape;
    const raw = readTensor(e);
    items.push({
      name: qlwcName,
      kind: 0,
      shape,
      pass_dtype: passDtypeCode(stDt),
      data: stDt === "BF16" || stDt === "F16" ? raw : convertF32toBf16(raw, shape.reduce((a, b) => a * b, 1)),
    });
    seen.add(qlwcName);
    nPass += 1;
  }

  if (items.length === 0) throw new Error("no tensors imported — check HF dir and format");
  writeQlwc(opt.out, scheme, gs, items);
  const sizeG = fs.statSync(opt.out).size / 1024 ** 3;
  console.log(`[import-awq] int4=${nQ} passthrough=${nPass}`);
  console.log(`[import-awq] wrote ${opt.out} (${sizeG.toFixed(2)} GiB)`);
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
