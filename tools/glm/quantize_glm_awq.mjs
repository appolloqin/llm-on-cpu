#!/usr/bin/env node
/**
 * tools/glm/quantize_glm_awq.mjs
 * BF16 GLMQ → AWQ INT4 GLMQ (experts / dense / shared MLP only).
 * Independent of tools/quantize_int4.mjs (QLWC).
 *
 * AWQ4 payload layout (per tensor, dtype=3):
 *   [packed_int4 qweight | f16 scales]
 *   shape[0]=M, shape[1]=K, shape[2]=group_size
 *
 * Usage:
 *   node tools/glm/quantize_glm_awq.mjs --src in.bf16.glmq --out out.awq.glmq [--group-size 128]
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

const MAGIC = Buffer.from("GLMQ");
const HEADER_SIZE = 88;
const REC_SIZE = 132;
const DTYPE_BF16 = 1;
const DTYPE_AWQ4 = 3;
const QUANT_AWQ = 1;
const QUANT_BF16 = 3;

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--src") || !a.includes("--out")) {
    console.error(
      "usage: node tools/glm/quantize_glm_awq.mjs --src <in.glmq> --out <out.awq.glmq> [--group-size 128]",
    );
    process.exit(2);
  }
  return {
    src: path.resolve(get("--src")),
    out: path.resolve(get("--out")),
    groupSize: Number(get("--group-size") ?? 128),
  };
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

function writeHeader(buf, off, hdr) {
  MAGIC.copy(buf, off);
  off += 4;
  buf.writeUInt16LE(hdr.version, off);
  off += 2;
  buf.writeUInt16LE(hdr.quant, off);
  off += 2;
  buf.writeUInt32LE(hdr.n_tensors, off);
  off += 4;
  buf.writeUInt32LE(hdr.n_expert_groups, off);
  off += 4;
  buf.writeBigUInt64LE(BigInt(hdr.catalog_bytes), off);
  off += 8;
  buf.writeBigUInt64LE(BigInt(hdr.data_bytes), off);
  off += 8;
  for (const k of [
    "hidden",
    "layers",
    "vocab",
    "n_heads",
    "n_kv",
    "head_dim",
    "n_experts",
    "topk",
    "moe_inter",
    "rms_eps_bits",
  ]) {
    buf.writeUInt32LE(hdr[k] >>> 0, off);
    off += 4;
  }
  // reserved[0..3] = group_size
  buf.writeUInt32LE(hdr.group_size >>> 0, off);
  buf.fill(0, off + 4, off + 16);
  off += 16;
  return off;
}

function writeTensorRec(buf, off, rec) {
  buf.fill(0, off, off + 96);
  Buffer.from(rec.name, "utf8").copy(buf, off, 0, Math.min(95, rec.name.length));
  off += 96;
  buf.writeUInt16LE(rec.dtype, off);
  off += 2;
  buf.writeUInt16LE(rec.ndim, off);
  off += 2;
  for (let i = 0; i < 4; ++i) {
    buf.writeUInt32LE(rec.shape[i] || 0, off);
    off += 4;
  }
  buf.writeBigUInt64LE(BigInt(rec.offset), off);
  off += 8;
  buf.writeBigUInt64LE(BigInt(rec.nbytes), off);
  off += 8;
  return off;
}

function readGlmq(pathIn) {
  const fd = fs.openSync(pathIn, "r");
  try {
    const head = Buffer.alloc(HEADER_SIZE);
    fs.readSync(fd, head, 0, HEADER_SIZE, 0);
    if (head.toString("utf8", 0, 4) !== "GLMQ") throw new Error("bad magic");
    const version = head.readUInt16LE(4);
    const quant = head.readUInt16LE(6);
    const n_tensors = head.readUInt32LE(8);
    const n_expert_groups = head.readUInt32LE(12);
    const catalog_bytes = Number(head.readBigUInt64LE(16));
    const data_bytes = Number(head.readBigUInt64LE(24));
    const geo = {};
    let o = 32;
    for (const k of [
      "hidden",
      "layers",
      "vocab",
      "n_heads",
      "n_kv",
      "head_dim",
      "n_experts",
      "topk",
      "moe_inter",
      "rms_eps_bits",
    ]) {
      geo[k] = head.readUInt32LE(o);
      o += 4;
    }
    const catBuf = Buffer.alloc(catalog_bytes);
    fs.readSync(fd, catBuf, 0, catalog_bytes, HEADER_SIZE);
    const tensors = [];
    for (let i = 0; i < n_tensors; ++i) {
      const p = i * REC_SIZE;
      const name = catBuf.toString("utf8", p, p + 96).replace(/\0+$/, "");
      const dtype = catBuf.readUInt16LE(p + 96);
      const ndim = catBuf.readUInt16LE(p + 98);
      const shape = [
        catBuf.readUInt32LE(p + 100),
        catBuf.readUInt32LE(p + 104),
        catBuf.readUInt32LE(p + 108),
        catBuf.readUInt32LE(p + 112),
      ];
      const offset = Number(catBuf.readBigUInt64LE(p + 116));
      const nbytes = Number(catBuf.readBigUInt64LE(p + 124));
      tensors.push({ name, dtype, ndim, shape, offset, nbytes });
    }
    return { fd, version, quant, n_expert_groups, data_bytes, geo, tensors, data0: HEADER_SIZE + catalog_bytes };
  } catch (e) {
    fs.closeSync(fd);
    throw e;
  }
}

function bf16ToF32(h) {
  const u = h << 16;
  const fbuf = Buffer.alloc(4);
  fbuf.writeUInt32LE(u >>> 0, 0);
  return fbuf.readFloatLE(0);
}

function f32ToF16Bits(x) {
  const fbuf = Buffer.alloc(4);
  fbuf.writeFloatLE(x, 0);
  const bits = fbuf.readUInt32LE(0);
  const sign = (bits >>> 16) & 0x8000;
  let exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;
  if (exp === 255) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp === 0) return sign;
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) return sign;
  let half = (exp << 10) | (mant >>> 13);
  if (mant & 0x1000) half += 1;
  return sign | half;
}

function packInt4(q, M, K) {
  const rb = (K + 1) >> 1;
  const out = Buffer.alloc(M * rb);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) {
      const qi = q[m * K + k] & 0xf;
      const bi = m * rb + (k >> 1);
      if (k & 1) out[bi] = (out[bi] & 0x0f) | (qi << 4);
      else out[bi] = (out[bi] & 0xf0) | qi;
    }
  }
  return out;
}

function quantAwqSym(W, M, K, groupSize) {
  if (K % groupSize !== 0) throw new Error("K % group_size != 0");
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
  const qPacked = packInt4(q, M, K);
  const sBuf = Buffer.alloc(scales.length * 2);
  for (let i = 0; i < scales.length; ++i) sBuf.writeUInt16LE(f32ToF16Bits(scales[i]), i * 2);
  return Buffer.concat([qPacked, sBuf]);
}

function shouldQuantize(name, shape, gs) {
  if (shape[0] <= 0 || shape[1] <= 0) return false;
  if (shape[1] % gs !== 0) return false;
  // experts / shared / dense MLP linears only
  if (/mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$/.test(name)) return true;
  if (/mlp\.shared_experts\.(gate|up|down)_proj\.weight$/.test(name)) return true;
  if (/mlp\.(gate|up|down)_proj\.weight$/.test(name) && !name.includes("experts")) return true;
  return false;
}

async function main() {
  const opt = parseArgs();
  const gs = opt.groupSize;
  const src = readGlmq(opt.src);
  if (src.quant !== QUANT_BF16 && src.quant !== QUANT_AWQ) {
    console.error("expected BF16 (or re-pack AWQ) GLMQ, quant=", src.quant);
    process.exit(1);
  }

  const outItems = [];
  let nQ = 0,
    nPass = 0;
  let dataOff = 0;

  for (const t of src.tensors) {
    const raw = Buffer.alloc(t.nbytes);
    fs.readSync(src.fd, raw, 0, t.nbytes, src.data0 + t.offset);
    const M = t.shape[0],
      K = t.shape[1];
    if (t.dtype === DTYPE_BF16 && shouldQuantize(t.name, t.shape, gs)) {
      const W = new Float32Array(M * K);
      for (let i = 0; i < M * K; ++i) W[i] = bf16ToF32(raw.readUInt16LE(i * 2));
      const payload = quantAwqSym(W, M, K, gs);
      outItems.push({
        name: t.name,
        dtype: DTYPE_AWQ4,
        ndim: 3,
        shape: [M, K, gs, 0],
        offset: dataOff,
        nbytes: payload.length,
        payload,
      });
      dataOff += payload.length;
      nQ++;
      if (nQ % 50 === 0) process.stderr.write(`\r[glm-awq] quantized ${nQ}...`);
    } else {
      outItems.push({
        name: t.name,
        dtype: t.dtype,
        ndim: t.ndim,
        shape: t.shape,
        offset: dataOff,
        nbytes: raw.length,
        payload: raw,
      });
      dataOff += raw.length;
      nPass++;
    }
  }
  if (nQ >= 50) process.stderr.write("\n");
  fs.closeSync(src.fd);

  const hdr = {
    version: 2,
    quant: QUANT_AWQ,
    n_tensors: outItems.length,
    n_expert_groups: src.n_expert_groups,
    catalog_bytes: outItems.length * REC_SIZE,
    data_bytes: dataOff,
    ...src.geo,
    group_size: gs,
  };

  await fsp.mkdir(path.dirname(opt.out), { recursive: true });
  const fd = fs.openSync(opt.out, "w");
  try {
    const preface = Buffer.alloc(HEADER_SIZE + outItems.length * REC_SIZE);
    let o = writeHeader(preface, 0, hdr);
    for (const it of outItems) o = writeTensorRec(preface, o, it);
    writeAt(fd, preface, 0);
    let pos = HEADER_SIZE + outItems.length * REC_SIZE;
    for (let i = 0; i < outItems.length; ++i) {
      writeAt(fd, outItems[i].payload, pos);
      pos += outItems[i].payload.length;
      if ((i + 1) % 200 === 0)
        process.stderr.write(`\r[glm-awq] writing ${i + 1}/${outItems.length}...`);
    }
    if (outItems.length >= 200) process.stderr.write("\n");
  } finally {
    fs.closeSync(fd);
  }

  const metaSrc = opt.src + ".meta.json";
  if (fs.existsSync(metaSrc)) {
    await fsp.copyFile(metaSrc, opt.out + ".meta.json");
  }
  const sizeG = fs.statSync(opt.out).size / 1024 ** 3;
  console.error(`[glm-awq] quantized=${nQ} passthrough=${nPass} group=${gs}`);
  console.error(`[glm-awq] wrote ${opt.out} (${sizeG.toFixed(3)} GiB)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
