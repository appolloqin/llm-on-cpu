#!/usr/bin/env node
/**
 * tools/glm/convert_glm_lwc.mjs
 * HF safetensors (GLM-5.3-Flash text) → BF16 GLMQ v2
 * Independent of tools/convert_lwc.mjs (Qwen/DeepSeek).
 *
 * Usage:
 *   node tools/glm/convert_glm_lwc.mjs --src models/GLM-5.3-Flash-hf --out models/GLM-5.3-Flash.bf16.glmq
 *   node tools/glm/convert_glm_lwc.mjs --src ... --out ... --limit-experts 8 --limit-layers 4
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

const MAGIC = Buffer.from("GLMQ");
const ITEMSIZE = { BF16: 2, F16: 2, F32: 4 };
const ST_OK = new Set(["BF16", "F16", "F32"]);
const HEADER_SIZE = 88;
const REC_SIZE = 132;

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--src") || !a.includes("--out")) {
    console.error(
      "usage: node tools/glm/convert_glm_lwc.mjs --src <hf_dir> --out <file.glmq> [--limit-experts N] [--limit-layers N]",
    );
    process.exit(2);
  }
  return {
    src: path.resolve(get("--src")),
    out: path.resolve(get("--out")),
    limitExperts: Number(get("--limit-experts") ?? 0),
    limitLayers: Number(get("--limit-layers") ?? 0),
  };
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
  const lenBuf = Buffer.alloc(8);
  await fh.read(lenBuf, 0, 8, 0);
  const headerLen = Number(lenBuf.readBigUInt64LE(0));
  const hdrBuf = Buffer.alloc(headerLen);
  await fh.read(hdrBuf, 0, headerLen, 8);
  return { fh, header: JSON.parse(hdrBuf.toString("utf8")), dataBase: 8 + headerLen };
}

/** Normalize HF names → GLMQ catalog names (text tower only). */
function mapName(orig) {
  let n = orig;
  if (n.startsWith("model.language_model.")) n = n.slice("model.language_model.".length);
  else if (n.startsWith("language_model.")) n = n.slice("language_model.".length);
  else if (n.startsWith("model.")) n = n.slice("model.".length);
  if (n.startsWith("visual.") || n.startsWith("vision.") || n.startsWith("model.visual")) return null;
  return n;
}

function halfToF32(h, bias) {
  const fbuf = Buffer.alloc(4);
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  let bits;
  if (e === 0) {
    if (m === 0) bits = s;
    else {
      let em = e + 1,
        mm = m;
      while ((mm & 0x400) === 0) {
        mm <<= 1;
        em--;
      }
      mm &= 0x3ff;
      bits = s | ((((em + 127 - bias) & 0xff) << 23) | (mm << 13));
    }
  } else if (e === 0x1f) bits = s | 0x7f800000 | (m << 13);
  else bits = s | ((((e + 127 - bias) & 0xff) << 23) | (m << 13));
  fbuf.writeUInt32LE(bits >>> 0, 0);
  return fbuf.readFloatLE(0);
}

function f32ToBf16(x) {
  const fbuf = Buffer.alloc(4);
  fbuf.writeFloatLE(x, 0);
  const u = fbuf.readUInt32LE(0);
  const r = (u + 0x7fff + ((u >> 16) & 1)) >>> 0;
  return (r >> 16) & 0xffff;
}

function toBf16Buffer(srcBuf, srcDt) {
  if (srcDt === "BF16") return Buffer.from(srcBuf);
  const n = srcBuf.length / ITEMSIZE[srcDt];
  const out = Buffer.alloc(n * 2);
  for (let i = 0; i < n; ++i) {
    let v;
    if (srcDt === "F32") v = srcBuf.readFloatLE(i * 4);
    else if (srcDt === "F16") v = halfToF32(srcBuf.readUInt16LE(i * 2), 15);
    else throw new Error("bad dtype " + srcDt);
    out.writeUInt16LE(f32ToBf16(v), i * 2);
  }
  return out;
}

function floatBits(f) {
  const b = Buffer.alloc(4);
  b.writeFloatLE(f, 0);
  return b.readUInt32LE(0);
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
  buf.fill(0, off, off + 16);
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

async function main() {
  const opt = parseArgs();
  const cfgPath = path.join(opt.src, "config.json");
  const cfg = JSON.parse(await fsp.readFile(cfgPath, "utf8"));
  const tc = cfg.text_config || cfg;

  const shards = listSafetensors(opt.src);
  if (!shards.length) throw new Error("no .safetensors under " + opt.src);

  const entries = [];
  const expertKeys = new Set();
  for (const file of shards) {
    const { fh, header, dataBase } = await readSafetensorsHeader(file);
    try {
      for (const [key, meta] of Object.entries(header)) {
        if (key === "__metadata__") continue;
        if (!ST_OK.has(meta.dtype)) continue;
        const name = mapName(key);
        if (!name) continue;
        const mExp = /^layers\.(\d+)\.mlp\.experts\.(\d+)\./.exec(name);
        if (mExp) {
          const layer = +mExp[1],
            eid = +mExp[2];
          if (opt.limitLayers && layer >= opt.limitLayers) continue;
          if (opt.limitExperts && eid >= opt.limitExperts) continue;
          expertKeys.add(`${layer}:${eid}`);
        } else {
          const mL = /^layers\.(\d+)\./.exec(name);
          if (mL && opt.limitLayers && +mL[1] >= opt.limitLayers) continue;
        }
        const [s, e] = meta.data_offsets;
        const nelem = meta.shape.reduce((a, b) => a * b, 1);
        entries.push({
          name,
          file,
          stDt: meta.dtype,
          start: dataBase + s,
          end: dataBase + e,
          shape: meta.shape,
          nbytes: nelem * 2, // always BF16 out
        });
      }
    } finally {
      await fh.close();
    }
  }
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  console.error(`[glm-convert] tensors=${entries.length} shards=${shards.length}`);

  let offset = 0;
  const catalog = entries.map((e) => {
    const rec = {
      name: e.name,
      dtype: 1,
      ndim: e.shape.length,
      shape: [e.shape[0] || 0, e.shape[1] || 0, e.shape[2] || 0, e.shape[3] || 0],
      offset,
      nbytes: e.nbytes,
    };
    offset += e.nbytes;
    return rec;
  });

  const headDim = tc.head_dim > 0 ? tc.head_dim : tc.qk_nope_head_dim || 256;
  const hdr = {
    version: 2,
    quant: 3,
    n_tensors: catalog.length,
    n_expert_groups: expertKeys.size,
    catalog_bytes: catalog.length * REC_SIZE,
    data_bytes: offset,
    hidden: tc.hidden_size || 4096,
    layers: opt.limitLayers || tc.num_hidden_layers || 0,
    vocab: tc.vocab_size || 0,
    n_heads: tc.num_attention_heads || 64,
    n_kv: tc.num_key_value_heads || tc.num_attention_heads || 64,
    head_dim: headDim,
    n_experts: opt.limitExperts || tc.n_routed_experts || 0,
    topk: tc.num_experts_per_tok || 8,
    moe_inter: tc.moe_intermediate_size || 2048,
    rms_eps_bits: floatBits(tc.rms_norm_eps || 1e-5),
  };

  await fsp.mkdir(path.dirname(opt.out), { recursive: true });
  const fd = fs.openSync(opt.out, "w");
  try {
    const preface = Buffer.alloc(HEADER_SIZE + catalog.length * REC_SIZE);
    let o = writeHeader(preface, 0, hdr);
    for (const rec of catalog) o = writeTensorRec(preface, o, rec);
    writeAt(fd, preface, 0);

    const fileCache = new Map();
    async function readSlice(file, start, end) {
      let fh = fileCache.get(file);
      if (!fh) {
        fh = await fsp.open(file, "r");
        fileCache.set(file, fh);
      }
      const n = end - start;
      const buf = Buffer.alloc(n);
      await fh.read(buf, 0, n, start);
      return buf;
    }

    let filePos = HEADER_SIZE + catalog.length * REC_SIZE;
    for (let i = 0; i < entries.length; ++i) {
      const e = entries[i];
      const raw = await readSlice(e.file, e.start, e.end);
      const bf = toBf16Buffer(raw, e.stDt);
      if (bf.length !== e.nbytes) throw new Error(`size mismatch ${e.name}`);
      writeAt(fd, bf, filePos);
      filePos += bf.length;
      if ((i + 1) % 200 === 0) console.error(`[glm-convert] packed ${i + 1}/${entries.length}`);
    }
    for (const fh of fileCache.values()) await fh.close();
  } finally {
    fs.closeSync(fd);
  }

  const meta = {
    arch: "glm53_flash",
    model_type: tc.model_type || cfg.model_type,
    layer_types: tc.layer_types || [],
    mlp_layer_types: tc.mlp_layer_types || [],
    first_k_dense_replace: tc.first_k_dense_replace ?? 0,
    mhc: !!tc.mhc,
    hc_mult: tc.hc_mult ?? 4,
    index_topk: tc.index_topk ?? 0,
    index_head_dim: tc.index_head_dim ?? 0,
    index_kpool: tc.index_kpool ?? 4,
    index_kpool_compress: tc.index_kpool_compress !== false,
    qk_nope_head_dim: tc.qk_nope_head_dim ?? 0,
    v_head_dim: tc.v_head_dim ?? 0,
    linear_attn_config: tc.linear_attn_config || null,
    n_shared_experts: tc.n_shared_experts ?? 1,
    source: opt.src,
  };
  await fsp.writeFile(opt.out + ".meta.json", JSON.stringify(meta, null, 2));
  const sizeG = fs.statSync(opt.out).size / 1024 ** 3;
  console.error(`[glm-convert] wrote ${opt.out} (${sizeG.toFixed(3)} GiB) + meta.json`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
