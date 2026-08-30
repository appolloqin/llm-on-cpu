#!/usr/bin/env node
/**
 * tools/glm/import_glm_nvfp4.mjs
 * ModelOpt weight-only NVFP4 HF dir (e.g. LibertAIDAI/GLM-5.3-Flash-NVFP4)
 * → GLMQ v2 (experts as NVFP4; rest BF16).
 *
 * HF triplets per quantized Linear:
 *   *.weight           UINT8 packed E2M1  shape [M, K/2]
 *   *.weight_scale     F8_E4M3            shape [M, K/16]
 *   *.weight_scale_2   F32                scalar / [1]
 *
 * GLMQ NVFP4 payload (dtype=4):
 *   [qweight | scales_fp8 | global_scale f32]
 *   shape[0]=M, shape[1]=K_logical, shape[2]=16
 *
 * Usage:
 *   node tools/glm/import_glm_nvfp4.mjs --src models/GLM-5.3-Flash-NVFP4 --out models/GLM.nvfp4.glmq
 *   node tools/glm/import_glm_nvfp4.mjs --src ... --out ... --limit-experts 4 --limit-layers 2
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

const MAGIC = Buffer.from("GLMQ");
const HEADER_SIZE = 88;
const REC_SIZE = 132;
const DTYPE_BF16 = 1;
const DTYPE_NVFP4 = 4;
const QUANT_NVFP4 = 2;
const ITEMSIZE = { BF16: 2, F16: 2, F32: 4, U8: 1, UINT8: 1, F8_E4M3: 1, F8E4M3FN: 1 };

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--src") || !a.includes("--out")) {
    console.error(
      "usage: node tools/glm/import_glm_nvfp4.mjs --src <HF_NVFP4_dir> --out <out.nvfp4.glmq> [--limit-experts N] [--limit-layers N]",
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

function mapName(orig) {
  let n = orig;
  if (n.startsWith("model.language_model.")) n = n.slice("model.language_model.".length);
  else if (n.startsWith("language_model.")) n = n.slice("language_model.".length);
  else if (n.startsWith("model.")) n = n.slice("model.".length);
  if (n.startsWith("visual.") || n.startsWith("vision.")) return null;
  return n;
}

function floatBits(f) {
  const b = Buffer.alloc(4);
  b.writeFloatLE(f, 0);
  return b.readUInt32LE(0);
}

function f32ToBf16(x) {
  const fbuf = Buffer.alloc(4);
  fbuf.writeFloatLE(x, 0);
  const u = fbuf.readUInt32LE(0);
  return ((u + 0x7fff + ((u >> 16) & 1)) >>> 16) & 0xffff;
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

function toBf16Buffer(srcBuf, srcDt) {
  if (srcDt === "BF16") return Buffer.from(srcBuf);
  const isz = ITEMSIZE[srcDt];
  if (!isz) throw new Error("cannot convert to bf16: " + srcDt);
  const n = srcBuf.length / isz;
  const out = Buffer.alloc(n * 2);
  for (let i = 0; i < n; ++i) {
    let v;
    if (srcDt === "F32") v = srcBuf.readFloatLE(i * 4);
    else if (srcDt === "F16") v = halfToF32(srcBuf.readUInt16LE(i * 2), 15);
    else throw new Error("bad bf16 src " + srcDt);
    out.writeUInt16LE(f32ToBf16(v), i * 2);
  }
  return out;
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

function isPackedU8(dt) {
  return dt === "U8" || dt === "UINT8";
}
function isFp8(dt) {
  return dt === "F8_E4M3" || dt === "F8E4M3FN" || dt === "F8_E4M3FN";
}

function layerOk(name, limitLayers, limitExperts) {
  const mExp = /^layers\.(\d+)\.mlp\.experts\.(\d+)\./.exec(name);
  if (mExp) {
    if (limitLayers && +mExp[1] >= limitLayers) return false;
    if (limitExperts && +mExp[2] >= limitExperts) return false;
    return true;
  }
  const mL = /^layers\.(\d+)\./.exec(name);
  if (mL && limitLayers && +mL[1] >= limitLayers) return false;
  return true;
}

async function main() {
  const opt = parseArgs();
  const cfgPath = path.join(opt.src, "config.json");
  const cfg = JSON.parse(await fsp.readFile(cfgPath, "utf8"));
  const tc = cfg.text_config || cfg;

  const shards = listSafetensors(opt.src);
  if (!shards.length) throw new Error("no .safetensors under " + opt.src);

  /** @type {Map<string, {file,start,end,dtype,shape}>} */
  const byName = new Map();
  for (const file of shards) {
    const { fh, header, dataBase } = await readSafetensorsHeader(file);
    try {
      for (const [key, meta] of Object.entries(header)) {
        if (key === "__metadata__") continue;
        const name = mapName(key);
        if (!name) continue;
        if (!layerOk(name, opt.limitLayers, opt.limitExperts)) continue;
        const [s, e] = meta.data_offsets;
        byName.set(name, {
          file,
          start: dataBase + s,
          end: dataBase + e,
          dtype: meta.dtype,
          shape: meta.shape,
        });
      }
    } finally {
      await fh.close();
    }
  }

  // Group NVFP4 triplets: base = name without .weight_scale(_2)
  const nvBases = new Set();
  for (const name of byName.keys()) {
    if (name.endsWith(".weight_scale_2")) nvBases.add(name.slice(0, -".weight_scale_2".length));
    else if (name.endsWith(".weight_scale")) nvBases.add(name.slice(0, -".weight_scale".length));
  }

  const outItems = []; // {name,dtype,ndim,shape,nbytes,file?,parts?}
  const used = new Set();
  const expertKeys = new Set();
  let nNv = 0,
    nBf = 0;

  for (const base of [...nvBases].sort()) {
    const w = byName.get(base + ".weight");
    const sc = byName.get(base + ".weight_scale");
    const sc2 = byName.get(base + ".weight_scale_2");
    if (!w || !sc || !sc2) {
      console.error(`[glm-nvfp4] incomplete triplet for ${base}, skip`);
      continue;
    }
    if (!isPackedU8(w.dtype)) {
      console.error(`[glm-nvfp4] expected U8 weight for ${base}, got ${w.dtype}`);
      continue;
    }
    const M = w.shape[0];
    // packed [M, K/2] → logical K
    const K = w.shape[1] * 2;
    const gs = 16;
    const q_bytes = w.end - w.start;
    const s_bytes = sc.end - sc.start;
    const nbytes = q_bytes + s_bytes + 4;
    const mExp = /^layers\.(\d+)\.mlp\.experts\.(\d+)\./.exec(base + ".weight");
    if (mExp) expertKeys.add(`${mExp[1]}:${mExp[2]}`);
    outItems.push({
      kind: "nvfp4",
      name: base + ".weight",
      dtype: DTYPE_NVFP4,
      ndim: 3,
      shape: [M, K, gs, 0],
      nbytes,
      w,
      sc,
      sc2,
    });
    used.add(base + ".weight");
    used.add(base + ".weight_scale");
    used.add(base + ".weight_scale_2");
    nNv++;
  }

  for (const [name, meta] of [...byName.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    if (used.has(name)) continue;
    if (name.endsWith(".weight_scale") || name.endsWith(".weight_scale_2")) continue;
    if (name.endsWith(".input_scale")) continue; // weight-only ckpt may omit; ignore if present
    if (!["BF16", "F16", "F32"].includes(meta.dtype)) {
      // skip exotic leftovers
      continue;
    }
    const nelem = meta.shape.reduce((a, b) => a * b, 1);
    outItems.push({
      kind: "bf16",
      name,
      dtype: DTYPE_BF16,
      ndim: meta.shape.length,
      shape: [meta.shape[0] || 0, meta.shape[1] || 0, meta.shape[2] || 0, meta.shape[3] || 0],
      nbytes: nelem * 2,
      meta,
    });
    nBf++;
  }

  outItems.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  let offset = 0;
  for (const it of outItems) {
    it.offset = offset;
    offset += it.nbytes;
  }

  const headDim = tc.head_dim > 0 ? tc.head_dim : tc.qk_nope_head_dim || 256;
  const hdr = {
    version: 2,
    quant: QUANT_NVFP4,
    n_tensors: outItems.length,
    n_expert_groups: expertKeys.size,
    catalog_bytes: outItems.length * REC_SIZE,
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
    group_size: 16,
  };

  console.error(`[glm-nvfp4] tensors nvfp4=${nNv} bf16=${nBf} total=${outItems.length}`);

  await fsp.mkdir(path.dirname(opt.out), { recursive: true });
  const fd = fs.openSync(opt.out, "w");
  const fileCache = new Map();
  async function readSlice(file, start, end) {
    let fh = fileCache.get(file);
    if (!fh) {
      fh = await fsp.open(file, "r");
      fileCache.set(file, fh);
    }
    const buf = Buffer.alloc(end - start);
    await fh.read(buf, 0, buf.length, start);
    return buf;
  }

  try {
    const preface = Buffer.alloc(HEADER_SIZE + outItems.length * REC_SIZE);
    let o = writeHeader(preface, 0, hdr);
    for (const it of outItems) o = writeTensorRec(preface, o, it);
    writeAt(fd, preface, 0);

    let pos = HEADER_SIZE + outItems.length * REC_SIZE;
    for (let i = 0; i < outItems.length; ++i) {
      const it = outItems[i];
      let payload;
      if (it.kind === "nvfp4") {
        const qw = await readSlice(it.w.file, it.w.start, it.w.end);
        const sc = await readSlice(it.sc.file, it.sc.start, it.sc.end);
        const sc2raw = await readSlice(it.sc2.file, it.sc2.start, it.sc2.end);
        let gscale = 1.0;
        if (sc2raw.length >= 4) gscale = sc2raw.readFloatLE(0);
        const gbuf = Buffer.alloc(4);
        gbuf.writeFloatLE(gscale, 0);
        payload = Buffer.concat([qw, sc, gbuf]);
        if (payload.length !== it.nbytes) throw new Error("nvfp4 size " + it.name);
      } else {
        const raw = await readSlice(it.meta.file, it.meta.start, it.meta.end);
        payload = toBf16Buffer(raw, it.meta.dtype);
        if (payload.length !== it.nbytes) throw new Error("bf16 size " + it.name);
      }
      writeAt(fd, payload, pos);
      pos += payload.length;
      if ((i + 1) % 200 === 0) console.error(`[glm-nvfp4] packed ${i + 1}/${outItems.length}`);
    }
  } finally {
    for (const fh of fileCache.values()) await fh.close();
    fs.closeSync(fd);
  }

  const meta = {
    arch: "glm53_flash",
    quant: "nvfp4",
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
  console.error(`[glm-nvfp4] wrote ${opt.out} (${sizeG.toFixed(3)} GiB) + meta.json`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
