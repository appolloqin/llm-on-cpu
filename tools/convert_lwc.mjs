#!/usr/bin/env node
// llm-on-cpu :: tools/convert_lwc.mjs
// HF safetensors → LWC v1 —— convert_lwc.py 的 Node 对等实现(零 npm 依赖)。
// 原理: 转换只"搬字节"不解释数值 → safetensors 头(8B 长度 + JSON)原生可解析,
//       无需 numpy/torch; bf16/f16/f32 一视同仁。
// 与 C++ src/weights/lwc_format.cpp 目录格式严格一致; checksum 落 0 哨兵,
// 由 lwc_verify --update 回填(与 Python 版工作流相同)。
//
// 用法:
//   node tools/convert_lwc.mjs --src models/qwen3827-hf --out models/qwen3827.lwc
//   node tools/convert_lwc.mjs --src ... --out ... --limit-experts 4 --verify
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

const ALIGN = 4096;
const MAGIC = Buffer.from("LWC1");
const DTYPE_CODE = { BF16: 1, F16: 2, F32: 3 };
const TORCH2LWC = { bfloat16: "BF16", float16: "F16", float32: "F32" };
const ST2LWC = { BF16: "BF16", F16: "F16", F32: "F32", BOOL: null }; // safetensors 头 dtype

/** config 可能无顶层 torch_dtype(如 Qwen3.5 把 dtype 放在 text_config) */
function dtypeFromConfig(cfg) {
  const raw =
    cfg.torch_dtype ??
    cfg.dtype ??
    cfg.text_config?.torch_dtype ??
    cfg.text_config?.dtype;
  if (raw == null) return null;
  return TORCH2LWC[String(raw).toLowerCase()] ?? null;
}
const PART_MAP = { gate_proj: "gate", up_proj: "up", down_proj: "down" };
const EXPERT_RE =
  /^(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$/;

// ---- fnv1a64(BigInt; 仅用于小体积 catalog, 性能无碍) ----
function fnv1a64(buf) {
  let h = 0xcbf29ce484222325n;
  const P = 0x100000001b3n, M = (1n << 64n) - 1n;
  for (const b of buf) h = ((h ^ BigInt(b)) * P) & M;
  return h;
}

// ---- dtype 转换: safetensors 实测 dtype 可能与目标(header)不一致时需逐元素下转 ----
// 关键点: LWC 是单 dtype 容器, 头声明的 dtype 决定 C++ 端每个元素的字节数。
// 若某张量在 safetensors 中是 F32(如 Qwen3.5 的 linear_attn.norm.weight/A_log),
// 必须把它转成目标 dtype, 否则 C++ 按 header dtype 解读会读成乱码。
const ITEMSIZE = { BF16: 2, F16: 2, F32: 4 };
const fbuf = Buffer.alloc(4);
function halfToF32(h, bias) {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  let bits;
  if (e === 0) {
    if (m === 0) { bits = s; }
    else {
      let em = e + 1, mm = m;
      while ((mm & 0x400) === 0) { mm <<= 1; em--; }
      mm &= 0x3ff;
      bits = s | ((((em + 127 - bias) & 0xff) << 23) | (mm << 13)) >>> 0;
    }
  } else if (e === 0x1f) {
    bits = s | 0x7f800000 | (m << 13);
  } else {
    bits = s | ((((e + 127 - bias) & 0xff) << 23) | (m << 13)) >>> 0;
  }
  fbuf.writeUInt32LE(bits >>> 0, 0);
  return fbuf.readFloatLE(0);
}
function f32ToHalf(x, tgtDt) {
  fbuf.writeFloatLE(x, 0);
  const u = fbuf.readUInt32LE(0);
  const sign = (u >> 16) & 0x8000;
  const exp = (u >> 23) & 0xff;
  const mant = u & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (tgtDt === "BF16") {
    const r = (u + 0x7fff + ((u >> 16) & 1)) >>> 0;
    return (r >> 16) & 0xffff;
  }
  const r = (u + 0xfff + ((u >> 13) & 1)) >>> 0;
  const rexp = (r >> 23) & 0xff;
  const rmant = r & 0x7fffff;
  const e = rexp - 127 + 15;
  if (e > 30) return sign | 0x7c00;
  if (e <= 0) return sign;
  return (sign | ((e & 0x1f) << 10) | ((rmant >> 13) & 0x3ff)) >>> 0;
}
function readVal(buf, off, dt) {
  if (dt === "F32") return buf.readFloatLE(off);
  if (dt === "BF16") return halfToF32(buf.readUInt16LE(off), 127);
  if (dt === "F16") return halfToF32(buf.readUInt16LE(off), 15);
  throw new Error("unknown src dtype " + dt);
}
function writeVal(buf, off, val, dt) {
  if (dt === "F32") { buf.writeFloatLE(val, off); return 4; }
  if (dt === "BF16") { buf.writeUInt16LE(f32ToHalf(val, "BF16"), off); return 2; }
  if (dt === "F16") { buf.writeUInt16LE(f32ToHalf(val, "F16"), off); return 2; }
  throw new Error("unknown tgt dtype " + dt);
}
function convertTensor(srcBuf, srcDt, tgtDt) {
  if (srcDt === tgtDt) return srcBuf;
  const nElems = srcBuf.length / ITEMSIZE[srcDt];
  const out = Buffer.alloc(nElems * ITEMSIZE[tgtDt]);
  for (let i = 0; i < nElems; ++i)
    writeVal(out, i * ITEMSIZE[tgtDt], readVal(srcBuf, i * ITEMSIZE[srcDt], srcDt), tgtDt);
  return out;
}

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : undefined; };
  if (!a.includes("--src") || !a.includes("--out")) {
    console.error("usage: node convert_lwc.mjs --src <hf_dir> --out <file.lwc> [--limit-experts N] [--verify]");
    process.exit(2);
  }
  return {
    src: get("--src"),
    out: get("--out"),
    limitExperts: Number(get("--limit-experts") ?? 0),
    verify: a.includes("--verify"),
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
  try {
    const lenBuf = Buffer.alloc(8);
    await fh.read(lenBuf, 0, 8, 0);
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    const hdrBuf = Buffer.alloc(headerLen);
    await fh.read(hdrBuf, 0, headerLen, 8);
    return {
      fh,
      header: JSON.parse(hdrBuf.toString("utf8")),
      // safetensors 规范: data_offsets 相对"数据区起点"= 8 + header_len。
      // 漏加此基址会把 header 区当权重拷走(safetensors.get_tensor 由库内处理,
      // 这就是 Python 版此前一直正确、Node 版首实现踩坑的原因)。
      dataBase: 8 + headerLen,
    };
  } catch (e) {
    await fh.close();
    throw e;
  }
}

function mapName(orig) {
  const m = EXPERT_RE.exec(orig);
  if (m) {
    const [, l, e, part] = m;
    return { expert: { layer: +l, eid: +e, part: PART_MAP[part] }, name: `layers.${l}.experts.${e}.${PART_MAP[part]}` };
  }
  if (orig === "model.embed_tokens.weight") return { name: "embedding.weight" };
  return { name: orig.replace(/^model\./, "") };
}

function buildCatalog(dtypeCode, align, tensors, groups) {
  const chunks = [];
  const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v >>> 0); return b; };
  const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; };
  const str = (s) => { const b = Buffer.from(s, "utf8"); return Buffer.concat([u32(b.length), b]); };
  chunks.push(u32(dtypeCode), u32(align), u64(tensors.length), u64(groups.length));
  for (const [name, shape, off, nb, ck] of tensors) {
    chunks.push(str(name), u64(shape.length));
    for (const d of shape) chunks.push(u64(d));
    chunks.push(u64(off), u64(nb), u64(ck));
  }
  for (const [layer, eid, names] of groups) {
    chunks.push(u32(layer), u32(eid), u64(names.length));
    for (const n of names) chunks.push(str(n));
  }
  return Buffer.concat(chunks);
}

async function main() {
  const opt = parseArgs();
  const src = path.resolve(opt.src);
  const out = path.resolve(opt.out);

  const cfg = JSON.parse(await fsp.readFile(path.join(src, "config.json"), "utf8"));
  let lwcDtype = dtypeFromConfig(cfg); // 可能为 null; 以 safetensors 实测为准

  // ---- pass 1: 元数据收集(仅读各文件头, 不进权重数据) ----
  const entries = []; // {name, file, start(绝对偏移), end, shape, nbytes}
  const experts = new Map(); // "l,e" -> {layer, eid, parts:{}}
  let skipped = 0;
  const seenSt = new Set();
  for (const file of listSafetensors(src)) {
    const { fh, header, dataBase } = await readSafetensorsHeader(file);
    for (const [key, meta] of Object.entries(header)) {
      if (key === "__metadata__") continue;
      const stDt = ST2LWC[meta.dtype] ?? null;
      if (stDt) {
        seenSt.add(stDt);
        // 权重实测优先于 config 缺省(旧逻辑缺 torch_dtype 时会误默认 F16)
        if (!lwcDtype || (stDt === "BF16" && lwcDtype !== "BF16")) lwcDtype = stDt;
      }
      const { expert, name } = mapName(key);
      if (expert) {
        if (opt.limitExperts && expert.eid >= opt.limitExperts) { skipped++; continue; }
        const k = `${expert.layer},${expert.eid}`;
        if (!experts.has(k)) experts.set(k, { layer: expert.layer, eid: expert.eid, parts: {} });
        experts.get(k).parts[expert.part] = name;
      }
      const [s, e] = meta.data_offsets;
      entries.push({
        name, file, stDt,
        start: dataBase + s,
        end: dataBase + e,
        shape: meta.shape,
        nbytes: e - s,
      });
    }
    fh.close(); // 头读取完毕; 数据区在 pass3 重新按需打开
  }
  if (!lwcDtype) lwcDtype = "F16"; // 最终兜底
  if (![...seenSt].every((d) => d === lwcDtype) && seenSt.size)
    console.warn(`[convert] warn: mixed safetensors dtypes=${[...seenSt]} header=${lwcDtype}`);

  const order = { gate: 0, up: 1, down: 2 };
  const isExpert = (e) => /^layers\.\d+\.experts\.\d+\.(gate|up|down)$/.test(e.name);
  const dense = entries.filter((e) => !isExpert(e));
  const expl = entries.filter(isExpert).sort((a, b) => {
    const pa = /(\d+)\.experts\.(\d+)\.(gate|up|down)$/.exec(a.name);
    const pb = /(\d+)\.experts\.(\d+)\.(gate|up|down)$/.exec(b.name);
    return +pa[1] - +pb[1] || +pa[2] - +pb[2] || order[pa[3]] - order[pb[3]];
  });
  const ordered = [...dense, ...expl];

  // ---- pass 2: 布局(目录长度与 offset 取值无关 → 两轮构建) ----
  // 目标字节数按 header dtype 计算; 源 dtype 与 header 不一致时会在 pass3 逐元素下转。
  const convInfo = ordered.map((e) => {
    const elems = e.shape.reduce((a, b) => a * b, 1);
    return { stDt: e.stDt, srcBytes: e.nbytes, outBytes: elems * ITEMSIZE[lwcDtype] };
  });
  const tensorsMeta = ordered.map((e, i) => [e.name, e.shape, 0, convInfo[i].outBytes, 0]);
  const groupsMeta = [...experts.values()]
    .sort((a, b) => a.layer - b.layer || a.eid - b.eid)
    .filter((g) => g.parts.gate && g.parts.up && g.parts.down)
    .map((g) => [g.layer, g.eid, [g.parts.gate, g.parts.up, g.parts.down]]);
  const probe = buildCatalog(DTYPE_CODE[lwcDtype], ALIGN, tensorsMeta, groupsMeta);
  let off = BigInt(ALIGN) * ((BigInt(24 + probe.length) + BigInt(ALIGN) - 1n) / BigInt(ALIGN));
  for (const tm of tensorsMeta) {
    tm[2] = off;
    off = (off + BigInt(tm[3]) + BigInt(ALIGN) - 1n) / BigInt(ALIGN) * BigInt(ALIGN);
  }
  const catalog = buildCatalog(DTYPE_CODE[lwcDtype], ALIGN, tensorsMeta, groupsMeta);
  const crc = fnv1a64(catalog);
  const preface = Buffer.concat([
    MAGIC,
    (() => { const b = Buffer.alloc(4); b.writeUInt32LE(1); return b; })(),
    (() => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(catalog.length)); return b; })(),
    (() => { const b = Buffer.alloc(8); b.writeBigUInt64LE(crc); return b; })(),
  ]);

  // ---- pass 3: 落盘(preface+catalog, 再逐张量按真实偏移搬字节) ----
  await fsp.mkdir(path.dirname(out), { recursive: true });
  const w = await fsp.open(out, "w");
  try {
    await w.write(preface, 0, preface.length, 0);
    await w.write(catalog, 0, catalog.length, preface.length);
    const srcHandles = new Map();
    const rbuf = Buffer.alloc(1 << 20);
    for (let i = 0; i < ordered.length; ++i) {
      const e = ordered[i];
      const ci = convInfo[i];
      if (!srcHandles.has(e.file)) srcHandles.set(e.file, await fsp.open(e.file, "r"));
      const rh = srcHandles.get(e.file);
      const dst = Number(tensorsMeta[i][2]);
      if (ci.stDt && ci.stDt !== lwcDtype) {
        // 源 dtype 与 header 不一致: 整张量读入后逐元素下转再写出
        const srcBuf = Buffer.alloc(ci.srcBytes);
        const { bytesRead } = await rh.read(srcBuf, 0, ci.srcBytes, e.start);
        if (bytesRead !== ci.srcBytes) throw new Error(`short read source: ${e.name}`);
        if (srcBuf.length / ITEMSIZE[ci.stDt] !== ci.outBytes / ITEMSIZE[lwcDtype])
          throw new Error(`elem count mismatch: ${e.name}`);
        const outBuf = convertTensor(srcBuf, ci.stDt, lwcDtype);
        await w.write(outBuf, 0, outBuf.length, dst);
      } else {
        let pos = e.start, remaining = e.nbytes;
        while (remaining > 0) {
          const want = Math.min(remaining, rbuf.length);
          const { bytesRead } = await rh.read(rbuf, 0, want, pos);
          if (bytesRead <= 0) throw new Error(`short read source: ${e.name}`);
          await w.write(rbuf, 0, bytesRead, dst);
          pos += bytesRead; dst += bytesRead; remaining -= bytesRead;
        }
      }
    }
    for (const h of srcHandles.values()) await h.close();
  } finally {
    await w.close();
  }

  const totalGiB = convInfo.reduce((s, c) => s + c.outBytes, 0) / 2 ** 30;
  console.log(`[convert] dtype=${lwcDtype} tensors=${ordered.length} ` +
    `(dense=${dense.length}, expert=${expl.length}) groups=${groupsMeta.length} ` +
    `size=${totalGiB.toFixed(2)} GiB`);
  if (skipped) console.log(`[convert] --limit-experts: skipped ${skipped} expert tensors`);
  console.log(`[convert] wrote ${out}`);
  console.log(`[next ] .\\build\\msvc-x64\\bin\\lwc_verify.exe ${out} --update   # Win; Linux: ./build/release/bin/lwc_verify`);

  if (opt.verify) {
    const head = Buffer.alloc(24);
    const fd = await fsp.open(out, "r");
    await fd.read(head, 0, 24, 0);
    const catBack = Buffer.alloc(catalog.length);
    await fd.read(catBack, 0, catalog.length, 24);
    await fd.close();
    if (!head.subarray(0, 4).equals(MAGIC)) throw new Error("magic mismatch");
    if (head.readBigUInt64LE(8) !== BigInt(catalog.length)) throw new Error("catalog len mismatch");
    if (head.readBigUInt64LE(16) !== fnv1a64(catBack)) throw new Error("catalog crc mismatch");
    if (!catBack.equals(catalog)) throw new Error("catalog bytes differ");
    console.log("[verify] header/catalog structure OK");
  }
}

main().catch((e) => { console.error("ERROR:", e.message); process.exit(1); });
