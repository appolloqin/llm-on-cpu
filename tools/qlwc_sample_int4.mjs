#!/usr/bin/env node
// Sample QLWC int4 tensor: scales/zeros stats + optional compare to AutoAWQ HF.
import fs from "node:fs";
import path from "node:path";

function u32(b, o) { return b.readUInt32LE(o); }
function u64(b, o) { return Number(b.readBigUInt64LE(o)); }
function f16(u) {
  const s = (u >> 15) & 1, e = (u >> 10) & 31, m = u & 1023;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}
function getStr(buf, o) {
  const n = u32(buf, o); o += 4;
  return { s: buf.toString("utf8", o, o + n), o: o + n };
}

const qlwcPath = process.argv[2];
const want = process.argv[3] || "layers.0.mlp.experts.0.gate_proj.weight";
if (!qlwcPath) {
  console.error("usage: node tools/qlwc_sample_int4.mjs <qlwc> [tensor_name]");
  process.exit(2);
}
const fd = fs.openSync(qlwcPath, "r");
const pref = Buffer.alloc(24);
fs.readSync(fd, pref, 0, 24, 0);
const catLen = Number(pref.readBigUInt64LE(8));
const cat = Buffer.alloc(catLen);
fs.readSync(fd, cat, 0, catLen, 24);
let o = 0;
const scheme = u32(cat, o); o += 4;
const gsFile = u32(cat, o); o += 4;
o += 4; // align
const n = u64(cat, o); o += 8;
let hit = null;
for (let i = 0; i < n; ++i) {
  const name = getStr(cat, o); o = name.o;
  const kind = u32(cat, o); o += 4;
  const nd = u64(cat, o); o += 8;
  const shape = [];
  for (let d = 0; d < nd; ++d) { shape.push(u64(cat, o)); o += 8; }
  if (kind === 0) {
    o += 4 + 8 + 8;
  } else {
    const gs = u32(cat, o); o += 4;
    const qOff = u64(cat, o); o += 8;
    const qNb = u64(cat, o); o += 8;
    const sOff = u64(cat, o); o += 8;
    const sNb = u64(cat, o); o += 8;
    const zOff = u64(cat, o); o += 8;
    const zNb = u64(cat, o); o += 8;
    if (name.s === want || (!hit && name.s.includes("experts.0.gate_proj"))) {
      hit = { name: name.s, shape, gs, qOff, qNb, sOff, sNb, zOff, zNb };
      if (name.s === want) break;
    }
  }
}
if (!hit) { console.error("tensor not found"); process.exit(1); }
console.log(JSON.stringify({ scheme, gsFile, ...hit }, null, 2));
const M = hit.shape[0], K = hit.shape[1], gs = hit.gs, ng = Math.ceil(K / gs);
const scales = Buffer.alloc(hit.sNb);
fs.readSync(fd, scales, 0, hit.sNb, hit.sOff);
const zeros = hit.zNb ? Buffer.alloc(hit.zNb) : null;
if (zeros) fs.readSync(fd, zeros, 0, hit.zNb, hit.zOff);
const q = Buffer.alloc(Math.min(hit.qNb, M * (K / 2))); // may be large; read first rows only
const rowBytes = K / 2;
const rowsSample = Math.min(M, 8);
fs.readSync(fd, q, 0, rowsSample * rowBytes, hit.qOff);
fs.closeSync(fd);

let sMin = Infinity, sMax = -Infinity, zMin = Infinity, zMax = -Infinity;
const zpApprox = [];
for (let i = 0; i < M * ng; ++i) {
  const sc = f16(scales.readUInt16LE(i * 2));
  if (sc < sMin) sMin = sc;
  if (sc > sMax) sMax = sc;
  if (zeros) {
    const z = f16(zeros.readUInt16LE(i * 2));
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
    // GPTQ: zero = -zp*scale ⇒ zp ≈ -zero/scale
    if (Math.abs(sc) > 1e-12) zpApprox.push(-z / sc);
  }
}
zpApprox.sort((a, b) => a - b);
const pct = (p) => zpApprox[Math.min(zpApprox.length - 1, Math.floor(p * (zpApprox.length - 1)))];
console.log(`scales f16: min=${sMin} max=${sMax} count=${M * ng}`);
if (zeros) {
  console.log(`zeros f16: min=${zMin} max=${zMax}`);
  console.log(
    `implied_zp (-zero/scale): min=${zpApprox[0]?.toFixed(3)} p50=${pct(0.5)?.toFixed(3)} p90=${pct(0.9)?.toFixed(3)} max=${zpApprox[zpApprox.length - 1]?.toFixed(3)}`,
  );
  const near8 = zpApprox.filter((z) => Math.abs(z - 8) < 0.25).length;
  console.log(`implied_zp near 8 (±0.25): ${near8}/${zpApprox.length} (${((100 * near8) / zpApprox.length).toFixed(1)}%)`);
}
// dequant first 4 weights of row0 with gptq: w=q*scale+zero
const g0 = 0;
const sc0 = f16(scales.readUInt16LE(0));
const z0 = zeros ? f16(zeros.readUInt16LE(0)) : 0;
const b0 = q[0];
const q0 = b0 & 0xf, q1 = (b0 >> 4) & 0xf;
console.log(`row0 g0 scale=${sc0} zero=${z0} nibble0=${q0}→w=${q0 * sc0 + z0} nibble1=${q1}→w=${q1 * sc0 + z0}`);
console.log(`awq_sym would use (q-8)*scale: ${(q0 - 8) * sc0} vs gptq ${q0 * sc0 + z0}`);
