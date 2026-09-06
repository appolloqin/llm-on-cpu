#!/usr/bin/env node
// Dump QLWC header: scheme / group_size / int4 zeros presence (diagnose sticky garbage).
import fs from "node:fs";
import path from "node:path";

function u32(buf, o) {
  return buf.readUInt32LE(o);
}
function u64(buf, o) {
  return Number(buf.readBigUInt64LE(o));
}
function getStr(buf, o) {
  const n = u32(buf, o);
  o += 4;
  const s = buf.toString("utf8", o, o + n);
  o += n;
  return { s, o };
}

const file = process.argv[2];
if (!file || !fs.existsSync(file)) {
  console.error("usage: node tools/qlwc_info.mjs <file.int4.qlwc>");
  process.exit(2);
}
const fd = fs.openSync(file, "r");
const pref = Buffer.alloc(24);
fs.readSync(fd, pref, 0, 24, 0);
if (pref.toString("ascii", 0, 4) !== "QLW1") {
  console.error("bad magic");
  process.exit(1);
}
const catLen = Number(pref.readBigUInt64LE(8));
const cat = Buffer.alloc(catLen);
fs.readSync(fd, cat, 0, catLen, 24);
fs.closeSync(fd);

let o = 0;
const scheme = u32(cat, o);
o += 4;
const gs = u32(cat, o);
o += 4;
const align = u32(cat, o);
o += 4;
const n = u64(cat, o);
o += 8;
const schemeName = scheme === 1 ? "gptq_asym" : scheme === 2 ? "awq_sym_zp8" : `unknown(${scheme})`;
let nInt4 = 0;
let nWithZeros = 0;
let nNoZeros = 0;
for (let i = 0; i < n; ++i) {
  const name = getStr(cat, o);
  o = name.o;
  const kind = u32(cat, o);
  o += 4;
  const nd = u64(cat, o);
  o += 8;
  o += nd * 8;
  if (kind === 0) {
    o += 4 + 8 + 8; // dtype + off + nbytes
  } else {
    nInt4 += 1;
    o += 4; // group_size
    o += 8 + 8; // q
    o += 8 + 8; // scales
    const zOff = u64(cat, o);
    o += 8;
    const zNb = u64(cat, o);
    o += 8;
    if (zNb > 0) nWithZeros += 1;
    else nNoZeros += 1;
    void zOff;
  }
}
const sz = fs.statSync(file).size;
console.log(`file=${path.resolve(file)}`);
console.log(`size_gib=${(sz / 1024 ** 3).toFixed(2)}`);
console.log(`scheme=${schemeName} (${scheme}) group_size=${gs} align=${align}`);
console.log(`tensors=${n} int4=${nInt4} with_zeros=${nWithZeros} no_zeros=${nNoZeros}`);
if (scheme === 2 && nNoZeros === nInt4 && nInt4 > 0) {
  console.log(
    "WARN: awq_sym with no zeros — OK for symmetric CT; BAD for AutoAWQ zero_point:true (re-import).",
  );
}
if (scheme === 1 && nWithZeros === 0) {
  console.log("WARN: gptq_asym but no zeros blobs — import likely broken.");
}
if (scheme === 1 && nWithZeros > 0) {
  console.log("OK: gptq_asym + zeros present (AutoAWQ zero_point path).");
}
