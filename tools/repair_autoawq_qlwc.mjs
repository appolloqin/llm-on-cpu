#!/usr/bin/env node
/**
 * In-place repair for QLWC produced by a buggy AutoAWQ import:
 *  1) scales stored as raw [ng,M] bytes but consumed as [M,ng]
 *  2) qweight/qzeros unpacked with sequential nibbles instead of AWQ order [0,2,4,6,1,3,5,7]
 *
 * Does NOT need the original HF safetensors. Idempotent via sidecar marker.
 *
 *   node tools/repair_autoawq_qlwc.mjs E:\model.int4.qlwc
 *   node tools/repair_autoawq_qlwc.mjs E:\model.int4.qlwc --force
 */
import fs from "node:fs";
import path from "node:path";

const AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7];
const AWQ_PACK_ORDER_INV = (() => {
  const inv = new Array(8);
  for (let i = 0; i < 8; ++i) inv[AWQ_PACK_ORDER[i]] = i;
  return inv;
})();

function u32(b, o) {
  return b.readUInt32LE(o);
}
function u64(b, o) {
  return Number(b.readBigUInt64LE(o));
}
function getStr(buf, o) {
  const n = u32(buf, o);
  o += 4;
  return { s: buf.toString("utf8", o, o + n), o: o + n };
}
function f16ToF32(u) {
  const s = (u >> 15) & 1;
  const e = (u >> 10) & 31;
  const m = u & 1023;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s ? -Infinity : Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}
function f32ToF16Bits(f) {
  if (!Number.isFinite(f)) return f !== f ? 0x7e00 : f > 0 ? 0x7c00 : 0xfc00;
  if (f === 0) return Object.is(f, -0) ? 0x8000 : 0;
  const sign = f < 0 ? 1 : 0;
  f = Math.abs(f);
  let exp = Math.floor(Math.log2(f));
  let mant = f / Math.pow(2, exp) - 1;
  exp += 15;
  if (exp <= 0) {
    mant = f / Math.pow(2, -14);
    const mi = Math.min(1023, Math.round(mant * 1024));
    return (sign << 15) | mi;
  }
  if (exp >= 31) return (sign << 15) | 0x7c00;
  const mi = Math.min(1023, Math.round(mant * 1024));
  return (sign << 15) | (exp << 10) | mi;
}

function parseArgs() {
  const a = process.argv.slice(2);
  let file = null;
  let force = false;
  for (const x of a) {
    if (x === "--force") force = true;
    else if (!x.startsWith("-")) file = x;
  }
  return { file, force };
}

function repairInt4(fd, t, stats) {
  const M = t.shape[0] | 0;
  const K = t.shape[1] | 0;
  const gs = t.gs | 0;
  if (M < 1 || K < 1 || gs < 1 || K % gs !== 0 || M % 8 !== 0 || K % 2 !== 0) {
    throw new Error(`bad shape ${t.name} shape=${t.shape} gs=${gs}`);
  }
  const ng = K / gs;
  if (t.sNb !== M * ng * 2 || t.zNb !== M * ng * 2) {
    throw new Error(
      `scale/zero size mismatch ${t.name}: sNb=${t.sNb} zNb=${t.zNb} expect=${M * ng * 2}`,
    );
  }
  if (t.qNb !== M * (K / 2)) {
    throw new Error(`q size mismatch ${t.name}: qNb=${t.qNb} expect=${M * (K / 2)}`);
  }

  const scales = Buffer.alloc(t.sNb);
  const zeros = Buffer.alloc(t.zNb);
  fs.readSync(fd, scales, 0, t.sNb, t.sOff);
  fs.readSync(fd, zeros, 0, t.zNb, t.zOff);

  const oldS = new Float32Array(M * ng);
  const oldZ = new Float32Array(M * ng);
  for (let i = 0; i < M * ng; ++i) {
    oldS[i] = f16ToF32(scales.readUInt16LE(i * 2));
    oldZ[i] = f16ToF32(zeros.readUInt16LE(i * 2));
  }

  const zpWrong = new Int32Array(M * ng);
  for (let m = 0; m < M; ++m) {
    for (let g = 0; g < ng; ++g) {
      const idx = m * ng + g;
      const sc = oldS[idx];
      let zp = Math.abs(sc) > 1e-20 ? Math.round(-oldZ[idx] / sc) : 0;
      if (zp < 0) zp = 0;
      if (zp > 15) zp = 15;
      zpWrong[idx] = zp;
    }
  }

  const newS = Buffer.alloc(t.sNb);
  const newZ = Buffer.alloc(t.zNb);
  for (let m = 0; m < M; ++m) {
    const pm = (m / 8) | 0;
    const srcLocal = AWQ_PACK_ORDER_INV[m & 7];
    for (let g = 0; g < ng; ++g) {
      const trueSc = oldS[g * M + m];
      const trueZp = zpWrong[(pm * 8 + srcLocal) * ng + g];
      const trueZ = -trueZp * trueSc;
      const o = (m * ng + g) * 2;
      newS.writeUInt16LE(f32ToF16Bits(trueSc), o);
      newZ.writeUInt16LE(f32ToF16Bits(trueZ), o);
    }
  }
  fs.writeSync(fd, newS, 0, newS.length, t.sOff);
  fs.writeSync(fd, newZ, 0, newZ.length, t.zOff);

  const rowBytes = K / 2;
  const groupBytes = 8 * rowBytes;
  const groupBuf = Buffer.alloc(groupBytes);
  const outBuf = Buffer.alloc(groupBytes);
  for (let pm = 0; pm < M / 8; ++pm) {
    const pos = t.qOff + pm * groupBytes;
    fs.readSync(fd, groupBuf, 0, groupBytes, pos);
    for (let local = 0; local < 8; ++local) {
      const srcLocal = AWQ_PACK_ORDER_INV[local];
      groupBuf.copy(outBuf, local * rowBytes, srcLocal * rowBytes, (srcLocal + 1) * rowBytes);
    }
    fs.writeSync(fd, outBuf, 0, groupBytes, pos);
  }

  stats.fixed += 1;
  stats.bytes += t.qNb + t.sNb + t.zNb;
}

function main() {
  const opt = parseArgs();
  if (!opt.file || !fs.existsSync(opt.file)) {
    console.error("usage: node tools/repair_autoawq_qlwc.mjs <file.int4.qlwc> [--force]");
    process.exit(2);
  }
  const abs = path.resolve(opt.file);
  const marker = abs + ".awq_layout_v2";
  if (fs.existsSync(marker) && !opt.force) {
    console.error(`already repaired (marker ${marker}); pass --force to run again (unsafe).`);
    process.exit(1);
  }

  const fd = fs.openSync(abs, "r+");
  const pref = Buffer.alloc(24);
  fs.readSync(fd, pref, 0, 24, 0);
  if (pref.toString("ascii", 0, 4) !== "QLW1") {
    console.error("bad QLWC magic");
    process.exit(1);
  }
  const catLen = Number(pref.readBigUInt64LE(8));
  const cat = Buffer.alloc(catLen);
  fs.readSync(fd, cat, 0, catLen, 24);

  let o = 0;
  const scheme = u32(cat, o);
  o += 4;
  const gsFile = u32(cat, o);
  o += 4;
  o += 4; // align
  const n = u64(cat, o);
  o += 8;
  console.log(`[repair] file=${abs} scheme=${scheme} gs=${gsFile} tensors=${n}`);

  const stats = { fixed: 0, skipped: 0, bytes: 0 };
  const t0 = Date.now();
  for (let i = 0; i < n; ++i) {
    const name = getStr(cat, o);
    o = name.o;
    const kind = u32(cat, o);
    o += 4;
    const nd = u64(cat, o);
    o += 8;
    const shape = [];
    for (let d = 0; d < nd; ++d) {
      shape.push(u64(cat, o));
      o += 8;
    }
    if (kind === 0) {
      o += 4 + 8 + 8;
      stats.skipped += 1;
      continue;
    }
    const gs = u32(cat, o);
    o += 4;
    const qOff = u64(cat, o);
    o += 8;
    const qNb = u64(cat, o);
    o += 8;
    const sOff = u64(cat, o);
    o += 8;
    const sNb = u64(cat, o);
    o += 8;
    const zOff = u64(cat, o);
    o += 8;
    const zNb = u64(cat, o);
    o += 8;
    if (zNb === 0) {
      stats.skipped += 1;
      continue;
    }
    repairInt4(fd, { name: name.s, shape, gs, qOff, qNb, sOff, sNb, zOff, zNb }, stats);
    if (stats.fixed % 256 === 0) {
      const sec = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`[repair] ${stats.fixed} int4 fixed (${sec}s) last=${name.s}`);
    }
  }
  fs.closeSync(fd);
  fs.writeFileSync(
    marker,
    `repaired ${new Date().toISOString()}\nfixed=${stats.fixed}\nbytes=${stats.bytes}\n`,
  );
  const sec = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(
    `[repair] done fixed=${stats.fixed} skipped_passthrough_or_nozeros=${stats.skipped} ` +
      `bytes≈${(stats.bytes / 2 ** 30).toFixed(2)} GiB in ${sec}s`,
  );
  console.log(`[repair] marker=${marker}`);
  console.log("[repair] restart llmoc_server_int4 and retest chat");
}

main();
