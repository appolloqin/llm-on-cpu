#!/usr/bin/env node
/**
 * Ground-truth roundtrip: AutoAWQ pack → importAwqToQlwc logic → engine GPTQ dequant
 * Must match official dequantize_gemm: unpack + reverse_order + (q-zp)*scale
 *
 *   node tools/test_awq_import_roundtrip.mjs
 */
import assert from "node:assert/strict";

const AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7];
const AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]; // = INV of PACK_ORDER
const INV = (() => {
  const inv = new Array(8);
  for (let i = 0; i < 8; ++i) inv[AWQ_PACK_ORDER[i]] = i;
  return inv;
})();

function awqNibbleIndex(mLocal) {
  return INV[mLocal & 7];
}

function pickScale(values, M, ng, m, g, layout) {
  if (layout === "m_g") return values[m * ng + g];
  if (layout === "g_m") return values[g * M + m];
  throw new Error(layout);
}

function layoutFromShape(shape, M, ng, preferGmForAwq) {
  if (Array.isArray(shape) && shape.length === 2) {
    const [a, b] = shape;
    if (a === M && b === ng) return "m_g";
    if (a === ng && b === M) return "g_m";
  }
  return preferGmForAwq ? "g_m" : "m_g";
}

/** Official AutoAWQ dequantize_gemm (packing_utils.py) */
function dequantOfficial(qweight, qzeros, scalesGm, M, K, gs) {
  const ng = K / gs;
  const colsPacked = Math.ceil(M / 8);
  // unpack sequential nibbles → [K, M]
  const iweight = new Int32Array(K * M);
  const izeros = new Int32Array(ng * M);
  for (let k = 0; k < K; ++k) {
    for (let pm = 0; pm < colsPacked; ++pm) {
      const word = qweight[k * colsPacked + pm] | 0;
      for (let i = 0; i < 8; ++i) {
        const m = pm * 8 + i;
        if (m >= M) break;
        iweight[k * M + m] = (word >>> (i * 4)) & 0xf;
      }
    }
  }
  for (let g = 0; g < ng; ++g) {
    for (let pm = 0; pm < colsPacked; ++pm) {
      const word = qzeros[g * colsPacked + pm] | 0;
      for (let i = 0; i < 8; ++i) {
        const m = pm * 8 + i;
        if (m >= M) break;
        izeros[g * M + m] = (word >>> (i * 4)) & 0xf;
      }
    }
  }
  // reverse_awq_order along M
  const revW = new Int32Array(K * M);
  const revZ = new Int32Array(ng * M);
  for (let base = 0; base < M; base += 8) {
    for (let local = 0; local < 8; ++local) {
      const src = base + AWQ_REVERSE_ORDER[local];
      const dst = base + local;
      if (dst >= M || src >= M) continue;
      for (let k = 0; k < K; ++k) revW[k * M + dst] = iweight[k * M + src];
      for (let g = 0; g < ng; ++g) revZ[g * M + dst] = izeros[g * M + src];
    }
  }
  // W[K,M] = (q-zp)*scale ; return as W[M,K] for engine compare
  const W = new Float32Array(M * K);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) {
      const g = (k / gs) | 0;
      const q = revW[k * M + m];
      const zp = revZ[g * M + m];
      const sc = scalesGm[g * M + m];
      W[m * K + k] = (q - zp) * sc;
    }
  }
  return W;
}

/** Pack like AutoAWQ gemm from_linear */
function packAwq(qTrue /*[M,K]*/, zpTrue /*[M,ng]*/, scalesGm /*[ng*M]*/, M, K, gs) {
  const ng = K / gs;
  const colsPacked = Math.ceil(M / 8);
  const qweight = new Int32Array(K * colsPacked);
  const qzeros = new Int32Array(ng * colsPacked);
  for (let k = 0; k < K; ++k) {
    for (let pm = 0; pm < colsPacked; ++pm) {
      let word = 0;
      for (let i = 0; i < 8; ++i) {
        const m = pm * 8 + AWQ_PACK_ORDER[i];
        if (m >= M) continue;
        word |= (qTrue[m * K + k] & 0xf) << (i * 4);
      }
      qweight[k * colsPacked + pm] = word;
    }
  }
  for (let g = 0; g < ng; ++g) {
    for (let pm = 0; pm < colsPacked; ++pm) {
      let word = 0;
      for (let i = 0; i < 8; ++i) {
        const m = pm * 8 + AWQ_PACK_ORDER[i];
        if (m >= M) continue;
        word |= (zpTrue[m * ng + g] & 0xf) << (i * 4);
      }
      qzeros[g * colsPacked + pm] = word;
    }
  }
  return { qweight, qzeros, scalesGm, colsPacked };
}

/** Current importAwqToQlwc unpack + GPTQ zero formula */
function importAndEngineDequant(qweight, qzeros, scalesGm, M, K, gs, scaleShape) {
  const ng = K / gs;
  const colsPacked = Math.ceil(M / 8);
  const scaleLayout = layoutFromShape(scaleShape, M, ng, true);
  // scales → m_g f16 domain (keep f32 here)
  const scalesMg = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m)
    for (let g = 0; g < ng; ++g) scalesMg[m * ng + g] = pickScale(scalesGm, M, ng, m, g, scaleLayout);

  const zpF = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m) {
    const pm = (m / 8) | 0;
    const nib = awqNibbleIndex(m % 8);
    for (let g = 0; g < ng; ++g) {
      const word = qzeros[g * colsPacked + pm] | 0;
      zpF[m * ng + g] = (word >>> (nib * 4)) & 0xf;
    }
  }
  const zeros = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m)
    for (let g = 0; g < ng; ++g) {
      const sc = pickScale(scalesGm, M, ng, m, g, scaleLayout);
      zeros[m * ng + g] = -zpF[m * ng + g] * sc;
    }

  const q = new Uint8Array(M * (K / 2));
  for (let k = 0; k < K; k += 2) {
    for (let kk = 0; kk < 2; ++kk) {
      const kkAbs = k + kk;
      for (let m = 0; m < M; ++m) {
        const pm = (m / 8) | 0;
        const nib = awqNibbleIndex(m % 8);
        const qi = (qweight[kkAbs * colsPacked + pm] >>> (nib * 4)) & 0xf;
        const byteIndex = m * (K / 2) + (k / 2);
        if (kk === 0) q[byteIndex] = (q[byteIndex] & 0xf0) | (qi & 0xf);
        else q[byteIndex] = (q[byteIndex] & 0x0f) | ((qi & 0xf) << 4);
      }
    }
  }

  // Engine GPTQ: w = q * scale + zero
  const W = new Float32Array(M * K);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) {
      const g = (k / gs) | 0;
      const b = q[m * (K / 2) + ((k / 2) | 0)];
      const qi = k & 1 ? (b >> 4) & 0xf : b & 0xf;
      W[m * K + k] = qi * scalesMg[m * ng + g] + zeros[m * ng + g];
    }
  }
  return { W, scalesMg, zeros, zpF, scaleLayout };
}

/** OLD buggy import: sequential nibble + m_g scale layout */
function importOldBuggy(qweight, qzeros, scalesGm, M, K, gs) {
  const ng = K / gs;
  const colsPacked = Math.ceil(M / 8);
  const scalesMg = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m)
    for (let g = 0; g < ng; ++g) scalesMg[m * ng + g] = scalesGm[m * ng + g]; // wrong m_g on g_m data

  const zpF = new Float32Array(M * ng);
  for (let m = 0; m < M; ++m) {
    for (let g = 0; g < ng; ++g) {
      const word = qzeros[g * colsPacked + ((m / 8) | 0)] | 0;
      zpF[m * ng + g] = (word >>> ((m % 8) * 4)) & 0xf;
    }
  }
  const zeros = new Float32Array(M * ng);
  for (let i = 0; i < M * ng; ++i) zeros[i] = -zpF[i] * scalesMg[i];

  const q = new Uint8Array(M * (K / 2));
  for (let k = 0; k < K; k += 2) {
    for (let kk = 0; kk < 2; ++kk) {
      for (let m = 0; m < M; ++m) {
        const pm = (m / 8) | 0;
        const qi = (qweight[(k + kk) * colsPacked + pm] >>> ((m % 8) * 4)) & 0xf;
        const byteIndex = m * (K / 2) + (k / 2);
        if (kk === 0) q[byteIndex] = (q[byteIndex] & 0xf0) | (qi & 0xf);
        else q[byteIndex] = (q[byteIndex] & 0x0f) | ((qi & 0xf) << 4);
      }
    }
  }
  const W = new Float32Array(M * K);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) {
      const g = (k / gs) | 0;
      const b = q[m * (K / 2) + ((k / 2) | 0)];
      const qi = k & 1 ? (b >> 4) & 0xf : b & 0xf;
      W[m * K + k] = qi * scalesMg[m * ng + g] + zeros[m * ng + g];
    }
  }
  return W;
}

function rmse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; ++i) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s / a.length);
}

function maxAbs(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; ++i) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function runCase(name, M, K, gs) {
  const ng = K / gs;
  const qTrue = new Int32Array(M * K);
  const zpTrue = new Int32Array(M * ng);
  const scalesGm = new Float32Array(ng * M);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) qTrue[m * K + k] = (m * 3 + k * 5 + 7) & 0xf;
    for (let g = 0; g < ng; ++g) {
      zpTrue[m * ng + g] = (m + g * 2 + 3) & 0xf;
      scalesGm[g * M + m] = 0.001 * (1 + ((m + g * 17) % 50));
    }
  }
  const packed = packAwq(qTrue, zpTrue, scalesGm, M, K, gs);
  const Wref = dequantOfficial(packed.qweight, packed.qzeros, scalesGm, M, K, gs);

  const cur = importAndEngineDequant(
    packed.qweight,
    packed.qzeros,
    scalesGm,
    M,
    K,
    gs,
    [ng, M],
  );
  const old = importOldBuggy(packed.qweight, packed.qzeros, scalesGm, M, K, gs);

  const rNew = rmse(cur.W, Wref);
  const rOld = rmse(old, Wref);
  const mNew = maxAbs(cur.W, Wref);
  console.log(
    `[${name}] M=${M} K=${K} gs=${gs} layout=${cur.scaleLayout} ` +
      `rmse_new=${rNew.toExponential(3)} max_new=${mNew.toExponential(3)} ` +
      `rmse_old=${rOld.toExponential(3)}`,
  );
  assert.equal(cur.scaleLayout, "g_m");
  assert.ok(rNew < 1e-6, `${name}: new import RMSE too high ${rNew}`);
  assert.ok(mNew < 1e-5, `${name}: new import maxAbs too high ${mNew}`);
  assert.ok(rOld > 1e-3, `${name}: old import should be wrong (got rmse ${rOld})`);
}

console.log("AWQ_REVERSE_ORDER vs INV", AWQ_REVERSE_ORDER.join(","), "vs", INV.join(","));
assert.deepEqual(AWQ_REVERSE_ORDER, INV);

runCase("expert-like", 512, 2048, 32);
runCase("router-like", 256, 2048, 32);
runCase("small", 16, 64, 32);

// Regression: packed qzeros are arbitrary int32 bit patterns. Storing via Float32Array
// (as importAwqToQlwc used to via tensorToF32 I32) corrupts |x|>2^24 and breaks zp.
{
  const word = 0x76543210 | 0;
  const viaF32 = new Float32Array([word])[0] | 0;
  assert.notEqual(viaF32, word, "Float32 must lose packed qzeros bits (regression guard)");
  const viaI32 = new Int32Array([word])[0] | 0;
  assert.equal(viaI32, word);

  // Realistic packing often sets high nibbles → same corruption in import path.
  const M = 64,
    K = 128,
    gs = 32,
    ng = K / gs;
  const qTrue = new Int32Array(M * K);
  const zpTrue = new Int32Array(M * ng);
  const scalesGm = new Float32Array(ng * M);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) qTrue[m * K + k] = (m * 3 + k * 5 + 7) & 0xf;
    for (let g = 0; g < ng; ++g) {
      zpTrue[m * ng + g] = (m * 7 + g * 11 + 13) & 0xf; // spreads across all nibbles
      scalesGm[g * M + m] = 0.002 * (1 + ((m + g * 19) % 40));
    }
  }
  const packed = packAwq(qTrue, zpTrue, scalesGm, M, K, gs);
  let corruptedWords = 0;
  for (let i = 0; i < packed.qzeros.length; ++i) {
    const w = packed.qzeros[i] | 0;
    if ((new Float32Array([w])[0] | 0) !== w) corruptedWords++;
  }
  console.log(`[f32-qzeros-trap] packed_words=${packed.qzeros.length} corrupted_via_f32=${corruptedWords}`);
  assert.ok(corruptedWords > 0, "test pack should hit Float32-unsafe qzeros words");

  const Wref = dequantOfficial(packed.qweight, packed.qzeros, scalesGm, M, K, gs);
  const cur = importAndEngineDequant(packed.qweight, packed.qzeros, scalesGm, M, K, gs, [ng, M]);
  assert.ok(rmse(cur.W, Wref) < 1e-6, "int32 qzeros path must match official");

  // Simulate old buggy importer: Float32Array round-trip on packed words
  const buggyZeros = new Int32Array(packed.qzeros.length);
  for (let i = 0; i < packed.qzeros.length; ++i) {
    buggyZeros[i] = new Float32Array([packed.qzeros[i] | 0])[0] | 0;
  }
  const bad = importAndEngineDequant(packed.qweight, buggyZeros, scalesGm, M, K, gs, [ng, M]);
  const rBad = rmse(bad.W, Wref);
  console.log(`[f32-qzeros-trap] rmse_after_f32_qzeros=${rBad.toExponential(3)}`);
  assert.ok(rBad > 1e-3, "Float32 qzeros path must break dequant (got " + rBad + ")");
}

// Variant: if scales were wrongly stored as [M,ng] in safetensors (some exporters)
{
  const M = 16,
    K = 64,
    gs = 32,
    ng = K / gs;
  const qTrue = new Int32Array(M * K);
  const zpTrue = new Int32Array(M * ng);
  const scalesGm = new Float32Array(ng * M);
  for (let m = 0; m < M; ++m) {
    for (let k = 0; k < K; ++k) qTrue[m * K + k] = (m + k) & 0xf;
    for (let g = 0; g < ng; ++g) {
      zpTrue[m * ng + g] = 7;
      scalesGm[g * M + m] = 0.01 * (g + 1);
    }
  }
  const packed = packAwq(qTrue, zpTrue, scalesGm, M, K, gs);
  // Pretend scales buffer is still g_m bytes but shape lies as [M,ng]
  const wrongShape = importAndEngineDequant(
    packed.qweight,
    packed.qzeros,
    scalesGm,
    M,
    K,
    gs,
    [M, ng],
  );
  const Wref = dequantOfficial(packed.qweight, packed.qzeros, scalesGm, M, K, gs);
  const r = rmse(wrongShape.W, Wref);
  console.log(`[shape-lie m_g] rmse=${r.toExponential(3)} layout=${wrongShape.scaleLayout}`);
  assert.ok(r > 1e-3, "lying shape [M,ng] on g_m data should break dequant");
}

console.log("OK: current AutoAWQ import matches official dequantize_gemm");
