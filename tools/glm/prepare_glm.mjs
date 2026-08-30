#!/usr/bin/env node
/**
 * tools/glm/prepare_glm.mjs
 * GLM 一键：下载 → GLMQ 导入/量化。默认只认一个模型 ID，不做 zai-org 夹带。
 *
 * 用法:
 *   node tools/glm/prepare_glm.mjs --prune-hf
 *   node tools/glm/prepare_glm.mjs --model LibertAIDAI/GLM-5.3-Flash-NVFP4
 *   node tools/glm/prepare_glm.mjs --quant awq --model ZhipuAI/GLM-5.3-Flash --prune-hf
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const MIN_WEIGHT_BYTES = 1 << 20;
const DEFAULT_NVFP4 = "LibertAIDAI/GLM-5.3-Flash-NVFP4";
const DEFAULT_AWQ = "ZhipuAI/GLM-5.3-Flash"; // ModelScope 原生 ID（HF 上为 zai-org/...）

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (a.includes("-h") || a.includes("--help")) {
    console.error(
      "usage: node tools/glm/prepare_glm.mjs [--quant nvfp4|awq] [--model org/name]\n" +
        "       [--source auto|modelscope|hf-mirror|hf]\n" +
        "       [--out-hf DIR] [--out-glmq FILE] [--prune-hf] [--remove-hf] [--keep-bf16]\n" +
        "       [--force] [--force-download] [--force-convert] [--force-quant]\n" +
        "       [--skip-download] [--skip-convert] [--skip-quant]\n" +
        "       [--limit-experts N] [--limit-layers N]\n" +
        `  default nvfp4: --model ${DEFAULT_NVFP4}\n` +
        `  default awq:   --model ${DEFAULT_AWQ}\n` +
        "  One model ID is used for ModelScope / hf-mirror / hf (no silent ID rewrite).",
    );
    process.exit(a.includes("-h") || a.includes("--help") ? 0 : 2);
  }
  const quant = (get("--quant") ?? "nvfp4").toLowerCase();
  if (quant !== "awq" && quant !== "nvfp4") {
    console.error(`ERROR: --quant must be awq|nvfp4, got ${quant}`);
    process.exit(2);
  }
  // --nvfp4-model kept as alias of --model for old scripts
  const model =
    get("--model") ?? get("--nvfp4-model") ?? (quant === "awq" ? DEFAULT_AWQ : DEFAULT_NVFP4);
  const short = model.split("/").pop() || "GLM";
  const force = a.includes("--force");
  return {
    quant,
    model,
    short,
    source: get("--source") ?? "auto",
    outHf:
      get("--out-hf") ??
      path.join("models", quant === "nvfp4" ? `${short}` : `${short}-hf`),
    outBf16: get("--out-bf16") ?? path.join("models", "GLM-5.3-Flash.bf16.glmq"),
    outAwq: get("--out-awq") ?? get("--out-glmq") ?? path.join("models", "GLM-5.3-Flash.awq.glmq"),
    outNvfp4:
      get("--out-nvfp4") ?? get("--out-glmq") ?? path.join("models", "GLM-5.3-Flash.nvfp4.glmq"),
    hardSkipDownload: a.includes("--skip-download"),
    hardSkipConvert: a.includes("--skip-convert"),
    hardSkipQuant: a.includes("--skip-quant"),
    forceDownload: force || a.includes("--force-download"),
    forceConvert: force || a.includes("--force-convert"),
    forceQuant: force || a.includes("--force-quant"),
    pruneHf: a.includes("--prune-hf"),
    removeHf: a.includes("--remove-hf"),
    keepBf16: a.includes("--keep-bf16"),
    limitExperts: get("--limit-experts"),
    limitLayers: get("--limit-layers"),
  };
}

function run(label, cmd, args) {
  console.log(`\n== [${label}] ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { cwd: ROOT, stdio: "inherit", shell: false });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${label} failed (exit ${r.status})`);
}

function abs(p) {
  return path.resolve(ROOT, p);
}

function fileSize(p) {
  const a = abs(p);
  if (!fs.existsSync(a)) return 0;
  return fs.statSync(a).size;
}

function formatBytes(n) {
  if (n >= 2 ** 30) return (n / 2 ** 30).toFixed(2) + " GiB";
  if (n >= 2 ** 20) return (n / 2 ** 20).toFixed(1) + " MiB";
  return n + " B";
}

function listSafetensors(dir) {
  const a = abs(dir);
  if (!fs.existsSync(a)) return [];
  return fs.readdirSync(a).filter((f) => f.endsWith(".safetensors"));
}

function hasGoodGlmq(p) {
  return fileSize(p) >= MIN_WEIGHT_BYTES;
}

function hasHfWeights(dir) {
  return listSafetensors(dir).length > 0;
}

function hasTokenizer(dir) {
  const a = abs(dir);
  return fs.existsSync(path.join(a, "tokenizer.json")) || fs.existsSync(path.join(a, "config.json"));
}

function nodeBin() {
  return process.execPath;
}

/** Download exactly --model (same ID for every source). No ms/hf rewrite. */
function downloadModel(modelId, outDir, opts) {
  const args = [
    path.join("tools", "download_model.mjs"),
    "--model",
    modelId,
    "--out",
    outDir,
    "--source",
    opts.source,
  ];
  run("download", nodeBin(), args);
}

function pruneSafetensors(dir) {
  const a = abs(dir);
  if (!fs.existsSync(a)) return 0;
  let n = 0;
  for (const f of fs.readdirSync(a)) {
    if (!f.endsWith(".safetensors")) continue;
    fs.unlinkSync(path.join(a, f));
    n++;
  }
  return n;
}

function removeDir(dir) {
  const a = abs(dir);
  if (fs.existsSync(a)) fs.rmSync(a, { recursive: true, force: true });
}

function main() {
  const o = parseArgs();
  console.log(`== [GLM] quant=${o.quant} model=${o.model} source=${o.source}`);

  if (o.quant === "awq") {
    // Always run download unless --skip-download. Partial shards must resume
    // (download_model.mjs checks remote sizes); "any .safetensors" is NOT complete.
    if (!o.hardSkipDownload) downloadModel(o.model, o.outHf, o);
    else console.log(`skip download (--skip-download)`);

    if (!hasHfWeights(o.outHf) || !hasTokenizer(o.outHf)) {
      throw new Error(
        `download incomplete under ${o.outHf} (need safetensors + tokenizer/config)`,
      );
    }

    const needConvert =
      !o.hardSkipConvert && (o.forceConvert || !hasGoodGlmq(o.outBf16));
    if (needConvert) {
      if (!hasHfWeights(o.outHf)) throw new Error(`missing safetensors under ${o.outHf}`);
      const args = [
        path.join("tools", "glm", "convert_glm_lwc.mjs"),
        "--src",
        o.outHf,
        "--out",
        o.outBf16,
      ];
      if (o.limitExperts) args.push("--limit-experts", o.limitExperts);
      if (o.limitLayers) args.push("--limit-layers", o.limitLayers);
      run("convert-glmq", nodeBin(), args);
    } else {
      console.log(`skip convert (${o.outBf16} ${formatBytes(fileSize(o.outBf16))})`);
    }

    const needAwq = !o.hardSkipQuant && (o.forceQuant || !hasGoodGlmq(o.outAwq));
    if (needAwq) {
      if (!hasGoodGlmq(o.outBf16)) throw new Error(`missing ${o.outBf16}`);
      run("quantize-awq", nodeBin(), [
        path.join("tools", "glm", "quantize_glm_awq.mjs"),
        "--src",
        o.outBf16,
        "--out",
        o.outAwq,
      ]);
      if (!o.keepBf16 && hasGoodGlmq(o.outAwq) && hasGoodGlmq(o.outBf16)) {
        fs.unlinkSync(abs(o.outBf16));
        console.log(`removed mid ${o.outBf16}`);
      }
    } else {
      console.log(`skip awq (${o.outAwq} ${formatBytes(fileSize(o.outAwq))})`);
    }

    if (o.pruneHf) console.log(`prune-hf: removed ${pruneSafetensors(o.outHf)} *.safetensors`);
    if (o.removeHf) removeDir(o.outHf);

    console.log(`\nOK. Engine weights: ${o.outAwq}`);
    console.log(`    Tokenizer: ${o.outHf}/`);
    console.log("Next: start_glm.cmd configs\\engine_glm_int4.yaml");
    return;
  }

  // --- nvfp4: single model ID only ---
  if (!o.hardSkipDownload) downloadModel(o.model, o.outHf, o);
  else console.log(`skip download (--skip-download)`);

  if (!hasTokenizer(o.outHf) || !hasHfWeights(o.outHf)) {
    throw new Error(
      `download incomplete under ${o.outHf} after ${o.model} (need safetensors + tokenizer/config)`,
    );
  }

  const needImport =
    !o.hardSkipConvert && (o.forceConvert || o.forceQuant || !hasGoodGlmq(o.outNvfp4));
  if (needImport) {
    if (!hasHfWeights(o.outHf)) throw new Error(`missing safetensors under ${o.outHf}`);
    const args = [
      path.join("tools", "glm", "import_glm_nvfp4.mjs"),
      "--src",
      o.outHf,
      "--out",
      o.outNvfp4,
    ];
    if (o.limitExperts) args.push("--limit-experts", o.limitExperts);
    if (o.limitLayers) args.push("--limit-layers", o.limitLayers);
    run("import-nvfp4", nodeBin(), args);
  } else {
    console.log(`skip import (${o.outNvfp4} ${formatBytes(fileSize(o.outNvfp4))})`);
  }

  if (o.pruneHf) console.log(`prune-hf: removed ${pruneSafetensors(o.outHf)} *.safetensors`);
  if (o.removeHf) removeDir(o.outHf);

  console.log(`\nOK. Engine weights: ${o.outNvfp4}`);
  console.log(`    HF checkout:   ${o.outHf}/  (${o.model})`);
  console.log(`    Tokenizer:     ${o.outHf}/`);
  console.log("Next: start_glm.cmd");
}

try {
  main();
} catch (e) {
  console.error("FATAL:", e.message || e);
  process.exit(1);
}
