#!/usr/bin/env node
/**
 * tools/glm/prepare_glm.mjs
 * GLM-5.3-Flash 一键：下载 → GLMQ 转换/导入 → 可选 AWQ；与 Qwen prepare_model 隔离。
 *
 * 用法:
 *   node tools/glm/prepare_glm.mjs --prune-hf
 *   node tools/glm/prepare_glm.mjs --quant awq --prune-hf
 *   node tools/glm/prepare_glm.mjs --quant nvfp4 --nvfp4-model LibertAIDAI/GLM-5.3-Flash-NVFP4 --force
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const MIN_WEIGHT_BYTES = 1 << 20;

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (a.includes("-h") || a.includes("--help")) {
    console.error(
      "usage: node tools/glm/prepare_glm.mjs [--quant awq|nvfp4]\n" +
        "       [--model org/name] [--nvfp4-model org/name]\n" +
        "       [--source auto|modelscope|hf-mirror|hf]\n" +
        "       [--out-hf DIR] [--out-nvfp4-hf DIR] [--out-bf16 FILE] [--out-awq FILE] [--out-nvfp4 FILE]\n" +
        "       [--prune-hf] [--remove-hf] [--keep-bf16]\n" +
        "       [--force] [--force-download] [--force-convert] [--force-quant]\n" +
        "       [--skip-download] [--skip-convert] [--skip-quant]\n" +
        "       [--limit-experts N] [--limit-layers N] [--ms-id] [--hf-id]\n" +
        "  default --quant nvfp4 → models/GLM-5.3-Flash.nvfp4.glmq (LibertAIDAI/GLM-5.3-Flash-NVFP4)\n" +
        "  --quant awq       → models/GLM-5.3-Flash.awq.glmq (base HF → convert → AWQ)",
    );
    process.exit(a.includes("-h") || a.includes("--help") ? 0 : 2);
  }
  const quant = (get("--quant") ?? "nvfp4").toLowerCase();
  if (quant !== "awq" && quant !== "nvfp4") {
    console.error(`ERROR: --quant must be awq|nvfp4, got ${quant}`);
    process.exit(2);
  }
  const model = get("--model") ?? "zai-org/GLM-5.3-Flash";
  const nvfp4Model = get("--nvfp4-model") ?? "LibertAIDAI/GLM-5.3-Flash-NVFP4";
  const short = "GLM-5.3-Flash";
  const force = a.includes("--force");
  return {
    quant,
    model,
    nvfp4Model,
    short,
    source: get("--source") ?? "auto",
    outHf: get("--out-hf") ?? path.join("models", `${short}-hf`),
    outNvfp4Hf: get("--out-nvfp4-hf") ?? path.join("models", `${short}-NVFP4`),
    outBf16: get("--out-bf16") ?? path.join("models", `${short}.bf16.glmq`),
    outAwq: get("--out-awq") ?? path.join("models", `${short}.awq.glmq`),
    outNvfp4: get("--out-nvfp4") ?? path.join("models", `${short}.nvfp4.glmq`),
    msId: get("--ms-id"),
    hfId: get("--hf-id"),
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
  if (opts.msId) args.push("--ms-id", opts.msId);
  if (opts.hfId) args.push("--hf-id", opts.hfId);
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
  console.log(`== [GLM] quant=${o.quant} model=${o.model}`);

  // --- tokenizer / base HF (always needed for configs' tokenizer_dir) ---
  const needBaseHf =
    !o.hardSkipDownload &&
    (o.forceDownload || !hasTokenizer(o.outHf) || (o.quant === "awq" && !hasHfWeights(o.outHf)));
  if (needBaseHf) {
    if (o.quant === "awq" || !hasTokenizer(o.outHf)) {
      downloadModel(o.model, o.outHf, o);
    }
  } else {
    console.log(`skip download base HF (${o.outHf} ok)`);
  }

  if (o.quant === "awq") {
    const needConvert =
      !o.hardSkipConvert && (o.forceConvert || !hasGoodGlmq(o.outBf16));
    if (needConvert) {
      if (!hasHfWeights(o.outHf)) {
        throw new Error(`missing safetensors under ${o.outHf}; download first`);
      }
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
      if (!hasGoodGlmq(o.outBf16)) {
        throw new Error(`missing ${o.outBf16}; convert first`);
      }
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

    if (o.pruneHf) {
      const n = pruneSafetensors(o.outHf);
      console.log(`prune-hf: removed ${n} *.safetensors under ${o.outHf}`);
    }
    if (o.removeHf) removeDir(o.outHf);

    console.log(`\nOK. Engine weights: ${o.outAwq}`);
    console.log(`    Tokenizer keep: ${o.outHf}/`);
    console.log("Next: start_glm.cmd   or   ./start_glm.sh");
    return;
  }

  // --- nvfp4 ---
  const needNv =
    !o.hardSkipDownload && (o.forceDownload || !hasHfWeights(o.outNvfp4Hf));
  if (needNv) {
    downloadModel(o.nvfp4Model, o.outNvfp4Hf, {
      ...o,
      msId: undefined,
      hfId: undefined,
    });
  } else {
    console.log(`skip download NVFP4 HF (${o.outNvfp4Hf} ok)`);
  }

  // Ensure tokenizer/config for engine_glm_nvfp4.yaml (tokenizer_dir = GLM-5.3-Flash-hf)
  if (!hasTokenizer(o.outHf)) {
    if (o.hardSkipDownload) {
      throw new Error(`missing tokenizer under ${o.outHf}; need base model for tokenizer_dir`);
    }
    downloadModel(o.model, o.outHf, o);
  }

  const needImport =
    !o.hardSkipConvert && (o.forceConvert || o.forceQuant || !hasGoodGlmq(o.outNvfp4));
  if (needImport) {
    if (!hasHfWeights(o.outNvfp4Hf)) {
      throw new Error(`missing safetensors under ${o.outNvfp4Hf}`);
    }
    const args = [
      path.join("tools", "glm", "import_glm_nvfp4.mjs"),
      "--src",
      o.outNvfp4Hf,
      "--out",
      o.outNvfp4,
    ];
    if (o.limitExperts) args.push("--limit-experts", o.limitExperts);
    if (o.limitLayers) args.push("--limit-layers", o.limitLayers);
    run("import-nvfp4", nodeBin(), args);
  } else {
    console.log(`skip import (${o.outNvfp4} ${formatBytes(fileSize(o.outNvfp4))})`);
  }

  if (o.pruneHf) {
    const n1 = pruneSafetensors(o.outNvfp4Hf);
    const n2 = pruneSafetensors(o.outHf);
    console.log(`prune-hf: removed ${n1}+${n2} *.safetensors`);
  }
  if (o.removeHf) {
    removeDir(o.outNvfp4Hf);
  }

  console.log(`\nOK. Engine weights: ${o.outNvfp4}`);
  console.log(`    Tokenizer keep: ${o.outHf}/`);
  console.log("Next: start_glm.cmd configs\\engine_glm_nvfp4.yaml");
}

try {
  main();
} catch (e) {
  console.error("FATAL:", e.message || e);
  process.exit(1);
}
