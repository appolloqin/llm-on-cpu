#!/usr/bin/env node
// llm-on-cpu :: tools/prepare_model.mjs
// 一键：下载 → 转 LWC → 校验 → 可选 prune / INT4。
// 默认自动判断跳过已完成步骤；可用 --force* 强制重跑。
//
// 用法:
//   node tools/prepare_model.mjs --model Qwen/Qwen3.5-4B --prune-hf
//   node tools/prepare_model.mjs --model Qwen/Qwen3.8-27B --prune-hf --int4
//   node tools/prepare_model.mjs --model org/name --force-convert
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MIN_WEIGHT_BYTES = 1 << 20; // 1 MiB：小于此视为坏产物 / 仅目录头

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--model")) {
    console.error(
      "usage: node tools/prepare_model.mjs --model <id> [--source auto|modelscope|hf-mirror|hf]\n" +
        "       [--out-hf DIR] [--out-lwc FILE] [--out-qlwc FILE]\n" +
        "       [--prune-hf] [--remove-hf] [--int4] [--no-verify]\n" +
        "       [--force] [--force-download] [--force-convert] [--force-int4]\n" +
        "       [--skip-download] [--skip-convert]  (legacy hard-skip)\n" +
        "       [--limit-experts N] [--ms-id] [--hf-id] [--all]\n" +
        "  Auto-skip: existing HF shards / good LWC / good QLWC are detected.\n" +
        "  --prune-hf   after OK LWC, delete *-hf *.safetensors (keep config/tokenizer)\n" +
        "  --int4       also quantize to .int4.qlwc and remove mid .lwc"
    );
    process.exit(2);
  }
  const model = get("--model");
  const short = model.split("/").pop();
  const force = a.includes("--force");
  return {
    model,
    short,
    source: get("--source") ?? "auto",
    outHf: get("--out-hf") ?? path.join("models", `${short}-hf`),
    outLwc: get("--out-lwc") ?? path.join("models", `${short}.lwc`),
    outQlwc:
      get("--out-qlwc") ?? path.join("models", `${short}.int4.qlwc`),
    msId: get("--ms-id"),
    hfId: get("--hf-id"),
    all: a.includes("--all"),
    // legacy hard skip
    hardSkipDownload: a.includes("--skip-download"),
    hardSkipConvert: a.includes("--skip-convert"),
    forceDownload: force || a.includes("--force-download"),
    forceConvert: force || a.includes("--force-convert"),
    forceInt4: force || a.includes("--force-int4"),
    noVerify: a.includes("--no-verify"),
    pruneHf: a.includes("--prune-hf"),
    removeHf: a.includes("--remove-hf"),
    int4: a.includes("--int4"),
    limitExperts: get("--limit-experts"),
  };
}

function run(label, cmd, args, opts = {}) {
  console.log(`\n== [${label}] ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, {
    cwd: ROOT,
    stdio: opts.stdio ?? "inherit",
    shell: false,
  });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${label} failed (exit ${r.status})`);
  return r;
}

function findLwcVerify() {
  const candidates = [
    path.join(ROOT, "bin", "lwc_verify.exe"),
    path.join(ROOT, "bin", "lwc_verify"),
    path.join(ROOT, "build", "msvc-x64", "bin", "lwc_verify.exe"),
    path.join(ROOT, "build", "release", "bin", "lwc_verify"),
    path.join(ROOT, "build", "release", "bin", "lwc_verify.exe"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  const build = path.join(ROOT, "build");
  if (!fs.existsSync(build)) return null;
  for (const d of fs.readdirSync(build)) {
    for (const name of ["lwc_verify.exe", "lwc_verify"]) {
      const p = path.join(build, d, "bin", name);
      if (fs.existsSync(p)) return p;
    }
  }
  return null;
}

function formatBytes(n) {
  if (n >= 2 ** 30) return (n / 2 ** 30).toFixed(2) + " GiB";
  if (n >= 2 ** 20) return (n / 2 ** 20).toFixed(1) + " MiB";
  return n + " B";
}

function abs(p) {
  return path.resolve(ROOT, p);
}

function fileSize(p) {
  const a = abs(p);
  if (!fs.existsSync(a)) return 0;
  return fs.statSync(a).size;
}

function listSafetensors(hfDir) {
  const a = abs(hfDir);
  if (!fs.existsSync(a)) return [];
  return fs.readdirSync(a).filter((f) => f.endsWith(".safetensors"));
}

function hfHasConfig(hfDir) {
  return fs.existsSync(path.join(abs(hfDir), "config.json"));
}

function hfHasTokenizer(hfDir) {
  const a = abs(hfDir);
  return (
    fs.existsSync(path.join(a, "tokenizer.json")) ||
    fs.existsSync(path.join(a, "tokenizer_config.json"))
  );
}

/** 体积达标的权重文件（LWC/QLWC） */
function weightLooksOk(p) {
  return fileSize(p) >= MIN_WEIGHT_BYTES;
}

/** lwc_verify 快速终检通过则认为可用 */
function lwcVerifyOk(bin, lwc) {
  if (!bin || !weightLooksOk(lwc)) return false;
  const r = spawnSync(bin, [lwc], {
    cwd: ROOT,
    stdio: "pipe",
    shell: false,
    encoding: "utf8",
  });
  return r.status === 0;
}

function pruneHfWeights(hfDir) {
  const a = abs(hfDir);
  if (!fs.existsSync(a)) {
    console.log(`\n== [prune-hf] skip: ${hfDir} not found`);
    return 0;
  }
  let freed = 0;
  for (const name of fs.readdirSync(a)) {
    const drop =
      name.endsWith(".safetensors") ||
      name === "model.safetensors.index.json" ||
      name.endsWith(".bin");
    if (!drop) continue;
    const p = path.join(a, name);
    const st = fs.statSync(p);
    if (!st.isFile()) continue;
    fs.unlinkSync(p);
    freed += st.size;
    console.log(`  removed ${name} (${formatBytes(st.size)})`);
  }
  return freed;
}

function removeHfDir(hfDir) {
  const a = abs(hfDir);
  if (!fs.existsSync(a)) {
    console.log(`\n== [remove-hf] skip: ${hfDir} not found`);
    return 0;
  }
  let freed = 0;
  const walk = (dir) => {
    for (const name of fs.readdirSync(dir)) {
      const p = path.join(dir, name);
      const st = fs.statSync(p);
      if (st.isDirectory()) walk(p);
      else freed += st.size;
    }
  };
  walk(a);
  fs.rmSync(a, { recursive: true, force: true });
  return freed;
}

function main() {
  const opt = parseArgs();
  const node = process.execPath;
  const verifyBin = findLwcVerify();

  if ((opt.pruneHf || opt.removeHf) && opt.noVerify) {
    console.error("[prepare] --prune-hf/--remove-hf need verify; drop --no-verify");
    process.exit(2);
  }

  const shards = listSafetensors(opt.outHf);
  const hasShards = shards.length > 0;
  const hasConfig = hfHasConfig(opt.outHf);
  let lwcOk =
    weightLooksOk(opt.outLwc) &&
    (opt.noVerify || !verifyBin || lwcVerifyOk(verifyBin, opt.outLwc));
  // 无 verify 时仅按体积判断，避免误杀
  if (!verifyBin && weightLooksOk(opt.outLwc)) lwcOk = true;

  const qlwcOk = weightLooksOk(opt.outQlwc);

  console.log("\n== [plan] auto-detect");
  console.log(`  hf dir:     ${opt.outHf}  config=${hasConfig} shards=${shards.length}`);
  console.log(
    `  lwc:        ${opt.outLwc}  ${lwcOk ? "OK" : weightLooksOk(opt.outLwc) ? "BAD/verify-fail" : "missing"} (${formatBytes(fileSize(opt.outLwc))})`
  );
  if (opt.int4) {
    console.log(
      `  qlwc:       ${opt.outQlwc}  ${qlwcOk ? "OK" : "missing/small"} (${formatBytes(fileSize(opt.outQlwc))})`
    );
  }

  // 最终目标是否已就绪
  const goalOk = opt.int4 ? qlwcOk : lwcOk;
  // 转 LWC / 量化还缺不缺源
  const needLwc = !lwcOk && !(opt.int4 && qlwcOk);
  const needShards = needLwc && !hasShards;

  // ---- download ----
  // Never treat "some *.safetensors exist" as complete — interrupted runs leave
  // partial shards. Always invoke download_model (per-file size resume) unless
  // hard-skip or the final engine weights are already OK.
  let doDownload = false;
  if (opt.hardSkipDownload) {
    console.log("\n== [download] skipped (--skip-download)");
  } else if (opt.forceDownload) {
    doDownload = true;
  } else if (goalOk && !opt.forceConvert) {
    console.log("\n== [download] skipped (target weights already OK)");
  } else if (needLwc || needShards || hasShards) {
    // hasShards but incomplete → still run; download_model resumes / re-gets bad sizes
    doDownload = true;
    if (hasShards) {
      console.log(
        `\n== [download] resume/verify (${shards.length} local safetensors — will check sizes)`,
      );
    }
  } else {
    console.log("\n== [download] skipped (not needed for remaining steps)");
  }

  if (doDownload) {
    const args = [
      path.join("tools", "download_model.mjs"),
      "--model",
      opt.model,
      "--source",
      opt.source,
      "--out",
      opt.outHf,
    ];
    if (opt.msId) args.push("--ms-id", opt.msId);
    if (opt.hfId) args.push("--hf-id", opt.hfId);
    if (opt.all) args.push("--all");
    run("download", node, args);
  }

  const shardsAfter = listSafetensors(opt.outHf);
  const hasShardsAfter = shardsAfter.length > 0;

  // ---- convert ----
  let didConvert = false;
  let doConvert = false;
  if (opt.hardSkipConvert) {
    console.log("\n== [convert] skipped (--skip-convert)");
  } else if (opt.forceConvert) {
    doConvert = true;
  } else if (lwcOk) {
    console.log("\n== [convert] skipped (LWC already OK)");
  } else if (opt.int4 && qlwcOk) {
    console.log("\n== [convert] skipped (QLWC already OK; LWC not needed)");
  } else if (!hasShardsAfter) {
    throw new Error(
      `need *.safetensors under ${opt.outHf} to convert, but none found.\n` +
        `  Re-run with --force-download, or place HF weights there.`
    );
  } else {
    doConvert = true;
  }

  if (doConvert) {
    const args = [
      path.join("tools", "convert_lwc.mjs"),
      "--src",
      opt.outHf,
      "--out",
      opt.outLwc,
    ];
    if (opt.limitExperts) args.push("--limit-experts", opt.limitExperts);
    run("convert", node, args);
    if (!weightLooksOk(opt.outLwc)) {
      throw new Error(
        `convert produced tiny/missing LWC (${formatBytes(fileSize(opt.outLwc))}): ${opt.outLwc}`
      );
    }
    didConvert = true;
    lwcOk = true;
  }

  // ---- verify ----
  const needLwcForInt4 = opt.int4 && !qlwcOk;
  const needVerify =
    !opt.noVerify &&
    weightLooksOk(opt.outLwc) &&
    (didConvert || needLwcForInt4 || opt.pruneHf || opt.removeHf || !opt.int4);

  if (needVerify) {
    if (!verifyBin) {
      console.error(
        "\n[prepare] lwc_verify not found. Build first:\n" +
          "  Windows: cmd /c scripts\\configure_dev.cmd && cmd /c scripts\\build_dev.cmd\n" +
          "  Linux/macOS: ./scripts/build_linux.sh"
      );
      process.exit(1);
    }
    if (didConvert || opt.forceConvert) {
      run("verify-update", verifyBin, [opt.outLwc, "--update"]);
      const config = path.join(opt.outHf, "config.json");
      if (fs.existsSync(config)) {
        try {
          run("verify-config", verifyBin, [opt.outLwc, "--config", config]);
        } catch (e) {
          console.error(
            "\n[prepare] config cross-check failed. Try:\n" +
              `  ${verifyBin} ${opt.outLwc} --set-dtype BF16\n` +
              "  then re-run (auto-skip will resume)."
          );
          throw e;
        }
      }
    }
    run("verify-final", verifyBin, [opt.outLwc]);
    lwcOk = true;
  } else if (opt.int4 && qlwcOk) {
    console.log("\n== [verify] skipped (QLWC already OK)");
  } else if (opt.noVerify) {
    console.log("\n== [verify] skipped (--no-verify)");
  } else if (!weightLooksOk(opt.outLwc) && !(opt.int4 && qlwcOk)) {
    throw new Error(`no usable LWC at ${opt.outLwc}`);
  } else {
    console.log("\n== [verify] skipped (LWC already verified / unchanged)");
  }

  // ---- prune / remove HF ----
  if (opt.removeHf) {
    if (!hfHasConfig(opt.outHf) && !hfHasTokenizer(opt.outHf) && !hasShardsAfter) {
      console.log("\n== [remove-hf] skipped (already gone)");
    } else {
      console.log(`\n== [remove-hf] ${opt.outHf}`);
      console.log(`  freed ${formatBytes(removeHfDir(opt.outHf))}`);
    }
  } else if (opt.pruneHf) {
    const left = listSafetensors(opt.outHf);
    if (left.length === 0) {
      console.log("\n== [prune-hf] skipped (no weight shards left)");
    } else if (!(lwcOk || (opt.int4 && qlwcOk))) {
      throw new Error("[prune-hf] refused: no OK engine weights yet");
    } else {
      console.log(
        `\n== [prune-hf] drop weight shards in ${opt.outHf} (keep config/tokenizer)`
      );
      console.log(`  freed ${formatBytes(pruneHfWeights(opt.outHf))}`);
    }
  }

  // ---- INT4 ----
  if (opt.int4) {
    if (qlwcOk && !opt.forceInt4) {
      console.log(`\n== [int4] skipped (already OK: ${opt.outQlwc})`);
    } else {
      if (!weightLooksOk(opt.outLwc)) {
        throw new Error(`INT4 needs LWC first: missing/small ${opt.outLwc}`);
      }
      run("int4-quantize", node, [
        path.join("tools", "quantize_int4.mjs"),
        "--src",
        opt.outLwc,
        "--out",
        opt.outQlwc,
        "--method",
        "gptq",
      ]);
      if (!weightLooksOk(opt.outQlwc)) {
        throw new Error(`quantize produced tiny/missing QLWC: ${opt.outQlwc}`);
      }
    }
    if (fs.existsSync(abs(opt.outLwc))) {
      const sz = fileSize(opt.outLwc);
      fs.unlinkSync(abs(opt.outLwc));
      console.log(
        `\n== [int4] removed intermediate LWC ${opt.outLwc} (${formatBytes(sz)})`
      );
    } else {
      console.log("\n== [int4] intermediate LWC already absent");
    }
    console.log(`\n=== prepare OK -> ${opt.outQlwc} ===`);
  } else {
    console.log(`\n=== prepare OK -> ${opt.outLwc} ===`);
  }
  return 0;
}

try {
  process.exit(main());
} catch (e) {
  console.error("\nERROR:", e.message || e);
  process.exit(1);
}
