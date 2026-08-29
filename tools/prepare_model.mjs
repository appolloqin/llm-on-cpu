#!/usr/bin/env node
// llm-on-cpu :: tools/prepare_model.mjs
// 一键：下载 HF 权重 → 转 LWC → lwc_verify --update（可选 --config）→ 可选清理 HF 大文件。
// 底层仍是独立工具；本脚本只负责串起来并解析默认路径。
//
// 用法:
//   node tools/prepare_model.mjs --model Qwen/Qwen3.5-4B
//   node tools/prepare_model.mjs --model Qwen/Qwen3.8-27B --prune-hf   # 校验通过后删 *.safetensors
//   node tools/prepare_model.mjs --model org/name --skip-download
//   node tools/prepare_model.mjs --model org/name --out-hf DIR --out-lwc FILE
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => {
    const i = a.indexOf(k);
    return i >= 0 ? a[i + 1] : undefined;
  };
  if (!a.includes("--model")) {
    console.error(
      "usage: node tools/prepare_model.mjs --model <id> [--source auto|modelscope|hf-mirror|hf]\n" +
        "       [--out-hf DIR] [--out-lwc FILE] [--skip-download] [--skip-convert]\n" +
        "       [--no-verify] [--prune-hf] [--remove-hf] [--limit-experts N]\n" +
        "       [--ms-id] [--hf-id] [--all]\n" +
        "  --prune-hf   校验通过后删除 *-hf 内 *.safetensors(+index)，保留 config/tokenizer\n" +
        "  --remove-hf  校验通过后删除整个 *-hf 目录（更省盘，需重下才能再核对）"
    );
    process.exit(2);
  }
  const model = get("--model");
  const short = model.split("/").pop();
  return {
    model,
    source: get("--source") ?? "auto",
    outHf: get("--out-hf") ?? path.join("models", `${short}-hf`),
    outLwc: get("--out-lwc") ?? path.join("models", `${short}.lwc`),
    msId: get("--ms-id"),
    hfId: get("--hf-id"),
    all: a.includes("--all"),
    skipDownload: a.includes("--skip-download"),
    skipConvert: a.includes("--skip-convert"),
    noVerify: a.includes("--no-verify"),
    pruneHf: a.includes("--prune-hf"),
    removeHf: a.includes("--remove-hf"),
    limitExperts: get("--limit-experts"),
  };
}

function run(label, cmd, args) {
  console.log(`\n== [${label}] ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { cwd: ROOT, stdio: "inherit", shell: false });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${label} failed (exit ${r.status})`);
}

function findLwcVerify() {
  const candidates = [
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

/** 删权重分片，保留 config/tokenizer 等小文件。返回释放字节数。 */
function pruneHfWeights(hfDir) {
  const abs = path.resolve(ROOT, hfDir);
  if (!fs.existsSync(abs)) {
    console.log(`\n== [prune-hf] skip: ${hfDir} not found`);
    return 0;
  }
  let freed = 0;
  const names = fs.readdirSync(abs);
  for (const name of names) {
    const drop =
      name.endsWith(".safetensors") ||
      name === "model.safetensors.index.json" ||
      name.endsWith(".bin");
    if (!drop) continue;
    const p = path.join(abs, name);
    const st = fs.statSync(p);
    if (!st.isFile()) continue;
    fs.unlinkSync(p);
    freed += st.size;
    console.log(`  removed ${name} (${formatBytes(st.size)})`);
  }
  return freed;
}

function removeHfDir(hfDir) {
  const abs = path.resolve(ROOT, hfDir);
  if (!fs.existsSync(abs)) {
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
  walk(abs);
  fs.rmSync(abs, { recursive: true, force: true });
  return freed;
}

function main() {
  const opt = parseArgs();
  const node = process.execPath;

  if ((opt.pruneHf || opt.removeHf) && opt.noVerify) {
    console.error("[prepare] --prune-hf/--remove-hf 需要先通过校验，请去掉 --no-verify");
    process.exit(2);
  }

  if (!opt.skipDownload) {
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
  } else {
    console.log(`\n== [download] skipped (--skip-download); expect ${opt.outHf}`);
  }

  if (!opt.skipConvert) {
    const args = [
      path.join("tools", "convert_lwc.mjs"),
      "--src",
      opt.outHf,
      "--out",
      opt.outLwc,
    ];
    if (opt.limitExperts) args.push("--limit-experts", opt.limitExperts);
    run("convert", node, args);
  } else {
    console.log(`\n== [convert] skipped (--skip-convert); expect ${opt.outLwc}`);
  }

  if (!opt.noVerify) {
    const bin = findLwcVerify();
    if (!bin) {
      console.error(
        "\n[prepare] 找不到 lwc_verify。请先构建:\n" +
          "  Windows: cmd /c scripts\\configure_dev.cmd && cmd /c scripts\\build_dev.cmd\n" +
          "  Linux/macOS: ./scripts/build_linux.sh"
      );
      process.exit(1);
    }
    // 先回填校验和(不带 --config); 再单独交叉核对, 避免核对失败时看不清 update 结果
    run("verify-update", bin, [opt.outLwc, "--update"]);
    const config = path.join(opt.outHf, "config.json");
    if (fs.existsSync(config)) {
      try {
        run("verify-config", bin, [opt.outLwc, "--config", config]);
      } catch (e) {
        console.error(
          "\n[prepare] config 交叉核对失败。常见原因:\n" +
            "  · 旧 convert 缺 torch_dtype 时误标 F16(权重实为 BF16) → 重转或:\n" +
            `    ${bin} ${opt.outLwc} --set-dtype BF16\n` +
            "  · 然后再跑本命令(可加 --skip-download --skip-convert --prune-hf)"
        );
        throw e;
      }
    }
    run("verify-final", bin, [opt.outLwc]);
  }

  if (opt.removeHf) {
    console.log(`\n== [remove-hf] ${opt.outHf}`);
    const freed = removeHfDir(opt.outHf);
    console.log(`  freed ${formatBytes(freed)}`);
  } else if (opt.pruneHf) {
    console.log(`\n== [prune-hf] drop weight shards in ${opt.outHf} (keep config/tokenizer)`);
    const freed = pruneHfWeights(opt.outHf);
    console.log(`  freed ${formatBytes(freed)}`);
  }

  console.log(`\n=== prepare OK -> ${opt.outLwc} ===`);
  return 0;
}

process.exit(main());
