#!/usr/bin/env node
// llm-on-cpu :: tools/download_model.mjs
// 模型权重多源下载 —— download_model.py 的 Node 对等实现(零 npm 依赖, Node≥18)。
// 三源可切换: modelscope(国内默认) / hf-mirror / hf; auto 按 此顺序降级。
// 全部走公开 HTTP REST(fetch 内置), 断点续传 = 同名同尺寸跳过。
//
// 用法:
//   node tools/download_model.mjs --model Qwen/Qwen3.8-27B
//   node tools/download_model.mjs --model org/name --source hf-mirror
//   node tools/download_model.mjs --model org/name --ms-id a/b --hf-id c/d --all
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

const HF = "https://huggingface.co";
const HF_MIRROR = "https://hf-mirror.com";
const MS = "https://modelscope.cn";
const KEEP = /(\.safetensors$|\.json$|^tokenizer|\.model$|\.jinja$)/; // 与 python 版一致, 跳过 *.bin
const KEEP_TOKENIZER_ONLY = /(\.json$|^tokenizer|\.model$|\.jinja$|^vocab|^merges|^special_tokens)/;

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (k) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : undefined; };
  if (!a.includes("--model")) {
    console.error(
      "usage: node download_model.mjs --model <id> [--source auto|modelscope|hf-mirror|hf]\n" +
        "       [--out DIR] [--ms-id] [--hf-id] [--all] [--tokenizer-only] [--revision R]",
    );
    process.exit(2);
  }
  const model = get("--model");
  return {
    model,
    source: get("--source") ?? "auto",
    out: get("--out") ?? path.join("models", model.split("/").pop() + "-hf"),
    msId: get("--ms-id") ?? model,
    hfId: get("--hf-id") ?? model,
    all: a.includes("--all"),
    tokenizerOnly: a.includes("--tokenizer-only"),
    revision: get("--revision"),
  };
}

async function jfetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  return r.json();
}

function formatBytes(n) {
  if (n >= 2 ** 30) return (n / 2 ** 30).toFixed(2) + " GiB";
  if (n >= 2 ** 20) return (n / 2 ** 20).toFixed(1) + " MiB";
  if (n >= 2 ** 10) return (n / 2 ** 10).toFixed(1) + " KiB";
  return n + " B";
}

function renderProgress(prefix, done, total, width = 28) {
  const hasTotal = total != null && total > 0;
  const pct = hasTotal ? Math.min(100, (done / total) * 100) : null;
  const filled = hasTotal ? Math.round((done / total) * width) : 0;
  const bar = "█".repeat(filled) + "░".repeat(width - filled);
  const sizeStr = hasTotal
    ? `${formatBytes(done)}/${formatBytes(total)} (${pct.toFixed(1)}%)`
    : formatBytes(done);
  return `${prefix} [${bar}] ${sizeStr}`;
}

function createOverallProgress(totalBytes) {
  const tty = process.stdout.isTTY;
  let lastLog = 0;
  const draw = (overall, fileLine, doneBytes) => {
    if (tty) {
      process.stdout.write(`\r\x1b[K${overall}\n\x1b[K${fileLine}\x1b[1A\r`);
      return;
    }
    const now = Date.now();
    const complete = totalBytes > 0 && doneBytes >= totalBytes;
    if (now - lastLog > 3000 || complete) {
      lastLog = now;
      console.log(overall);
      console.log(fileLine);
    }
  };
  const clear = () => {
    if (tty) process.stdout.write("\r\x1b[K\n\x1b[K\x1b[1A\r");
  };
  return { draw, clear };
}

async function downloadTo(url, dstPath, expectSize, onProgress) {
  if (fs.existsSync(dstPath)) {
    const sz = fs.statSync(dstPath).size;
    // Only skip when size is known AND matches. Partial / truncated files re-download.
    if (expectSize != null && expectSize > 0 && sz === expectSize) return "skip";
    if (expectSize != null && expectSize > 0 && sz !== expectSize) {
      console.log(
        `  incomplete ${path.basename(dstPath)}: ${formatBytes(sz)}/${formatBytes(expectSize)} — re-download`,
      );
      fs.unlinkSync(dstPath);
    } else if ((expectSize == null || expectSize <= 0) && sz > 0) {
      // No authoritative size: keep existing non-empty (small tokenizer/json).
      return "skip";
    }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  const total = expectSize ?? (Number(r.headers.get("content-length")) || null);
  await fsp.mkdir(path.dirname(dstPath), { recursive: true });
  const tmpPath = dstPath + ".partial";
  let downloaded = 0;
  const counter = new Transform({
    transform(chunk, _enc, cb) {
      downloaded += chunk.length;
      onProgress?.(downloaded, total);
      cb(null, chunk);
    },
  });
  try {
    await pipeline(Readable.fromWeb(r.body), counter, fs.createWriteStream(tmpPath));
    if (expectSize != null && expectSize > 0 && downloaded !== expectSize) {
      throw new Error(
        `size mismatch after download ${path.basename(dstPath)}: got ${downloaded}, want ${expectSize}`,
      );
    }
    fs.renameSync(tmpPath, dstPath);
  } catch (e) {
    try {
      if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    } catch {
      /* ignore */
    }
    throw e;
  }
  return "downloaded";
}

// ---- 源实现: 每个源返回 {list(): [{path,size}], rawUrl(path)} ----
const sources = {
  async modelscope(id, rev, opt) {
    const base = `${MS}/api/v1/models/${id}/repo`;
    rev = rev ?? "master";
    return {
      async list() {
        const j = await jfetch(`${base}/files?Revision=${rev}&Recursive=true`);
        return (j?.Data?.Files ?? [])
          .filter((f) => f.Type === "blob")
          .map((f) => ({ path: f.Path, size: f.Size }));
      },
      rawUrl: (p) => `${base}?FilePath=${encodeURIComponent(p)}&Revision=${rev}`,
    };
  },
  makeHf(host) {
    return async (id, rev, opt) => {
      rev = rev ?? "main";
      return {
        async list() {
          const j = await jfetch(`${host}/api/models/${id}/tree/${rev}?recursive=true`);
          return j.filter((f) => f.type === "file").map((f) => ({ path: f.path, size: f.size }));
        },
        rawUrl: (p) => `${host}/${id}/resolve/${rev}/${p}`,
      };
    };
  },
};
sources["hf-mirror"] = sources.makeHf(HF_MIRROR);
sources.hf = sources.makeHf(HF);

async function runSource(name, opt) {
  const id = name === "modelscope" ? opt.msId : opt.hfId;
  const api = await sources[name](id, opt.revision, opt);
  const files = await api.list();
  const keepRe = opt.tokenizerOnly ? KEEP_TOKENIZER_ONLY : KEEP;
  const wanted = files.filter((f) => {
    if (opt.all && !opt.tokenizerOnly) return true;
    return keepRe.test(f.path.split("/").pop() || f.path);
  });
  if (!opt.tokenizerOnly && !wanted.some((f) => f.path.endsWith(".safetensors")))
    throw new Error("file list has no *.safetensors (wrong repo id?)");
  if (opt.tokenizerOnly && wanted.length === 0)
    throw new Error("tokenizer-only: no tokenizer/config files in repo");
  const totalBytes = wanted.reduce((s, f) => s + (f.size || 0), 0);

  // Drop leftover *.partial from interrupted runs (any depth)
  const rmPartials = (dir) => {
    if (!fs.existsSync(dir)) return;
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) rmPartials(p);
      else if (ent.name.endsWith(".partial")) {
        fs.unlinkSync(p);
        console.log(`  removed stale ${path.relative(opt.out, p)}`);
      }
    }
  };
  try {
    rmPartials(opt.out);
  } catch {
    /* ignore */
  }

  // Resume check: count missing / size-mismatched files (do not treat "any shard" as done).
  let missing = 0;
  let badSize = 0;
  for (const f of wanted) {
    const dst = path.join(opt.out, ...f.path.split("/"));
    if (!fs.existsSync(dst)) {
      missing++;
      continue;
    }
    const sz = fs.statSync(dst).size;
    if (f.size != null && f.size > 0 && sz !== f.size) badSize++;
  }
  const alreadyOk = missing === 0 && badSize === 0;
  console.log(
    `  plan: ${wanted.length} files, ${formatBytes(totalBytes)}` +
      (alreadyOk
        ? " (local complete — will skip each file)"
        : ` (local: missing=${missing}, size-mismatch=${badSize})`),
  );
  if (opt.tokenizerOnly) {
    console.log(`  [tokenizer-only] ${wanted.length} files, ${formatBytes(totalBytes)}`);
  }

  const progress = createOverallProgress(totalBytes);
  let done = 0;
  let doneBytes = 0;
  let downloadedN = 0;
  for (const f of wanted) {
    const dst = path.join(opt.out, ...f.path.split("/"));
    const fileIdx = done + 1;
    const report = (d, t) => {
      const overall = renderProgress(
        `  [${fileIdx}/${wanted.length}] total`, doneBytes + d, totalBytes, 20);
      const fileLine = renderProgress(`       ${f.path}`, d, t, 20);
      progress.draw(overall, fileLine, doneBytes + d);
    };
    report(0, f.size || null);
    const how = await downloadTo(api.rawUrl(f.path), dst, f.size, report);
    if (how === "downloaded") downloadedN++;
    done++;
    doneBytes += f.size || 0;
    progress.clear();
    console.log(`  [${done}/${wanted.length}] ${how}  ${f.path} (${formatBytes(f.size || 0)})`);
  }

  // Final integrity gate: every wanted file must exist with matching size.
  const bad = [];
  for (const f of wanted) {
    const dst = path.join(opt.out, ...f.path.split("/"));
    if (!fs.existsSync(dst)) {
      bad.push(`${f.path}: missing`);
      continue;
    }
    const sz = fs.statSync(dst).size;
    if (f.size != null && f.size > 0 && sz !== f.size) {
      bad.push(`${f.path}: ${sz} != ${f.size}`);
    }
  }
  if (bad.length) {
    throw new Error(
      `download incomplete (${bad.length} files):\n  - ${bad.slice(0, 8).join("\n  - ")}` +
        (bad.length > 8 ? `\n  - ... +${bad.length - 8} more` : ""),
    );
  }

  const marker = path.join(opt.out, ".llmoc_download_ok");
  await fsp.writeFile(
    marker,
    JSON.stringify(
      {
        model: id,
        source: name,
        count: wanted.length,
        bytes: totalBytes,
        downloaded_this_run: downloadedN,
        files: wanted.map((f) => ({ path: f.path, size: f.size })),
      },
      null,
      2,
    ),
  );
  return { count: done, bytes: totalBytes, downloadedN };
}

async function main() {
  const opt = parseArgs();
  const order = opt.source === "auto"
    ? ["modelscope", "hf-mirror", "hf"]
    : [opt.source];
  const errors = [];
  for (const name of order) {
    console.log(`\n=== [${name}] downloading ${name === "modelscope" ? opt.msId : opt.hfId} -> ${opt.out} ===`);
    try {
      if (name === "modelscope" && opt.all)
        console.log("  [warn] modelscope 源使用全量文件列表, --all 无过滤语义差异");
      const { count, bytes } = await runSource(name, opt);
      console.log(`=== [${name}] OK (${count} files, ${(bytes / 2 ** 30).toFixed(2)} GiB) -> ${opt.out}`);
      console.log(`[next] node tools/convert_lwc.mjs --src ${opt.out} --out ${opt.out.replace(/-hf$/, "")}.lwc`);
      return 0;
    } catch (e) {
      console.log(`=== [${name}] FAILED: ${e.message}`);
      errors.push(`${name}: ${e.message}`);
    }
  }
  console.log("\nALL SOURCES FAILED:");
  for (const e of errors) console.log("  - " + e);
  return 1;
}

process.exit(await main());
