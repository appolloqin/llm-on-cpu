// toks_bench.mjs — measure decode tok/s of a running llmoc server
// usage: node toks_bench.mjs <port> [max_tokens] [runs] [prompt]
const port = process.argv[2] || "15085";
const maxTokens = Number(process.argv[3] || 128);
const runs = Number(process.argv[4] || 3);
const prompt = process.argv[5] || "用中文写一段关于秋天的短文,大约两百字。";

async function waitReady() {
  for (let i = 0; i < 120; ++i) {
    try { const h = await fetch(`http://127.0.0.1:${port}/healthz`); if (h.ok) return; } catch {}
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("server not ready");
}

async function oneRun() {
  const t0 = Date.now();
  const r = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      stream: true,
      temperature: 0,
    }),
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "", n = 0, ttft = -1;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      try {
        const j = JSON.parse(data);
        const tok = j.choices?.[0]?.delta?.content;
        if (tok != null && tok !== "") { if (ttft < 0) ttft = Date.now() - t0; n++; }
      } catch {}
    }
  }
  const total = Date.now() - t0;
  const decodeMs = total - (ttft > 0 ? ttft : 0);
  return { n, total, ttft, decodeMs, tps: n / (total / 1000), decodeTps: n / (decodeMs / 1000) };
}

await waitReady();
const res = [];
for (let i = 0; i < runs; ++i) {
  const r = await oneRun();
  res.push(r);
  console.log(`run${i}: tokens=${r.n} total=${(r.total / 1000).toFixed(2)}s ttft=${r.ttft}ms decode=${r.decodeMs}ms  tps=${r.tps.toFixed(2)} decode_tps=${r.decodeTps.toFixed(2)}`);
}
res.sort((a, b) => a.decodeTps - b.decodeTps);
const med = res[Math.floor(res.length / 2)];
console.log(`MEDIAN decode_tps=${med.decodeTps.toFixed(2)} (tokens=${med.n}, ttft=${med.ttft}ms)`);
