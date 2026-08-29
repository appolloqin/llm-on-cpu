# llm-on-cpu :: 一键自检链(Windows 开发机)
# 覆盖: 构建 → 15 项单测 → 假HF生成 → 转换 → 校验/回填/config核对 → 真实LWC进预取基准
$ErrorActionPreference = 'Stop'
$BIN = ".\build\msvc-x64\bin"

Write-Host "== [1/6] build ==" -ForegroundColor Cyan
cmd /c "scripts\build_dev.cmd" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "== [2/6] unit tests ==" -ForegroundColor Cyan
& "$BIN\llmoc_unit_tests.exe" | Select-String -Pattern "passed"
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }

Write-Host "== [3/6] fake HF -> convert ==" -ForegroundColor Cyan
python scripts\make_fake_hf.py
python tools\convert_lwc.py --src models\_selftest-hf --out models\_selftest.lwc --verify
if ($LASTEXITCODE -ne 0) { throw "convert failed" }

Write-Host "== [4/6] lwc_verify update + config cross-check ==" -ForegroundColor Cyan
& "$BIN\lwc_verify.exe" models\_selftest.lwc --update
& "$BIN\lwc_verify.exe" models\_selftest.lwc --config models\_selftest-hf\config.json
if ($LASTEXITCODE -ne 0) { throw "config cross-check failed" }

Write-Host "== [5/6] pipeline bench on real LWC ==" -ForegroundColor Cyan
& "$BIN\m2_pipeline_bench.exe" --file models\_selftest.lwc --topk 2 --compute-us 200 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) { throw "pipeline bench failed" }

Write-Host "== [6/6] isa/bandwidth smoke ==" -ForegroundColor Cyan
& "$BIN\m0_isa.exe" | Select-Object -Last 3

Write-Host "`nALL SELFTEST PASSED" -ForegroundColor Green
