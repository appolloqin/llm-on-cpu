#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_model.py — 模型权重多源下载（可换环境）
源: modelscope(国内默认) / hf-mirror(HF国内镜像) / hf(HuggingFace官方)

用法:
  python tools/download_model.py --model Qwen/Qwen3.8-27B
        # auto: modelscope → hf-mirror → hf 依次尝试, 首个成功即停

  python tools/download_model.py --model Qwen/Qwen3.8-27B --source hf
  python tools/download_model.py --model Qwen/Qwen3.8-27B --out models/qwen3827-hf
  python tools/download_model.py --model org/name --ms-id org/name_ms --hf-id org/name
  python tools/download_model.py --model org/name --all          # 不做后缀过滤(默认跳过 *.bin 避免双份体积)

依赖: 按 source 二选一
  modelscope → pip install modelscope
  hf / hf-mirror → pip install huggingface_hub
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
# 默认只拉引擎需要的文件: 权重(safetensors) + 配置 + 分词器。
# 旧格式 *.bin 与 safetensors 常并存, 不加过滤会双倍占盘 —— --all 可关掉过滤。
DEFAULT_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "*.model", "*.jinja"]


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def dl_modelscope(model_id: str, out: Path, patterns, revision: str) -> Path:
    if not have("modelscope"):
        raise RuntimeError("modelscope 未安装 → pip install modelscope")
    from modelscope import snapshot_download

    kwargs = dict(model_id=model_id, local_dir=str(out))
    if revision:
        kwargs["revision"] = revision
    try:
        p = snapshot_download(allow_patterns=patterns, **kwargs)
    except TypeError:  # 旧版 modelscope 不支持 allow_patterns
        p = snapshot_download(**kwargs)
    return Path(p)


def dl_hf(model_id: str, out: Path, patterns, revision: str, mirror: bool) -> Path:
    if not have("huggingface_hub"):
        raise RuntimeError("huggingface_hub 未安装 → pip install huggingface_hub")
    if mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            local_dir=str(out),
            allow_patterns=None if patterns is None else list(patterns),
            revision=revision or "main",
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="模型权重多源下载(auto 默认国内优先)")
    ap.add_argument("--model", required=True, help="仓库 ID(两端通用时只填一次)")
    ap.add_argument("--source", choices=("auto", "modelscope", "hf-mirror", "hf"),
                    default="auto")
    ap.add_argument("--out", default=None, help="目标目录(默认 models/<name>-hf)")
    ap.add_argument("--ms-id", default=None, help="ModelScope 仓库 ID 覆盖")
    ap.add_argument("--hf-id", default=None, help="HF 仓库 ID 覆盖")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--all", action="store_true", help="下载全部文件(默认跳过 *.bin 等)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path("models") / (
        args.model.split("/")[-1] + "-hf")
    patterns = None if args.all else DEFAULT_PATTERNS
    ms_id = args.ms_id or args.model
    hf_id = args.hf_id or args.model

    plan = {
        "modelscope": lambda: dl_modelscope(ms_id, out, patterns, args.revision),
        "hf-mirror": lambda: dl_hf(hf_id, out, patterns, args.revision, mirror=True),
        "hf": lambda: dl_hf(hf_id, out, patterns, args.revision, mirror=False),
    }
    order = ("modelscope", "hf-mirror", "hf") if args.source == "auto" \
        else (args.source,)

    errors = []
    for src in order:
        print(f"\n=== [{src}] downloading {ms_id if src == 'modelscope' else hf_id} "
              f"-> {out} ===")
        try:
            result = plan[src]()
        except Exception as e:  # noqa: BLE001 —— 逐源降级, 最后统一汇报
            print(f"=== [{src}] FAILED: {e}")
            errors.append(f"{src}: {e}")
            continue
        size = sum(f.stat().st_size for f in result.rglob("*") if f.is_file())
        print(f"=== [{src}] OK -> {result} ({size / 2**30:.2f} GiB)")
        print(f"[next] python tools/convert_lwc.py --src {result} "
              f"--out {result.with_name(result.name.replace('-hf', ''))}.lwc")
        return 0

    print("\nALL SOURCES FAILED:")
    for e in errors:
        print("  - " + e)
    print("hint: 国内网络优先 pip install modelscope; 或手动指定 --source hf-mirror")
    return 1


if __name__ == "__main__":
    sys.exit(main())
