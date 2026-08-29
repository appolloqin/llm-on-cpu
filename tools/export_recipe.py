#!/usr/bin/env python3
"""HF config.json → models/recipes/*.json (P1 LayerSpec)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(hf_config: Path, out: Path, name: str) -> None:
    root = json.loads(hf_config.read_text(encoding="utf-8"))
    tc = root.get("text_config", root)
    layers = []
    types = tc.get("layer_types")
    n = int(tc.get("num_hidden_layers", 32))
    if not types:
        types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)
        ]
    attn = {
        "num_heads": tc.get("num_attention_heads", 16),
        "num_kv_heads": tc.get("num_key_value_heads", 4),
        "head_dim": tc.get("head_dim", 256),
        "q_gate": bool(tc.get("attn_output_gate", True)),
    }
    linear = {
        "num_key_heads": tc.get("linear_num_key_heads", 16),
        "num_value_heads": tc.get("linear_num_value_heads", 32),
        "key_head_dim": tc.get("linear_key_head_dim", 128),
        "value_head_dim": tc.get("linear_value_head_dim", 128),
        "conv_kernel": tc.get("linear_conv_kernel_dim", 4),
    }
    mlp = {"intermediate_size": tc.get("intermediate_size", 9216), "act": "silu"}
    for t in types:
        layers.append({"type": t, "attn": attn, "linear": linear, "mlp": mlp})
    rope = tc.get("rope_parameters", {})
    recipe = {
        "name": name,
        "hidden_size": tc.get("hidden_size", 2560),
        "vocab_size": tc.get("vocab_size", 248320),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rms_one_plus_weight": True,
        "rope": {
            "theta": rope.get("rope_theta", 10000000),
            "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25),
        },
        "tie_word_embeddings": tc.get("tie_word_embeddings", True),
        "weight_prefix": "language_model.",
        "layers": layers,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(layers)} layers)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="qwen3_5")
    args = ap.parse_args()
    export(Path(args.hf_config), Path(args.out), args.name)


if __name__ == "__main__":
    main()
