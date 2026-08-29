#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_fake_hf.py — 微型 MoE HF 目录(转换器/链路自检)"""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

root = Path("models/_selftest-hf")
root.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)
H, I, E, V = 16, 32, 2, 64
tensors = {}


def fp16(shape):
    return rng.standard_normal(shape).astype(np.float16)


tensors["model.embed_tokens.weight"] = fp16((V, H))
tensors["model.norm.weight"] = fp16((H,)) + np.float16(1.0)
for l in range(2):
    tensors[f"model.layers.{l}.input_layernorm.weight"] = fp16((H,)) + np.float16(1.0)
    tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = fp16((H,)) + np.float16(1.0)
    tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = fp16((H, H))
    tensors[f"model.layers.{l}.self_attn.k_proj.weight"] = fp16((H, H))
    tensors[f"model.layers.{l}.self_attn.v_proj.weight"] = fp16((H, H))
    tensors[f"model.layers.{l}.self_attn.o_proj.weight"] = fp16((H, H))
    tensors[f"model.layers.{l}.mlp.gate.weight"] = fp16((E, H))  # router
    for e in range(E):
        for part, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
            tensors[f"model.layers.{l}.mlp.experts.{e}.{part}_proj.weight"] = fp16(shape)

save_file(tensors, str(root / "model-00001.safetensors"))
cfg = {
    "architectures": ["SelftestMoEForCausalLM"],
    "num_hidden_layers": 2,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": H,
    "n_routed_experts": E,
    "num_experts_per_tok": 2,
    "hidden_size": H,
    "moe_intermediate_size": I,
    "vocab_size": V,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "float16",
    "tie_word_embeddings": True,
}
(root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
# 极简 tokenizer stub for server smoke (optional)
print(f"fake HF model written to {root} ({len(tensors)} tensors)")
