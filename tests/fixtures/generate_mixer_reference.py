"""Run the pinned upstream Qwen3.5 mixer and cache on synthetic weights.

Usage: python generate_mixer_reference.py MODELING_QWEN3_5.py CACHE_UTILS.py
Requires PyTorch. The fixture contains synthetic weights only.
"""

import ast
import hashlib
import json
import math
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sources = [
    (Path(sys.argv[1]), "c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2"),
    (Path(sys.argv[2]), "29651e0d89cc34d1af4b7d5d7f2c5dd81b584c8c5336f15652218640ea8465e5"),
]
trees = []
for path, digest in sources:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"unexpected upstream digest: {path.name}")
    trees.append(ast.parse(data))

functions = {"apply_mask_to_padding_states", "causal_conv1d_update", "causal_conv1d_fn",
             "l2norm", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"}
classes = {"Qwen3_5RMSNormGated", "Qwen3_5GatedDeltaNet"}
cache_classes = {"LinearAttentionCacheLayerMixin", "LinearAttentionLayer"}
nodes = []
for node in trees[0].body:
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        node.decorator_list = []
        nodes.append(node)
    if isinstance(node, ast.ClassDef) and node.name in classes:
        node.decorator_list = []
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method.decorator_list = []
        nodes.append(node)
for node in trees[1].body:
    if isinstance(node, ast.ClassDef) and node.name in cache_classes:
        nodes.append(node)
expected = functions | classes | cache_classes
if {node.name for node in nodes} != expected:
    raise ValueError("missing upstream mixer or cache definition")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
context = {"torch": torch, "nn": nn, "F": F, "ACT2FN": {"silu": F.silu},
           "ABC": ABC, "abstractmethod": abstractmethod,
           "is_torchdynamo_exporting": lambda: False,
           "is_torchdynamo_compiling": lambda: True}
exec(compile(module, "pinned_qwen35_mixer.py", "exec"), context)
chunk = context["torch_chunk_gated_delta_rule"]
context["torch_chunk_gated_delta_rule"] = lambda *args, **kw: chunk(*args, chunk_size=4, **kw)


class Cache:
    """Expose the upstream cache-layer methods through the model's cache API."""

    def __init__(self):
        self.layer = context["LinearAttentionLayer"]()

    def get_conv_state(self, layer_idx, state_idx=0):
        return self.layer.conv_states[state_idx] if self.layer.has_previous_state[state_idx] else None

    def get_recurrent_state(self, layer_idx, state_idx=0):
        return self.layer.recurrent_states[state_idx] if self.layer.has_previous_state[state_idx] else None

    def update_conv_state(self, values, layer_idx, **kw):
        return self.layer.update_conv_state(values, **kw)

    def update_recurrent_state(self, values, layer_idx, **kw):
        return self.layer.update_recurrent_state(values, **kw)

    def is_recording_past(self, layer_idx):
        return self.layer.record_past


def pattern(name, shape):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) % 97
    values = [((i * 17 + seed) % 101 - 50) / 113.0 for i in range(math.prod(shape))]
    tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
    return tensor + 1.0 if name == "norm.weight" else tensor


def flat(tensor):
    return tensor.flatten().tolist()


cases = []
for key_heads, value_heads in [(1, 2), (2, 2)]:
    cfg = SimpleNamespace(hidden_size=6, linear_num_key_heads=key_heads,
                          linear_num_value_heads=value_heads, linear_key_head_dim=2,
                          linear_value_head_dim=3, linear_conv_kernel_dim=3,
                          hidden_act="silu", rms_norm_eps=1e-5,
                          layer_types=["linear_attention"])
    model = context["Qwen3_5GatedDeltaNet"](cfg, 0).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(pattern(name, parameter.shape))
    hidden = pattern("hidden", (1, 5, cfg.hidden_size))
    mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.int64)
    with torch.no_grad():
        full_cache = Cache()
        full = model(hidden, cache=full_cache, attention_mask=mask)
        full_conv = full_cache.get_conv_state(0).clone()
        full_recurrent = full_cache.get_recurrent_state(0).clone()
        chunks = Cache()
        prefix = model(hidden[:, :2], cache=chunks, attention_mask=mask[:, :2])
        prefix_conv = chunks.get_conv_state(0).clone()
        prefix_recurrent = chunks.get_recurrent_state(0).clone()
        suffix = model(hidden[:, 2:], cache=chunks, attention_mask=mask[:, 2:])
        steps = Cache()
        step_outputs = [model(hidden[:, i:i+1], cache=steps, attention_mask=mask[:, i:i+1])
                        for i in range(hidden.shape[1])]
        chunks.layer.reset()
        reset = model(hidden, cache=chunks, attention_mask=mask)
    for actual, expected_value in [
        (torch.cat([prefix, suffix], 1), full), (torch.cat(step_outputs, 1), full),
        (reset, full), (steps.get_conv_state(0), full_conv),
        (steps.get_recurrent_state(0), full_recurrent),
        (chunks.get_conv_state(0), full_conv), (chunks.get_recurrent_state(0), full_recurrent),
    ]:
        torch.testing.assert_close(actual, expected_value, atol=4e-6, rtol=1e-6)
    cases.append({
        "hidden_size": cfg.hidden_size, "tokens": 5, "key_heads": key_heads,
        "value_heads": value_heads, "key_dim": 2, "value_dim": 3, "kernel": 3,
        "epsilon": cfg.rms_norm_eps, "hidden": flat(hidden), "mask": flat(mask),
        "weights": {name: flat(parameter) for name, parameter in model.named_parameters()},
        "output": flat(full), "conv_state": flat(full_conv), "state": flat(full_recurrent),
        "prefix_output": flat(prefix), "prefix_conv_state": flat(prefix_conv),
        "prefix_state": flat(prefix_recurrent),
    })

Path(__file__).with_name("qwen35-mixer-reference.json").write_text(
    json.dumps(cases, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
