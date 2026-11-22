"""Run the pinned Qwen3.5 mixer using the first base-checkpoint layer.

Usage: python generate_checkpoint_mixer_reference.py MODEL.py CACHE.py HEADER.json DIR
DIR contains the two locally extracted, checksum-verified layer segments.
The output reference JSON also stays in DIR and is not committed.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

model_path, cache_path, header_path, output_dir = map(Path, sys.argv[1:5])
header_bytes = header_path.read_bytes()
if hashlib.sha256(header_bytes).hexdigest() != "c894b6055113f9d5a9c5c68e703662de8ba27df5a9631b47f10b75ac3fe0b083":
    raise ValueError("unexpected base Safetensors header")
header = json.loads(header_bytes)
segments = []
for name, offset, digest in [
    ("qwen35-base-layer0-scalars.bin", 0, "274bf2ffc2f29f3381a158434c3028694c8859a2c1bafe83df567058fa2fafc7"),
    ("qwen35-base-layer0-block.bin", 508571776, "6c68593c6ecdc8ac34a85216d1db652e3fc02899fa6f3bce4a8ecfe227d39eb8"),
]:
    data = (output_dir / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"unexpected base checkpoint segment: {name}")
    segments.append((offset, data))


def tensor(name):
    entry = header["model.language_model.layers.0.linear_attn." + name]
    start, end = entry["data_offsets"]
    for offset, data in segments:
        if start >= offset and end <= offset + len(data):
            raw = bytearray(data[start - offset:end - offset])
            dtype = {"F32": torch.float32, "BF16": torch.bfloat16}[entry["dtype"]]
            return torch.frombuffer(raw, dtype=dtype).reshape(entry["shape"]).float().clone()
    raise ValueError(f"tensor is outside the verified segments: {name}")


# Reuse only the upstream-source loader and cache adapter from the synthetic
# generator. Its fixture-generation loop is deliberately not executed.
helper = Path(__file__).with_name("generate_mixer_reference.py")
tree = ast.parse(helper.read_text(encoding="utf-8"))
cut = next(i for i, node in enumerate(tree.body)
           if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "cases" for target in node.targets))
original_args = sys.argv[:]
sys.argv = [str(helper), str(model_path), str(cache_path)]
namespace = {"__file__": str(helper)}
exec(compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(helper), "exec"), namespace)
sys.argv = original_args

cfg = SimpleNamespace(hidden_size=1024, linear_num_key_heads=16, linear_num_value_heads=16,
                      linear_key_head_dim=128, linear_value_head_dim=128,
                      linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6,
                      layer_types=["linear_attention"])
model = namespace["context"]["Qwen3_5GatedDeltaNet"](cfg, 0).eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        source = tensor(name)
        if source.shape != parameter.shape:
            raise ValueError(f"unexpected checkpoint tensor shape: {name}")
        parameter.copy_(source)
hidden = namespace["pattern"]("checkpoint-hidden", (1, 3, cfg.hidden_size))
cache_type = namespace["Cache"]
with torch.no_grad():
    full_cache = cache_type()
    full = model(hidden, cache=full_cache)
    full_conv = full_cache.get_conv_state(0).clone()
    full_state = full_cache.get_recurrent_state(0).clone()
    chunk_cache = cache_type()
    prefix = model(hidden[:, :2], cache=chunk_cache)
    prefix_conv = chunk_cache.get_conv_state(0).clone()
    prefix_state = chunk_cache.get_recurrent_state(0).clone()
    suffix = model(hidden[:, 2:], cache=chunk_cache)
for actual, expected in [
    (torch.cat([prefix, suffix], dim=1), full),
    (chunk_cache.get_conv_state(0), full_conv),
    (chunk_cache.get_recurrent_state(0), full_state),
]:
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
flat = namespace["flat"]
result = {"hidden": flat(hidden), "output": flat(full),
          "conv_state": flat(full_conv), "state": flat(full_state),
          "prefix_output": flat(prefix), "prefix_conv_state": flat(prefix_conv),
          "prefix_state": flat(prefix_state)}
(output_dir / "qwen35-real-mixer-reference.json").write_text(
    json.dumps(result, allow_nan=False), encoding="utf-8"
)
print("generated real-weight f32 mixer reference")
