"""Generate Qwen3.5 delta-rule fixtures from a hash-checked upstream file.

Usage: python generate_delta_reference.py MODELING_QWEN3_5.py
Requires PyTorch. No checkpoint weights or model code are copied into the
fixture: all inputs and the optional starting states are synthetic.
"""

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

source = Path(sys.argv[1]).read_bytes()
expected = "c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2"
if hashlib.sha256(source).hexdigest() != expected:
    raise ValueError("unexpected Transformers source digest")
names = {"l2norm", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"}
nodes = []
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        node.decorator_list = []
        nodes.append(node)
if {node.name for node in nodes} != names:
    raise ValueError("missing upstream reference function")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
context = {"torch": torch, "F": F, "is_torchdynamo_exporting": lambda: False}
exec(compile(module, "pinned_qwen3_5.py", "exec"), context)
recurrent = context["torch_recurrent_gated_delta_rule"]
chunked = context["torch_chunk_gated_delta_rule"]


def pattern(shape, seed, divisor):
    values = [((index * 17 + seed) % 37 - 18) / divisor
              for index in range(math.prod(shape))]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def flat(tensor):
    return tensor.flatten().tolist()


cases = []
for tokens, key_heads, value_heads, key_dim, value_dim, seeded in [
    (5, 2, 4, 3, 2, True),
    (4, 2, 2, 2, 3, False),
]:
    q = pattern((1, tokens, key_heads, key_dim), 1, 19)
    k = pattern((1, tokens, key_heads, key_dim), 3, 23)
    v = pattern((1, tokens, value_heads, value_dim), 7, 17)
    g = -pattern((1, tokens, value_heads), 9, 31).abs()
    beta = pattern((1, tokens, value_heads), 13, 21).sigmoid()
    initial = (pattern((1, value_heads, key_dim, value_dim), 19, 29) if seeded else None)

    def run(function, start, end, state):
        kw = {"initial_state": state.clone() if state is not None else None,
              "output_final_state": True, "use_qk_l2norm_in_kernel": True}
        if function is chunked:
            kw["chunk_size"] = 4
        return function(q[:, start:end], k[:, start:end], v[:, start:end],
                        g[:, start:end], beta[:, start:end], **kw)

    with torch.no_grad():
        full, final = run(recurrent, 0, tokens, initial)
        chunk_full, chunk_final = run(chunked, 0, tokens, initial)
        prefix, prefix_state = run(chunked, 0, 2, initial)
        suffix, suffix_state = run(chunked, 2, tokens, prefix_state)
        reset, reset_state = run(recurrent, 0, tokens, None)
    for actual, expected in [(chunk_full, full), (chunk_final, final),
                             (torch.cat([prefix, suffix], dim=1), full),
                             (suffix_state, final)]:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    cases.append({
        "tokens": tokens, "key_heads": key_heads, "value_heads": value_heads,
        "key_dim": key_dim, "value_dim": value_dim,
        "query": flat(q), "key": flat(k), "value": flat(v),
        "log_decay": flat(g), "beta": flat(beta),
        "initial_state": flat(initial) if initial is not None else None,
        "output": flat(full), "state": flat(final),
        "prefix_output": flat(prefix), "prefix_state": flat(prefix_state),
        "reset_output": flat(reset), "reset_state": flat(reset_state),
    })

Path(__file__).with_name("qwen35-delta-reference.json").write_text(
    json.dumps(cases, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
