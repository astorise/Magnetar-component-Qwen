"""Extract the first Qwen3.5 mixer after verifying the whole base checkpoint.

Usage: python extract_checkpoint_mixer.py CHECKPOINT.safetensors OUTPUT_DIR
Or:    python extract_checkpoint_mixer.py --download OUTPUT_DIR
Only the verified header and two small byte ranges are retained.
"""

import contextlib
import hashlib
import os
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

CHECKPOINT_SHA = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"
HEADER_SHA = "c894b6055113f9d5a9c5c68e703662de8ba27df5a9631b47f10b75ac3fe0b083"
URL = ("https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/resolve/"
       "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/"
       "model.safetensors-00001-of-00001.safetensors?download=true")
RANGES = [
    ("qwen35-base-layer0-scalars.bin", 0, 576,
     "274bf2ffc2f29f3381a158434c3028694c8859a2c1bafe83df567058fa2fafc7"),
    ("qwen35-base-layer0-block.bin", 508571776, 529658016,
     "6c68593c6ecdc8ac34a85216d1db652e3fc02899fa6f3bce4a8ecfe227d39eb8"),
]
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
source = urllib.request.urlopen(URL, timeout=120) if sys.argv[1] == "--download" else open(sys.argv[1], "rb")

with contextlib.closing(source) as stream, tempfile.TemporaryDirectory(dir=output) as temp:
    temporary = Path(temp)
    digest = hashlib.sha256()
    length_bytes = stream.read(8)
    if len(length_bytes) != 8:
        raise ValueError("truncated Safetensors length")
    header_length = struct.unpack("<Q", length_bytes)[0]
    if header_length != 59840:
        raise ValueError("unexpected Safetensors header length")
    header = stream.read(header_length)
    if len(header) != header_length or hashlib.sha256(header).hexdigest() != HEADER_SHA:
        raise ValueError("unexpected Safetensors header")
    digest.update(length_bytes)
    digest.update(header)
    files = [(temporary / name).open("wb") for name, *_ in RANGES]
    offset = 0
    try:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            end = offset + len(chunk)
            for (_, begin, finish, _), target in zip(RANGES, files):
                left, right = max(offset, begin), min(end, finish)
                if left < right:
                    target.write(chunk[left - offset:right - offset])
            offset = end
    finally:
        for target in files:
            target.close()
    if offset + 8 + header_length != 1746942600 or digest.hexdigest() != CHECKPOINT_SHA:
        raise ValueError("base checkpoint size or SHA-256 does not match")
    for name, begin, finish, expected in RANGES:
        path = temporary / name
        if path.stat().st_size != finish - begin or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"unexpected extracted segment: {name}")
    (temporary / "kev-qwen35-base-header.json").write_bytes(header)
    for name in ["kev-qwen35-base-header.json", *(item[0] for item in RANGES)]:
        os.replace(temporary / name, output / name)
print("verified base checkpoint and extracted first mixer")
