#!/usr/bin/env python3
"""Reassemble the verified Flash C1 hybrid as an ordinary GGUF file."""

import argparse
from contextlib import ExitStack
from hashlib import file_digest, sha256
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from tempfile import NamedTemporaryFile

PINS = Path(__file__).with_name("pins.json")
COPY_CHUNK = 16 * 1024 * 1024
ALIGNMENT = 32
PAGE = 4096
C1 = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.(weight|c1_rows|c1_table)$")
ELEMENT_TYPES = {
    0: (1, 4), 1: (1, 2), 7: (32, 24), 8: (32, 34), 12: (256, 144),
    13: (256, 176), 20: (32, 18), 30: (1, 2), 43: (16, 9), 44: (16, 9),
}
GGUF_FIXED_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pad(size, alignment):
    return (size + alignment - 1) // alignment * alignment


def encode_string(value):
    return struct.pack("<Q", len(value)) + value


def read_struct(stream, fmt):
    return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))


def read_string(stream):
    size, = read_struct(stream, "<Q")
    data = stream.read(size)
    require(len(data) == size, "truncated GGUF string")
    return data.decode()


def skip_value(stream, kind, count):
    if kind in GGUF_FIXED_WIDTH:
        stream.seek(GGUF_FIXED_WIDTH[kind] * count, io.SEEK_CUR)
    elif kind == 8:
        for _ in range(count):
            size, = read_struct(stream, "<Q")
            stream.seek(size, io.SEEK_CUR)
    elif kind == 9 and count == 1:
        element, count = read_struct(stream, "<IQ")
        skip_value(stream, element, count)
    else:
        raise ValueError(f"unexpected GGUF metadata field type {kind}")


def parse_header(header):
    stream = io.BytesIO(header)
    magic, version, tensor_count, metadata_count = read_struct(stream, "<4sIQQ")
    require((magic, version) == (b"GGUF", 3), "expected GGUF v3")
    metadata = {}
    for _ in range(metadata_count):
        start = stream.tell()
        key = read_string(stream)
        kind, = read_struct(stream, "<I")
        skip_value(stream, kind, 1)
        require(key not in metadata, f"duplicate GGUF metadata: {key}")
        if not key.startswith("split."):
            metadata[key] = header[start:stream.tell()]
    tensors = []
    for _ in range(tensor_count):
        name = read_string(stream)
        ndim, = read_struct(stream, "<I")
        require(1 <= ndim <= 4, f"unexpected tensor rank for {name}")
        shape = read_struct(stream, "<" + "Q" * ndim)
        kind, offset = read_struct(stream, "<IQ")
        tensors.append((name, shape, kind, offset))
    require(stream.tell() == len(header), "unexpected bytes after tensor directory")
    return metadata, tensors


def verify_file(stream, pin, headers_only=False):
    label = pin["local_path"]
    stat = os.fstat(stream.fileno())
    require(stat.st_size == pin["file_bytes"], f"wrong size for {label}")
    if not headers_only:
        result = file_digest(stream, "sha256").hexdigest()
        require(result == pin["sha256"], f"SHA-256 mismatch for {label}")
        require(os.fstat(stream.fileno()).st_mtime_ns == stat.st_mtime_ns, f"{label} changed during verification")
        stream.seek(0)
    return stat


def read_inputs(files, pins, headers_only):
    metadata = None
    sources, verified = [], []
    for idx, (stream, pin) in enumerate(zip(files, pins, strict=True)):
        verified.append(verify_file(stream, pin, headers_only))
        header = stream.read(pin["header_bytes"])
        require(sha256(header).hexdigest() == pin["header_sha256"], f"header mismatch for {pin['local_path']}")
        fields, tensors = parse_header(header)
        if idx == 1:
            require(not tensors, "expected metadata-only donor shard 1")
            metadata = fields
        sources.append(tensors)
    return metadata, sources, verified


def select_tensors(sources):
    compact = {tensor[0]: tensor for tensor in sources[0] if C1.fullmatch(tensor[0])}
    require(len(compact) == 432, "unexpected number of C1 banks and companion tensors")
    donor_names = set()
    chosen = []
    for idx, tensors in enumerate(sources[2:], 2):
        for tensor in tensors:
            name, shape, _, _ = tensor
            require(name not in donor_names, f"duplicate donor tensor {name}")
            donor_names.add(name)
            match = C1.fullmatch(name)
            if not match:
                chosen.append((idx, tensor))
                continue
            require(match[3] == "weight", f"unexpected donor tensor {name}")
            weight_type = 44 if match[2] == "down" else 43
            row_count = 2560 if match[2] == "down" else 640
            replacement = (
                ("weight", shape, weight_type),
                ("c1_rows", (row_count, 512), 1),
                ("c1_table", (2, 256, 512), 1),
            )
            for suffix, wanted_shape, wanted_type in replacement:
                row = compact[name.removesuffix("weight") + suffix]
                require(row[1:3] == (wanted_shape, wanted_type), f"unexpected compact tensor {row[0]}")
                chosen.append((0, row))
    require(len(donor_names) == 1224 and len(chosen) == 1512, "unexpected model geometry")
    return chosen


def assemble_plan(pin, metadata, sources):
    ple = pin["ple"]
    selected = select_tensors(sources)
    selected.sort(key=lambda item: item[1][0] != ple["tensor"])
    require(selected[0][1][0] == ple["tensor"], "missing donor PLE table")
    entries, table = [], bytearray()
    offset = 0
    for idx, (name, shape, kind, source_offset) in selected:
        block, width = ELEMENT_TYPES[kind]
        size = math.prod(shape) // block * width
        record = pin["inputs"][idx]
        src = pad(record["header_bytes"], ALIGNMENT) + source_offset
        require(shape[0] % block == 0 and src + size <= record["file_bytes"], f"invalid source range for {name}")
        new_type = {43: 150, 44: 151}.get(kind, kind)
        table += encode_string(name.encode()) + struct.pack("<I", len(shape))
        table += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", new_type, offset)
        entries.append((idx, src, size, offset))
        offset += pad(size, ALIGNMENT)
    prefix = b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(metadata) + 1) + b"".join(metadata.values())
    prefix += encode_string(b"ciru.storage.padding") + struct.pack("<I", 8)
    ple_index, ple_src, ple_size, _ = entries[0]
    alignment = (ple_src - len(prefix) - 8 - len(table)) % PAGE
    head = prefix + encode_string(b" " * alignment) + table
    require((pin["inputs"][ple_index]["local_path"], ple_src, ple_size, len(head)) ==
            (ple["source_file"], ple["source_offset"], ple["bytes"], ple["hybrid_offset"]), "PLE range mismatch")
    output = pin["output"]
    require(sha256(head).hexdigest() == output["header_sha256"], "assembled header SHA-256 mismatch")
    require(len(head) + offset == output["file_bytes"] and len(entries) == output["tensors"], "assembled layout mismatch")
    return head, entries


def stream_model(target, model_pin, files, source_stats, header, entries):
    digest = sha256()
    position = 0

    def append(data):
        nonlocal position
        if target is not None:
            target.write(data)
        digest.update(data)
        position += len(data)

    append(header)
    for idx, src, size, offset in entries:
        require(position == len(header) + offset, "unexpected generated offset")
        stream = files[idx]
        stream.seek(src)
        remaining = size
        while remaining:
            data = stream.read(min(remaining, COPY_CHUNK))
            require(bool(data), "input ended during tensor copy")
            append(data)
            remaining -= len(data)
        append(b"\0" * (pad(size, ALIGNMENT) - size))
    require(position == model_pin["file_bytes"] and digest.hexdigest() == model_pin["sha256"],
            "assembled model SHA-256 or output size mismatch")
    for src, before in zip(files, source_stats, strict=True):
        after = os.fstat(src.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "input changed during assembly")
    return digest.hexdigest()


def write_model(target, model_pin, files, source_stats, header, entries):
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), f"output exists: {target}")
    path = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as out:
            path = Path(out.name)
            digest = stream_model(out, model_pin, files, source_stats, header, entries)
            out.flush()
            os.fsync(out.fileno())
        os.link(path, target)
    finally:
        if path is not None:
            path.unlink()
    print(f"{digest}  {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_dir", type=Path, help="Directory containing compact/ and donor/")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--headers-only", action="store_true",
                         help="Validate the pinned header and inspect the output layout")
    actions.add_argument("--verify-assembly", action="store_true",
                         help="Verify the draft and the complete assembled byte stream without writing it")
    args = parser.parse_args()
    pin = json.loads(PINS.read_bytes())
    root = args.asset_dir.resolve()
    output = root / pin["output"]["local_path"]
    with ExitStack() as stack:
        files = [stack.enter_context((root / item["local_path"]).open("rb")) for item in pin["inputs"]]
        draft = stack.enter_context((root / pin["draft"]["local_path"]).open("rb"))
        verify_file(draft, pin["draft"], args.headers_only)
        metadata, tensors, stats = read_inputs(files, pin["inputs"], args.headers_only or args.verify_assembly)
        header, entries = assemble_plan(pin, metadata, tensors)
        if args.headers_only:
            print(f"Validated {len(entries)} tensors and pinned output header {pin['output']['header_sha256']}")
        elif args.verify_assembly:
            digest = stream_model(None, pin["output"], files, stats, header, entries)
            print(f"Verified draft, {len(entries)} assembled tensors and output digest {digest}")
        else:
            write_model(output, pin["output"], files, stats, header, entries)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, struct.error) as error:
        sys.exit(f"build-hybrid: {error}")
