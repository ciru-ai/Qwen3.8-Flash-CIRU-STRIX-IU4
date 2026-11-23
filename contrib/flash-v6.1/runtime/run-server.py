"""Serve Qwen3.8 Flash C1 using the v6.1 runtime on AMD Strix Halo."""
import argparse
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "runtime.json").read_bytes())
binary = root / "bin"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path, nargs="?")
parser.add_argument("draft", type=Path, nargs="?")
parser.add_argument("ple", type=Path, nargs="?")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--slot-dir", type=Path, default=Path("flash-v6.1-slots"))
parser.add_argument("--rocm-dir", type=Path, default=root / "rocm")
parser.add_argument("--compat-lib-dir", type=Path, help="Optional directory containing a compatible Linux loader and C runtime")
parser.add_argument("--check", action="store_true", help="Verify release files and check native loader resolution")
args = parser.parse_args()
rocm = args.rocm_dir.resolve(strict=True)

for directory, entries in ((binary, manifest["application_sha256"]), (rocm, manifest["rocm_sha256"])):
    for name, expected in entries.items():
        path = directory / name
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                parser.error(f"unexpected SHA-256 for {path}")
for name, target in manifest["application_links"].items():
    if (binary / name).readlink() != Path(target):
        parser.error(f"incorrect library link for {name}")
if not 1 <= args.port <= 65535:
    parser.error("port must be between 1 and 65535")

compat = args.compat_lib_dir.resolve(strict=True) if args.compat_lib_dir else None
loader = (compat / "ld-linux-x86-64.so.2") if compat else Path("/lib64/ld-linux-x86-64.so.2")
if not loader.is_file():
    parser.error("Linux loader unavailable. Supply its directory with --compat-lib-dir")
library_path = os.pathsep.join(str(p) for p in (binary, rocm, compat) if p)
server = binary / "llama-server"
command = [str(loader), "--library-path", library_path]
prefixes = ("CIRU_", "GGML_", "LLAMA_", "HALO_C1_", "HSA_", "FLASH_", "ROCPROF_", "ROCP_TOOL")
env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH") and not k.startswith(prefixes)}
env.update(LC_ALL="C", ROCBLAS_TENSILE_LIBPATH=str(rocm / "rocblas/library"),
           HIPBLASLT_TENSILE_LIBPATH=str(rocm / "hipblaslt/library"))
if args.check:
    command.extend(("--list", str(server)))
    os.execve(str(loader), command, env)
if args.model is None or args.draft is None or args.ple is None:
    parser.error("provide the assembled hybrid model, draft GGUF and pinned donor PLE shard")
weights = {}
for key, path in (("model", args.model), ("draft", args.draft), ("ple", args.ple)):
    path = path.resolve(strict=True)
    if not path.is_file():
        parser.error(f"missing GGUF file for {key}")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            parser.error(f"invalid GGUF header for {key}")
    weights[key] = path
slots = args.slot_dir.resolve()
slots.mkdir(parents=True, exist_ok=True)
values = dict(weights, bin=binary, host=args.host, port=args.port, slots=slots)
# The chat-template JSON is a literal argument. Only exact placeholders are substituted.
def argument(value):
    return str(values[value[1:-1]]) if value.startswith("{") and value[1:-1] in values else value

env.update({key: argument(value) for key, value in manifest["flags"].items()})
command += [str(server), *(argument(value) for value in manifest["server_arguments"])]
os.execve(str(loader), command, env)
