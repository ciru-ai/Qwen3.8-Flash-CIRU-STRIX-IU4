"""Extract the pinned HaloBox ROCm libraries beside the Flash release runtime."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

EXPECTED = "53f93341b360099cf8ea9018ae0cdce6af2386741ed2436e51197eebedd07d01"
URL = "https://github.com/halo-box/strix-llama.cpp/releases/download/b1064/llama-b1064-ubuntu-rocm-gfx1151-x64.zip"


def install(archive, destination):
    with archive.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != EXPECTED:
        raise SystemExit("HaloBox archive SHA-256 does not match b1064")
    if destination.exists():
        raise SystemExit(f"ROCm directory already exists at {destination}")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".rocm-") as temp:
        stage = Path(temp) / "runtime"
        stage.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if member.is_dir() or not member.filename.startswith(("lib", "rocblas/", "hipblaslt/")):
                    continue
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(member.filename)
                out = stage.joinpath(*path.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, out.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
        stage.rename(destination)
    print(destination)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, help="Use a local copy of the pinned HaloBox archive")
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "rocm")
args = parser.parse_args()
if args.archive:
    install(args.archive, args.output)
else:
    with tempfile.TemporaryDirectory(prefix="flash-rocm-") as temp:
        archive = Path(temp) / "b1064.zip"
        with urllib.request.urlopen(URL, timeout=60) as source, archive.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        install(archive, args.output)
