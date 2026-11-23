#!/usr/bin/env bash
# Materialize the complete source of the qualified gfx1151 research runtime.
set -euo pipefail
cd "$(dirname "$0")"
notes=$PWD
revision=b9f3c0e878f82f80ea1528969f305017a909b73e
out=$(realpath -m "${1:?usage: prepare.sh NEW_SOURCE_DIRECTORY [LOCAL_HALOBOX_REPOSITORY]}")
repo=${2:-https://github.com/halo-box/strix-llama.cpp.git}
test ! -e "$out"
( cd "$notes"; sha256sum -c PATCH-SHA256SUMS )
git_dir=$(mktemp -d)
trap 'rm -rf -- "$git_dir"' EXIT
git -C "$git_dir" init -q
git -C "$git_dir" fetch -q --no-tags --depth=1 "$repo" "$revision"
test "$(git -C "$git_dir" rev-parse FETCH_HEAD)" = "$revision"
mkdir "$out"
git -C "$git_dir" archive "$revision" | tar -x -C "$out"
(
    cd "$out"
    git apply --check "$notes/halobox-b1064-flash-v6.1.patch"
    git apply "$notes/halobox-b1064-flash-v6.1.patch"
    LC_ALL=C find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | cmp - "$notes/SOURCE-SHA256SUMS"
)
printf 'Verified patched source at %s\n' "$out"
