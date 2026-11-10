# Install and run Flash IU4 v6.0.0 on Strix Halo

This is a **custom CIRU runtime** for the supplied model. A stock llama.cpp
build cannot read its compact expert representation or its external embedding
table. Start with this repository's [model card](README.md) for model downloads,
checksums, benchmark results and model background. You do not need an earlier
CIRU installation or a GitHub release tag.

## Requirements

- Linux x86-64, AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), and **128 GB unified memory** for the measured configuration.
- A working AMD GPU driver with access to `/dev/kfd` and the render device in `/dev/dri/`. Use a current Strix Halo-capable kernel/firmware. The build script installs user-space software; it does not replace the driver.
- A local NVMe SSD for model files and the paged embedding table. Network storage performance is not covered by these measurements.
- Roughly **121 GiB for the target, draft and embedding files**, plus about **40 GiB of headroom for the source, SDK and build directories**. Keep extra space if retaining downloaded parts, a Hugging Face cache, or the previous model version.
- Ubuntu 24.04 LTS or a recent Debian release with a compatible kernel is the source-build path below. The historical benchmark runtime ran on **NixOS with ROCm/TheRock 10.0.0 and the pinned custom HIP/ROCr runtime**. The v5.5.0 serving qualification used a GNU 16.2.1 build made on Arch Linux and run on Ciru's NixOS system with that same SDK and custom GPU runtime. The v5.5.7 recurrent-state reset correction passed baseline/candidate CPU and HIP checks, an expanded CPU suite with no additional observed regressions, and OpenVINO CPU translation checks. The suite retains three baseline-matched exceptions; see [patch validation](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v5.5.7/docs/qualification/v5.5.7/RELEASE-NOTES.md). A corrected HIP library was also checked in four-slot text/thinking/vision serving on NixOS. These checks do not repeat the historical performance or broad quality benchmarks. Ubuntu/Debian commands have been checked against the source and package interfaces; the complete build and inference benchmark have **not** been repeated there.

The v5.5.8 graph-pool correction was checked with CPU pool-lifetime regressions,
native HIP graph-handle lifecycle checks and public GGML F16/F32 graph arithmetic
on gfx1151. These checks do not repeat model-quality or performance benchmarks.
See [v5.5.8 validation](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v5.5.8/docs/qualification/v5.5.8/RELEASE-NOTES.md).

Do not run another large GPU model while loading this one. The default reserves
262,144 tokens of context, one request slot, F16 target/draft KV caches, a
4 GiB embedding cache and a 1 GiB prompt cache. The model has substantial RAM
usage even with a short prompt; see the model card for measured resident memory.

## 1. Download the model and extract its source package

Use the download commands in [README.md](README.md). The resulting model directory
must contain these matched files:

```text
Qwen3.8-Flash-CIRU-STRIX-IU4-v5.0.gguf
mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf
ple/ple.payload.bin
ple/ple.manifest.json
ple/ple.scale.bf16
vision/mmproj-Qwen3.8-Flash-F16.mmproj
source-tree-flash-v6.0.0.tar.gz
```

The three `ple/` files contain the model's SSD-paged n-gram/hash embedding table,
including its scale and layout metadata. They are required model data. Compact
expert weights remain in the main GGUF; they are not stored in this sidecar.
Keep the target and MTP draft from the same variant. Flash and Orca drafts are
different. The v5.0 download commands retrieve complete files; no assembly is
required. Verify the provided checksums before starting the server.

From the downloaded model directory:

```bash
export MODEL_DIR="$PWD"
mkdir -p runtime-v6.0.0
tar -xzf source-tree-flash-v6.0.0.tar.gz -C runtime-v6.0.0
export RUNTIME_DIR="$MODEL_DIR/runtime-v6.0.0"
cd "$RUNTIME_DIR"
```

## Alternative: current GitHub source

The canonical source repository is
[ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4).
Its `main` branch includes the graph-pool lifetime, recurrent-state reset and prior automatic C1 corrections. For the exact source
of this release, use the public `v6.0.0` tag in a new directory instead of
extracting the archive above:

```bash
export MODEL_DIR="/absolute/path/to/flash-model"
git clone --branch v6.0.0 --single-branch \
  https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git \
  "$MODEL_DIR/runtime-v6.0.0"
export RUNTIME_DIR="$MODEL_DIR/runtime-v6.0.0"
cd "$RUNTIME_DIR"
```

Continue with the same build steps below. The model files stay in `MODEL_DIR`.
To follow subsequent source changes, use a separate checkout of `main`;
rebuild after updating it and record the exact commit used. Old source
archives and old release tags retain their original code.

## 2. Build on Ubuntu/Debian

This command installs host build packages through apt, creates a private Python
environment, installs the complete gfx1151 SDK pinned to **10.0.0**, builds
pwilkin's HIP/ROCr at commit
`7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1`, then builds the included engine.
It does not start a server. Run it as your normal user; apt uses sudo.

```bash
BUILD_JOBS=2 bash scripts/ciru/setup-linux-amd.sh --install-host-deps
```

With prerequisites already installed, omit `--install-host-deps`. They are:
`build-essential git ninja-build pkg-config libssl-dev libdrm-dev libelf-dev
libnuma-dev libgl-dev zlib1g-dev python3-venv ca-certificates curl`.
The private environment also pins CMake 4.1.2, CppHeaderParser 2.7.4 and ply 3.11.
Keep `.venv-rocm/`, `runtime/` and `build-gfx1151-sdk/` in place. Do not combine
the executable or shared libraries with another runtime build.

For an existing complete matching SDK, use the manual steps in
`runtime-v6.0.0/docs/BUILD_LINUX.md`. AMD documents the package installation at
[ROCm 10 installation](https://rocmdocs.amd.com/en/latest/install/rocm.html);
the custom runtime source is [pwilkin/rocm-systems](https://github.com/pwilkin/rocm-systems/tree/7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1).

## 3. Start the server

In the same terminal after building:

```bash
export ROCM_ROOT="$("$RUNTIME_DIR/.venv-rocm/bin/rocm-sdk" path --root)"
MODEL_VARIANT=IU4 MODEL_RELEASE=5.0 ENABLE_UI=0 ENABLE_VISION=1 \
  MODEL_DIR="$MODEL_DIR" bash "$RUNTIME_DIR/scripts/ciru/run-server.sh"
```

After setting `ROCM_ROOT`, the equivalent model-directory shortcut is
`bash "$MODEL_DIR/run-server-v6.0.0.sh"`. It uses the same rebuilt source
launcher and accepts the same environment overrides.

The default is API-only at `http://127.0.0.1:8080`, with image input enabled,
the matching projector loaded, and MTP depth **3**,
16,384-token batch, 8,192-token microbatch and 262,144-token context. The startup output prints
the resolved engine/HIP library and its checksum. A checksum from a fresh build
can differ from the benchmark binary because the compiler and distribution
differ. Loading and first-request warmup may take several minutes.

To launch again from a new terminal, set `MODEL_DIR` to the downloaded model
directory, `RUNTIME_DIR="$MODEL_DIR/runtime-v6.0.0"`, and repeat the ROCM_ROOT and
launch lines. These are filesystem paths on your computer, not repository IDs.

Use `CONTEXT_SIZE=65536` for a smaller maximum context. `ENABLE_MTP=0` disables
speculative decoding and permits omitting the draft file. One slot remains
the default. `PARALLEL_SLOTS=2` enables corrected concurrent MTP with one shared
262,144-token KV pool, microbatch 2,048 and four checkpoints per slot. MTP remains
enabled, but the single-slot draft-head shortlist is disabled, with a measured
single-request performance penalty. Read [CONCURRENCY.md](CONCURRENCY.md) for
copyable commands, capacity limits, exact sweeps and memory measurements.

The server stops with Ctrl-C. `HOST` and `PORT` select another address/port.
Keep the default loopback address for local clients; configure authentication
and network access separately before exposing the server to other machines.

## 4. Health, chat, and tools

In another terminal:

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-CIRU-STRIX-IU4","messages":[{"role":"user","content":"Explain unified memory in three sentences."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":256,"temperature":0.7}'
```

The benchmark's nonthinking mode is the request setting
`"chat_template_kwargs":{"enable_thinking":false}`. Set it to `true` to request
thinking; allow a larger output budget. The server uses the model's embedded
chat template with `--jinja`; do not substitute a template from another model.

Function calling uses the OpenAI-compatible `tools` and `tool_choice` fields:

```bash
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-CIRU-STRIX-IU4","messages":[{"role":"user","content":"What is the weather in Paris? Use the weather tool."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":256,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto"}'
```

The server returns a proposed call in `choices[0].message.tool_calls`; it does
not execute the function. Your client validates and executes allowed functions,
then sends the assistant tool-call message and a `role: "tool"` reply carrying
the matching `tool_call_id`, followed by another completion request. Tool calls
are model predictions and can be incorrect; the model card reports tool accuracy.

## Image input and optional web UI

API-only operation needs no Node.js or UI assets. The source includes the UI
application under `tools/ui`; a UI build is optional. To build it separately,
install a Node.js/npm version compatible with `tools/ui/package.json`, then run
`npm ci --ignore-scripts --no-audit --no-fund` and `npm run build` in `tools/ui`.
Launch with `ENABLE_UI=1 UI_DIR="$RUNTIME_DIR/tools/ui/dist"` after the resulting
directory contains `index.html`. UI assets are independent of inference kernels.

The default v6.0.0 launch loads `vision/mmproj-Qwen3.8-Flash-F16.mmproj`
and keeps MTP depth 3. `--vision` explicitly enables the same behavior.
Set `MMPROJ=/absolute/path/to/projector.mmproj` to override the location,
or `ENABLE_VISION=0` for text-only operation. The runtime must be rebuilt
from the current v6.0.0 source archive to retain the image/MTP attention
correction and install automatic C1 dispatch.
The target GGUF, MTP draft and projector do not need to be converted again.

Send an image as an OpenAI-compatible `image_url` content part. This
example uses a local PNG and a data URL, so no image-fetch support is required:

```bash
python3 - image.png <<'PY'
import base64, json, pathlib, sys, urllib.request
image = base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
payload = {
    "model": "Qwen3.8-Flash-CIRU-STRIX-IU4",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}
    ]}],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0, "max_tokens": 256
}
request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
PY
```

See [vision/README.md](vision/README.md) for projector provenance and the
separate image qualification. Text benchmark results do not measure
vision performance. Clients must also declare that this model accepts
images; loading the projector cannot change a client-side modality setting.

## Update from an earlier runtime

v6.0.0 fixes image input in prompts longer than about 2,048 tokens, speeds up
multi-slot serving and adds an optional disk tier for the prompt cache. It
includes every earlier correction (v5.5.6 automatic C1 dispatch, v5.5.7
recurrent-state reset, v5.5.8 graph-pool lifetime). No model conversion is
required. See the [v6.0.0 notes](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v6.0.0/docs/qualification/v6.0.0/RELEASE-NOTES.md).

To update:

1. Keep the verified v5.0 target, matching draft, PLE files and projector.
2. Download the new `source-tree-flash-v6.0.0.tar.gz`, `run-server-v6.0.0.sh` and
   `checksums-v6.0.0.sha256` with the command in the model card, or clone the
   `v6.0.0` tag of [the source repository](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4), which builds both the IU4
   and Orca variants.
3. Extract into a **new** runtime directory (`runtime-v6.0.0`) and run the
   complete build in step 2 of this guide. The server, core library and HIP
   backend changed together; do not copy single libraries into an older build.
4. Stop your server, point `RUNTIME_DIR` at the new directory and start it
   again. Keep the previous runtime until the new one has loaded and answered
   a request.

Launch options are unchanged for one slot. With `PARALLEL_SLOTS` above 1 the
launcher enables the v6 multi-slot paths by itself; see
[CONCURRENCY.md](CONCURRENCY.md). To keep large conversation states on disk,
add `PROMPT_CACHE_DISK_MIB=65536` (MiB of local disk the cache may use) and
optionally `PROMPT_CACHE_DISK_DIR=/fast/local/path`.

The older single-library vision patch below does not install the current
package.

## Troubleshooting

- GPU access: confirm `/dev/kfd` and `/dev/dri/render*` exist and your account has the required device permissions. Follow your distribution's AMD GPU setup for the `render`/`video` groups, then log in again.
- Missing HIP/ROCr: finish `build-custom-runtime.sh`; `runtime/hip/lib/libamdhip64.so` and `runtime/rocr/lib/libhsa-runtime64.so` must both exist.
- Missing SDK packages: use the complete pinned `rocm[libraries,devel,device-gfx1151]` installation. Do not mix SDK versions or repair missing packages with cross-version symlinks.
- Out of memory: stop other large GPU workloads, reduce `CONTEXT_SIZE`, or use `ENABLE_MTP=0`. Lowering context does not reduce the resident target weights.
- Missing embedding files: assemble and verify all three `ple/` files. The model cannot operate with the main GGUF alone.
- Image ignored in a long or tool-heavy prompt: update to v6.0.0; v5.5.0 through v5.5.8 lose the image once the prompt passes about 2,048 tokens.
- Unexpected speed: check the reported GPU, selected source/build, model variant, MTP mode, context depth and batch sizes. Prompt processing at an existing depth differs from filling an empty context.

<a id="update-an-existing-v50-build"></a>

## Historical v5.0 to v5.0.1 vision-only patch

The following commands are retained only for reproducing the older v5.0.1 vision update. They do not install the current concurrency or input-memory corrections.

Existing matching model, draft, embedding and projector weights can be reused.
The [small source patch](vision/qsa-position-gap-v5.0.1.patch) changes only
`src/llama-memory-hybrid-idx.cpp`. It applies to both released v5.0 source
trees. Rebuilding the `llama` target updates its C++ library and reuses the
existing GPU kernels. Stop the server before rebuilding so its loaded libraries
are not replaced during a request. Keep the original source and build directory
together; do not substitute a library from another installation.

From the existing model directory, create a small download environment if needed:

```bash
export MODEL_DIR="$PWD"
python3 -m venv .venv-hf
.venv-hf/bin/pip install --upgrade huggingface_hub
.venv-hf/bin/hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 --revision v5.0.1 \
  vision/qsa-position-gap-v5.0.1.patch vision/mmproj-Qwen3.8-Flash-F16.mmproj \
  run-server-v5.0.1.sh upgrade-v5.0.1.sha256 INSTALL.md --local-dir "$MODEL_DIR"
sha256sum -c upgrade-v5.0.1.sha256
export RUNTIME_DIR="$MODEL_DIR/runtime-v5.0"
export BUILD_DIR="$RUNTIME_DIR/build-gfx1151-sdk"
cd "$RUNTIME_DIR"
patch --dry-run -p1 < "$MODEL_DIR/vision/qsa-position-gap-v5.0.1.patch"
patch -p1 < "$MODEL_DIR/vision/qsa-position-gap-v5.0.1.patch"
"$RUNTIME_DIR/.venv-rocm/bin/cmake" --build "$BUILD_DIR" --target llama -j2
export ROCM_ROOT="$("$RUNTIME_DIR/.venv-rocm/bin/rocm-sdk" path --root)"
bash "$MODEL_DIR/run-server-v5.0.1.sh"
```

If `patch` is missing on Ubuntu/Debian, install it with `sudo apt-get install
patch`. If the dry run fails, the source differs from the original release or
already contains this correction; use the complete v5.0.1 source-build route
above instead of forcing a patch. Adjust `RUNTIME_DIR` and `BUILD_DIR` when the
existing installation uses other directory names. The new wrapper sets image
input on even when using the old launcher's source; `ENABLE_VISION=0` opts out.

## Previous versions

v5.0.1 uses the unchanged v5.0 target filenames and its own runtime directory.
The original v5.0 source archives and launchers remain available at tag `v5.0.0`. Keep the old
GGUF and old matching runtime to continue using v4.4.1; do not overwrite either.
Source licenses and contributor notices are included in the runtime archive.
