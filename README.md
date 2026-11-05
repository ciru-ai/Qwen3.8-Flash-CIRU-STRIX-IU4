---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - Qwen/Qwen3.8-Flash-Next-FP8
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: image-text-to-text
inference: false
tags:
  - qwen
  - gguf
  - amd
  - rocm
  - gfx1151
  - strix-halo
  - iu4
  - mtp
  - long-context
  - vision
---

# Qwen3.8 Flash CIRU Strix IU4 — v5.5.0 runtime update

Compact weights and a native CIRU runtime for AMD Strix Halo. **The original
v4.4.1 GGUF remains in this repository** at its existing filename. v5.0 is
an additional, versioned GGUF. Both use the same multi-token prediction
(MTP) draft model and the required SSD-paged embedding files under
`ple/`.
This package is for Flash. Orca uses a separate model package and its own
matching draft; do not interchange their model or draft files.

> [!IMPORTANT]
> This is a custom-runtime model with text and image input. Stock llama.cpp and hosted
> inference do not support its compact expert format or required SSD-paged
> embedding table.
> Build the included matching [CIRU runtime](INSTALL.md)
> and keep all three `ple/` files on fast local storage.

## One slot by default; optional multi-slot serving

Runtime **v5.5.0** retains the v5 model and runtime features, adding the
corrected concurrent MTP/attention paths and post-image memory-pressure fix.
The existing v5.0 weights are unchanged.
The default remains **one slot, 262,144 context tokens, MTP3**, with a
16,384-token batch and an **8,192-token microbatch**.

After rebuilding the matching source as described in [INSTALL.md](INSTALL.md),
opt into two slots by adding `PARALLEL_SLOTS=2` to the launch command:

```bash
PARALLEL_SLOTS=2 MODEL_VARIANT=IU4 MODEL_RELEASE=5.0 ENABLE_UI=0 \
  MODEL_DIR="$MODEL_DIR" RUNTIME_DIR="$RUNTIME_DIR" \
  bash "$RUNTIME_DIR/scripts/ciru/run-server.sh"
```

Two slots share **one 262,144-token context pool**, use microbatch 2,048 and
four retained checkpoints per slot. A request can borrow most of the pool
while its peer is idle; both requests cannot each consume the full pool at
the same time. **Multi-slot mode has a performance penalty:** it disables
the single-slot MTP draft-head shortlist. MTP itself remains enabled.
The matched IU4 test measured **47.00 to 42.20 tok/s (-10.21%)** for one
active request when the second slot was configured but idle. Both test arms
used microbatch 2,048; this is not a comparison of the launcher defaults.

The completed IU4 226,712-token, two-image replay finished without swap or
OOM. Its minimum available RAM improved from **7.788 to 13.162 GiB** with
the fenced single-copy input path at microbatch 8,192. This addresses the
demonstrated replay workspace pressure; it is not a claim that all possible
image-memory failures or an idle leak have been eliminated.

The rebuilt runtime also passed **eight image checks across IU4 and Orca,
each in one- and two-slot modes** on Ciru. Orca's separate 226,712-token
two-image replay completed with **13.490 GiB minimum available RAM**,
no swap and no memory-guard event. This confirms the shared correction is
installed and exercised for both variants; there is no matched Orca
before/after memory-gain claim. See the [release check record](benchmarks/release-20261006/validation.json).

See [multi-slot instructions, sweeps and machine/settings details](CONCURRENCY.md)
for the 256K pool comparison, the separate 1/2/4-request diagnostic, optional
GDN prefill results, memory budgets and validation limits. The historical
IU4 measurements must not be read as Orca benchmark results. Existing v5.0
and v5.0.1 benchmark and vision records below remain preserved.

## Earlier vision update in v5.0.1

The v5.0.1 runtime enables image input by default with the matching
projector and keeps MTP depth 3. It restores the QSA attention
position-gap correction needed when image tokens are followed by
speculative decoding. Model, draft and projector weights are unchanged.
Install the new source package and rebuild the runtime; replacing only
the launcher does not apply this fix. Existing v5 users can apply the
[small source update](INSTALL.md#update-an-existing-v50-build) without
rebuilding GPU kernels. Set `ENABLE_VISION=0` for a
text-only launch. See [vision setup and validation](vision/README.md).

On Radeon 8060S, Flash IU4 passed six functional checks with its
default MTP depth 3: two separate image descriptions, image
follow-up, two images in one request, a thinking-mode image request,
and a text regression. These are synthetic-image smoke tests, not a
broad vision benchmark. [Validation record](vision/validation-v5.0.1.json).
The speed and memory tables below remain the v5.0 text-only measurements.
Loading the vision projector adds encoder memory; those tables do not
measure the new image-enabled default.

## What changed in v5.0

The routed expert weights use a **4.5175-bits-per-weight compact indexed
affine representation**. Its GGUF is 72,112,333,312 bytes (67.16 GiB),
6.785 GiB smaller than the retained v4.4.1 GGUF. The runtime uses native
packed IU4 for short expert operations and tiled BF16 decoding for matrix
prefill. Projection, recurrent-state, SSD paging and long-context attention
paths were also improved. IU4 here means four-bit integer dot-product arithmetic
on the AMD GPU; it does not mean every model weight uses four-bit storage.
These are complete-model gains: the large prefill improvements are **not
established as IU4-specific**. The limits of the same-value IU8
comparison are explained in [BENCHMARKS.md](BENCHMARKS.md#interpretation).

## How the model works

Qwen3.8 Flash is a mixture-of-experts model: each token activates a
selected subset of expert networks. This release adapts its storage and
inference kernels for 128-GiB AMD Strix Halo systems. It is intended for
local chat, coding and tool-enabled applications. It is a quantized adaptation of the base model with a matching native runtime.

The v5.0 expert representation stores four-bit weight codes plus compact
indices into scale/offset tables and per-row multipliers. Including those
extra values, the expert bank averages 4.5175 bits per weight. The rest
of the model keeps mixed higher-precision formats; the entire model is
not uniformly four-bit.

For short expert operations, the runtime uses packed four-bit integer
dot products, activation scaling and sums, and the matching affine
corrections. For long prompt batches, it reconstructs only the temporary
BF16 tiles needed by the GPU and reuses efficient batched expert matrix
kernels. The resident expert weights remain compact. These two paths let
the runtime use appropriate arithmetic for different workloads.

A separate large n-gram embedding table is paged from fast local NVMe
through the three required `ple/` files. It is distinct from the compact
routed experts. Multi-token prediction (MTP) uses the matching draft to
propose tokens that the main model verifies; the default depth is three.
Thinking can be enabled for reasoning or disabled for direct responses.
The published task comparison below uses thinking off throughout.

## Benchmarks at a glance

The complete [benchmark report](BENCHMARKS.md) contains **all six tested
stacks**: CIRU v4.4.1, CIRU v5.0, HaloBox, Gufo, AgentionAI AP-Q5_K_XL,
and Halogen. It includes all **18 prompt/depth points per stack through
256,000 prior tokens**, each requesting 512 output tokens; full-prefix
prefill; objective accuracy; resident memory; HumanEval 0–9 speed;
the 15-case hard tool set; and two HermesAgent-20 passes. Download the
[108-row speed CSV](speed-grid.csv) or [numerical results](benchmark-data.json).

### Historical v5.0 versus retained v4.4.1

Same Ryzen AI Max+ 395 / Radeon 8060S, one slot, 262,144 context,
F16 attention cache and MTP depth 3. Each request appends 2,048 new tokens
to the cached depth, then generates 512. Each row is one observation;
PP is prompt tok/s and TG is generation tok/s. TG uses the server's
511 inter-token timing convention.

| Prior depth | v4.4.1 PP | v5.0 PP | v4.4.1 TG | v5.0 TG |
| --- | --- | --- | --- | --- |
| 64,000 | 716.52 | 902.73 | 42.91 | 55.48 |
| 256,000 | 453.78 | 675.56 | 32.58 | 51.39 |

The full six-stack suite used the preceding v5.0 build. The release build
above preserves the measured output token IDs and selected attention
positions/scores, but the full quality panel and task suite were not rerun
after its final long-context change.

### Complete-suite quality, memory and behavior

The objective panel scores 2,048 next-token positions against the same
BF16 full-vocabulary reference. Lower KL/PPL is better; higher top-1
agreement is better. Native rows use their actual width-four path.
Memory below is loaded GPU-visible unified-memory allocation, not total
physical memory. These are different complete weights/runtime stacks.

| Stack | KL ↓ | Top-1 ↑ | Tail PPL ↓ | Ready GPU GiB |
| --- | --- | --- | --- | --- |
| CIRU v4.4.1 | 0.056117 | 93.750% | 2.05516 | 90.60 |
| CIRU v5.0 | 0.050030 | 94.482% | 2.04505 | 87.20 |
| HaloBox | 0.029182 | 96.143% | 2.03642 | 93.40 |
| Gufo | 0.027966 | 96.436% | 2.03502 | 88.54 |
| AgentionAI AP | 0.029135 | 96.436% | 2.02636 | 99.64 |
| Halogen | 0.085905 | 92.773% | 2.07968 | 34.47† |

† Halogen also had 78.16 GiB engine process PSS at 64K. GTT, RSS/PSS and
system RAM overlap on unified memory and must not be added. See the
[resident-memory details](BENCHMARKS.md#actual-resident-memory).
The small quality panel does not establish long-context factual accuracy.

| Stack | HumanEval speed TG | Hard tools /100 | Hermes pass 1 /100 | Hermes pass 2 /100 |
| --- | --- | --- | --- | --- |
| CIRU v4.4.1 | 60.77 | 67 | 89 | 88 |
| CIRU v5.0 | 55.70 | 80 | 93 | 96 |
| HaloBox | 45.84 | 63 | 90 | 96 |
| Gufo | 58.75 | 73 | 95 | 95 |
| AgentionAI AP | 46.42 | 73 | 91 | 91 |
| Halogen | 69.28 | 70 | 93 | 92 |

HumanEval 0–9 produced ten normal first responses on every stack and was
used for speed, **not code pass@1**. v5.0 was slower on this short-generation
check than v4.4.1. Hermes uses a fixed turn budget, and scores can coexist
with incomplete runs; exact completion counts, harness corrections and
client-error audits are in [the report](BENCHMARKS.md#thinking-off-task-checks).

## Choose a version

| File | Purpose | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `Qwen3.8-Flash-CIRU-STRIX-IU4.gguf` | Retained v4.4.1 | 79,397,818,720 | `c0ea11e4e24d0f909720b6c4e7462aa1e6fbf5e0f6acc796063f2aed4cf46ed0` |
| `Qwen3.8-Flash-CIRU-STRIX-IU4-v5.0.gguf` | New compact v5.0 | 72,112,333,312 | `745bf5b48ef5bd4ded36b3136cde3d35145b2ec5a3aabf03da0e5a792bc5a27d` |
| `mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf` | Shared draft | 4,135,893,248 | `e6743badef1f2619fcb5addfa4344a2a3368cb75214735117e3af80c70b80642` |
| `ple/ple.payload.bin` | Required SSD-paged embedding table | 52,429,053,952 | `687fc742efb6888c6cd7cf9c80cb4b1ac8cb4707b9409c206699c43363e239b2` |

The other required embedding files are `ple.manifest.json` and
`ple.scale.bf16`; the v5.0 model and installation files are listed in
[`checksums-v5.5.0.sha256`](checksums-v5.5.0.sha256). The v4.4.1 file is retained
unchanged for existing users and can be downloaded alone with its matching
runtime.

## Download and run v5.5.0

Download the versioned GGUF, matching MTP draft, the three embedding-table
files, the matching vision projector, and the complete source package. Existing users can reuse the same
verified MTP and `ple/` files.

```bash
python3 -m venv .venv-hf
.venv-hf/bin/pip install --upgrade huggingface_hub
.venv-hf/bin/hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  --revision v5.5.0 \
  Qwen3.8-Flash-CIRU-STRIX-IU4-v5.0.gguf \
  mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf \
  ple/ple.payload.bin ple/ple.manifest.json ple/ple.scale.bf16 \
  vision/mmproj-Qwen3.8-Flash-F16.mmproj \
  source-tree-flash-v5.5.0.tar.gz run-server-v5.5.0.sh INSTALL.md CONCURRENCY.md checksums-v5.5.0.sha256 \
  --local-dir ./flash-model
cd flash-model
sha256sum -c checksums-v5.5.0.sha256
```

On Ubuntu/Debian, install `python3-venv` first if needed. Continue with
**[INSTALL.md](INSTALL.md)** for the complete source build and launch steps.
The included source archive supplies the v5.0 inference code; the guide builds
its pinned AMD SDK and custom HIP/ROCr dependencies from public sources.
It also covers health checks, chat, function calling, thinking settings,
image input, and common setup errors. No earlier CIRU installation or
unpublished GitHub tag is required.

Use 128 GiB unified memory and fast local NVMe for the measured configuration.
The model files take about 120 GiB, with additional space needed for the SDK
and build. The default local server uses one slot, 16,384-token batches and 8,192-token microbatches,
262,144-token context, F16 attention cache, and MTP depth three.
Performance was measured on NixOS; the Ubuntu/Debian guide is a source-build
route, not a claim of equal measured performance on every distribution.

The download includes the matching vision projector, and the v5.5.0
launcher enables it by default. Text benchmark results do not measure image
performance. See [vision/README.md](vision/README.md) for the image test scope.

To keep using v4.4.1, follow its
[retained model card and instructions](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/b5c87a5f009fec986f94f2f6acd40f7c4046e591/README.md).
Its GGUF, matching runtime packages, original launcher, and checksums remain
available. The new model, source archive and launcher use separate filenames.

## Historical v4 benchmark record

Earlier 2026 builds of the retained v4 model recorded
HumanEval **160/164**, HumanEval+ **155/164**, ARC-Challenge
**1,143/1,172**, ToolEval Standard **115/138** and ToolEval Hard
**23/30**. These were separate local runs with earlier runtime/settings,
not reruns of the v5.0 model and not directly comparable to the small
thinking-off checks above. The [release history](HISTORY.md) preserves v4.1–v4.4.1 changes,
v3 comparisons, their caveats and immutable old instructions. Detailed
quality history is in [BENCHMARKS.md](BENCHMARKS.md#historical-published-results).

## Credits

Qwen produced [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
and its FP8 source. CIRU builds on
[llama.cpp/ggml](https://github.com/ggml-org/llama.cpp). The Strix Halo
runtime draws on the work of
[pwilkin](https://github.com/pwilkin/strix-halo) and the
[Halogen project](https://github.com/peonist-ai/halogen-flash-server).
Ported Q5 kernels and the expert prefill architecture use
[Gufo](https://github.com/gufo-org/gufo) work; relevant source notices
and its MIT terms are retained. The Qwen experimental/MTP work credits
Ryan Monsurate; the chunked GDN gate incorporates a
[HaloBox](https://github.com/halo-box/strix-llama.cpp) lead. AMD's
ROCm/TheRock projects provide the toolchain. See [NOTICE.md](NOTICE.md)
and the runtime source notices for provenance. This community release
is independent of Qwen and AMD.
