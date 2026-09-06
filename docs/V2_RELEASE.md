# Runtime v2.0 qualification

The original released source is now available as the [GitHub `v2.0` tag](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0), with a [direct diff from v1.1.1](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/compare/v1.1.1...v2.0). [Source identity](PROVENANCE.md#v20-source-in-git) records its exact match to the published archive.

The released MTP profile requires the exported environment as well as draft flags. Its maximum depth 6 is workload-dependent; low acceptance can make a shallower draft faster. The controlled 42.3 tok/s coding result does not establish an optimal depth for other workloads, and the MTP-off sweep below does not compare draft depths. See [profile verification and depth selection](RUNNING.md#confirm-the-mtp-profile-and-choose-a-draft-depth).

## Three-way comparison

The three-way context sweep ran on **Sozo: Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151, 128 GiB shared memory, NixOS**, with one model workload at a time. Ciru handled the new CIRU BF16 captures and the clean Ubuntu build/GPU smoke. The comparisons preserve each arm's recorded execution settings; they are package comparisons, not a controlled kernel-only experiment.

| Arm | Target and runner | MTP implementation available in the package |
|---|---|---|
| CIRU v2.0 | Released IU4 weights; locked RC2 ROCm 10 runtime | Fixed maximum 6, p-min 0, 32,768-row draft shortlist; target F16 KV, draft Q8_0 KV |
| Agention / Laurent | [FP4 FAST model](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF/tree/ad4c5717254a630ee0c5a8db5208eb1f8476e56c); [Laurent's Vulkan fork](https://github.com/LaurentZuijdwijk/llama.cpp/commit/5e085d123eead2e89b5c19f824fccb05727da6a2) | Publisher adaptive 2–4 with its FP4 draft; target Q8_0 KV |
| Unsloth / recommended | [IQ4_XS model](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66); [Daniel Han Chen's MTP branch](https://github.com/danielhanchen/llama.cpp/commit/d1a92352cbd417fd840b4e765c0b82f5fe3d1d89), native Vulkan build | Unsloth [MTP README option 2](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/README.md), maximum 2 and shared Q8_0 draft; publisher defaults |

Only these three combinations are included. Exact model/shard hashes and runner commits are in [competitor artifacts](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/competitor-artifacts.json); complete recorded settings are in [runtime recipes](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/runtime-recipes.json).

### BF16 numerical fidelity

![BF16 fidelity diagnostic](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/resolve/v2.0/assets/v2-bf16-fidelity.png)

| Configuration | Mean forward KL ↓ | p95 KL ↓ | BF16 top-token agreement | Observed-token PPL ↓ |
|---|---:|---:|---:|---:|
| BF16 reference | 0 | 0 | 64/64 | 2.01799 |
| CIRU ROCm10 v2.0 | **0.03045** | 0.15453 | 61/64 | 2.08401 |
| Agention / Laurent | **0.11223** | 0.29765 | 60/64 | 2.38380 |
| Unsloth / recommended | **0.17457** | 0.55722 | 61/64 | 2.42931 |

This fixed diagnostic contains **64 full-vocabulary distributions and 60 observed next-token losses** across four short domain prefixes. MTP is off. CIRU uses F16 KV and flash-attention auto; the retained competitor captures use Q8_0 KV and flash attention on. CIRU's two independent model loads produced byte-identical logits.

The result measures these implementations' fidelity on a small shared panel. It does not establish general task-quality superiority or full-corpus perplexity. The old CIRU/Q5 diagnostic used an earlier runtime and is superseded for the current three-way comparison.

[Full statistics and hashes](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/bf16-fidelity.json) · [CSV](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/bf16-fidelity.csv)

### MTP-off context sweeps

![MTP-off context sweeps](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/resolve/v2.0/assets/v2-context-sweep.png)

Each cell is **prefill / generation tok/s**. All seven points use the same exact-token source fixture, a cold prompt, and 128 generated tokens.

| Prompt tokens | CIRU ROCm10 v2.0 | Agention / Laurent | Unsloth / recommended |
|---:|---:|---:|---:|
| 512 | 306.53 / 22.55 | 357.63 / 25.93 | 235.03 / 24.54 |
| 2,048 | 382.03 / 20.94 | 363.83 / 25.58 | 270.12 / 23.98 |
| 8,192 | 370.41 / 19.19 | 325.59 / 24.95 | 274.78 / 22.69 |
| 16,384 | 352.13 / 17.47 | 302.72 / 24.36 | 266.87 / 21.28 |
| 32,768 | 321.33 / 14.32 | 273.95 / 22.77 | 254.68 / 18.79 |
| 65,536 | 282.99 / 10.17 | 229.68 / 20.25 | 228.78 / 13.37 |
| 131,072 | 232.95 / 6.75 | 184.33 / 17.86 | 189.79 / 9.98 |

Server context is 262,144, sampling is greedy with seed 1234, EOS is ignored for the 128-token measurement, and a 512+32 warmup is excluded. Each row verifies exact prompt/output counts and zero drafted/accepted tokens.

CIRU leads prefill at 2K and above. **Both competitor arms have higher MTP-off generation rates across this sweep.** These are target-only context measurements, separate from the native MTP panel. Unsloth's target-only sweep uses Q8_0 KV, 16 threads, batch 2048/microbatch 512 and explicit CPU PLE placement with lazy mode off; its native MTP panel uses publisher defaults. CIRU uses F16 KV, 8 threads, and its 4 GiB PLE cache. See the recorded recipes for all differences.

[Full sweep CSV](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/mtp-off-context-sweep.csv) includes TTFP and memory. [Structured results](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/comparison.json) preserve idle/peak/delta RAM, GTT and VRAM counters. RAM is whole-system usage; these overlapping shared-memory counters must not be summed.



## What changed in v2.0

The locked candidate includes the retained Sozo quality-lane and Ciru speed-lane fixes:

- Correct attention descriptor strides, GPU admission for long QSA top-k, and restored-KV fast-path admission.
- Seven-column Q5_K weight reuse, the GPU MTP top-10 selector and a 32,768-row draft-output shortlist.
- Decode-entry and internal-microbatch synchronization repairs.
- Fresh HIP graph recapture, plus a launcher working-directory fix.
- The complete ROCm SDK installation helpers from v1.1.1, with the updated depth-6 production profile.

The previous allocator-lifetime and cached-prefix/MTP state fixes remain included. The adaptive runner and broad expert-reuse/fusion experiments are not enabled in this release. No weights were requantized or retrained.

The core source matches **`qwen38-ciru-rocm10-20260905-rc2`**. Portable build helpers are based on public v1.1.1 (`764ee491`); the retained core patch is based on v1.1 (`baba5e06`). [Provenance and runtime hashes](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v2.0/benchmarks/v2.0/ciru-provenance.json) record the exact inclusion evidence. The additional long-QSA/restored-KV admission code does not by itself establish a measured production cache-recovery speedup.

