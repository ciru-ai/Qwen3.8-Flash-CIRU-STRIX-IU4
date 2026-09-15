# CIRU runtime v4.2.0

## Attention correctness fix for IU4 and Orca

V4.2 repairs an attention error introduced by the v4.0 graph/backend integration and inherited by v4.1. When the QSA3 fast path declined a batch, especially a small generation batch, the generic fallback ignored the selected-key IDs. It could attend outside the intended selection, and paths without a dense mask could include keys outside the causal prefix. This inconsistency is a concrete source of incorrect model computation associated with the reproduced repetition incident.

The corrected fallback honors selected keys, intersects an existing mask, returns zero for empty selections, and uses the existing float32 accumulation path for 256-wide attention on gfx1151. Scratch masks are processed in 64-query strips, bounded to 32 MiB at 256K context. QSA3 remains enabled; its kernel is unchanged. Disabling QSA3 alone did not restore correct selected-key semantics.

**Keep the existing model files.** IU4 and Orca reuse their current GGUF, PLE, matching Q8 MTP head and optional projector. This is a runtime update. Thinking duration and production reasoning/sampling settings are unchanged; long thinking by itself is not an error.

## Validation

- 48/48 independent CPU-reference attention checks passed with QSA3 on and off, including small batches, the 127/128/129-query boundary, strided inputs, unsorted IDs, existing masks and empty selections. Worst relative L2 error: 0.0238%, below the unchanged 0.1% incident gate.
- IU4 and Orca completed the same frozen medium-effort incident replay with valid tool calls and no detected repeated long line. This is a targeted regression check across both variants, not a broad new quality benchmark. The external reporter's exact prompt was unavailable.
- Full 262144 context, MTP6, F16 target/draft KV, batch/microbatch 8192, normal production graphs, IO32 and QSA3 enabled were retained in uninstrumented model checks.
- The current model-directory launcher's independent vision/MTP settings are carried into the versioned runtime: MTP stays on by default with vision; ENABLE_MTP=0 opts out. Prior image smoke checks used target-only generation. Vision with MTP has not been newly inference-qualified.

## Measured decode difference and confidence

On one 12,960-token IU4 prompt, a stock/corrected/corrected/stock sequence used four independent server loads, one excluded warm-up per load and equal 512-token measured outputs on gfx1151. Context was 262144, MTP depth 6, target/draft KV F16, QSA3 on, temperature 1.0, top-p 0.95, top-k 20, min-p 0 and seed 150915. EOS was ignored only for this equal-length throughput screen.

| Metric | v4.1 | v4.2 correction | Pooled difference |
| --- | ---: | ---: | ---: |
| Prompt processing | 954-976 tok/s | 984-990 tok/s | +2.3%; within the 5% parity band |
| Decode | 25.54-25.55 tok/s | 21.20-21.27 tok/s | -16.9% |

The mirrored decode comparisons were -17.04% and -16.71%, so the direction and size were repeatable for this workload. Only one prompt, one seed, two measured requests per build and one hardware/configuration were covered. This is not evidence of a universal 17% slowdown, an Orca speed result, or a formal confidence interval across workloads.

Draft acceptance changed from 364/877 to 329/1085. The corrected attention changes the generated stream and verification work, so the complete throughput difference cannot be assigned to the fallback kernel alone. The older engine performed incorrect attention and is not an equal-quality comparator. Historical v3/v4 benchmark tables retain their original versions and protocols.

**We are investigating new GPU kernels and profiling the corrected attention path to increase performance further.** Candidates must preserve attention correctness and demonstrate a gain in complete serving measurements. No additional speed improvement is promised by this release.

## Upgrade

Build the v4.2.0 tag or use the complete matching NixOS/gfx1151 binary archive with its recorded dependencies. Other Linux hosts should follow docs/BUILD_LINUX.md and build from source. Keep the executable, shared libraries, launchers, profile and UI together. The versioned launcher uses fresh v4.2.0 slot-state directories.

Original fast-prefill work: Halogen creator Peonist.ai (https://github.com/peonist-ai/halogen-flash-server), reproduced in open-source llama.cpp by pwilkin (https://pwilkin.github.io/strix-halo/). CIRU maintains this package's compatibility integration and correction. Existing model, runtime and third-party licenses and credits remain in place.
