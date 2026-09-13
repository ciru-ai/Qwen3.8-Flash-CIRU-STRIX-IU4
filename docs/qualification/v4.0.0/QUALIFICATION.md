# CIRU runtime v4.0.0 qualification

V4 corrects the release designation previously published as v3.1.0 and expands attribution. Inference source, executable/shared-library bytes, model artifacts and all measured results are unchanged. Launcher slot-directory/version labels and the UI build-version metadata use v4.0.0; the original experiment records retain their original identities.

Credit for the original fast-prefill breakthrough goes to Halogen's creator, [Peonist.ai (`peonist-ai`)](https://github.com/peonist-ai/halogen-flash-server). [pwilkin](https://pwilkin.github.io/strix-halo/) reproduced that performance in an open-source llama.cpp implementation and published the kernels and configuration that CIRU adapted for these existing weights. CIRU's contribution is the compatibility integration and validation.

The pwilkin reference engine is integrated with the existing CIRU IU4 model, CIRUPLE1 pager and Q8_0 MTP artifact. Model and projector bytes are unchanged. This release replaces the earlier engine with reference commit `f5daaa3cfa6358e5dd398911ec741813745a5440` plus the recorded CIRUPLE1, typed Q4_1 expert, MTP state and external-draft parameter adaptations. The build candidate name `ciru-reference-release-r2` is an identifier, not a Git commit.

## Measured result

On Ciru (Ryzen AI Max+ 395, Radeon 8060S/gfx1151, 128 GB UMA, NixOS, stock TheRock ROCm 10), the integrated own-weight server measured approximately **992–1,001 prompt tokens/s** on five cold 30.8K-token coding requests. A cold 65,295-token recall request measured **947.89 PP**. These are actual served requests with MTP enabled, not the different reference model's measurements.

| Own-weight request | Input tokens | PP tok/s | Generation tok/s | First piece s | Whole request s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed tokens + 128 output | 4,096 | 631.04 | 14.61 | 6.49 | 15.19 |
| Cold key recall | 7,951 | 973.71 | 26.23 | 8.25 | 8.82 |
| Coding HE0 | 30,886 | 999.85 | 35.13 | 31.00 | 35.90 |
| Coding HE1 | 30,878 | 996.68 | 39.32 | 31.00 | 37.18 |
| Coding HE2 | 30,847 | 991.76 | 40.80 | 31.12 | 33.41 |
| Coding HE3 | 30,881 | 1,000.75 | 36.91 | 30.88 | 32.26 |
| Coding HE4 | 30,880 | 1,000.81 | 37.20 | 30.87 | 35.55 |
| Cold key recall | 65,295 | 947.89 | 32.56 | 69.00 | 69.77 |
| Fixed tokens + 128 output | 16,384 | 1,003.33 | 14.43 | 16.45 | 25.25 |
| Capacity, fixed tokens + 128 output | 261,888 | 748.38 | 6.22 | 349.97 | 370.39 |

These are one observation per request from a diagnostic run with `LLAMA_TRACE=1` and verbosity 4, not a clean interleaved comparison. The production launcher omits tracing. Cached requests are excluded from this PP table. Generation speed varies substantially by content and speculative acceptance; a universal decode improvement is not claimed. The three short coding requests measured 51.21–52.93 generation tok/s. Full-context generation is much slower. Timings include the actual sampler and output-length differences recorded in the raw requests.

A separate target-only native pilot with the same own weights measured **959.51 PP / 24.27 TG** at PP16384/TG128, depth 0, batch/microbatch 16384, 16 threads, one repetition. The earlier qualified CIRU core measured 631.47 PP; the approximately 52% difference is directional across runs, not a matched repeated serving comparison. The final serving batch is 8192, not the native pilot's 16384.

The complete unmodified reference package, on its own different weights, measured **1,361.01 ± 10.93 PP** (three repetitions); the same engine/weights with the published 36 kernel variables unset measured **641.09 ± 1.16 PP**. That 2.12× within-reference result motivated adoption. It does not establish that our quantization causes the remaining gap. The author's custom HIP/ROCr runtime was not needed to exceed 1000 PP and was not built in this campaign. Additional decode gains from it remain untested.

## Correctness and serving scope

- The GPU numerical panel passed **22/22** cases, including typed Q4_1 routed expert, GLU, BF16 down/reduction chains and retained IQ4 regressions. Actual chain dispatch was observed. R2 changes only the CPU external-draft parameter handoff after R1; GPU sources and numerical evidence carry forward explicitly.
- The complete own-weight serving run finished all **15 requests**, with **5/5 key-recall checks**, **3/3 cache/replay checks**, and MTP 1,269 accepted tokens out of 1,960 drafted tokens. Minimum available memory was **6.82 GiB**. Prefixes without a usable MTP state checkpoint safely reprocess; immediate identical replays reused cache.
- The original **standalone-module** HumanEval 0–7 first-sample score is **7/8 base and extended tests**. HE3 omitted `from typing import List`, which was supplied in its user prompt. The original failure and `finite_panel_pass=false` are retained.
- A separately labeled **prompt-provided-import** rescore of all eight unchanged answers passed **8/8 base and extended tests**. It prepended only the original prompt's top-level imports uniformly across all cases. No answer was regenerated or patched selectively. This secondary score does not replace the standalone 7/8 result. Five prompts were padded to about 30.8K; three were short. Evaluation used EvalPlus v0.1.10 in the existing isolated sandbox. This bounded nonthinking panel does not establish broad quality, full-suite accuracy, thinking quality or tool reliability.
- The exact capacity request completed **261,888 input tokens plus 128 output**, cache 0, without truncation, at context 262144. Minimum available memory was **7.70 GiB**. This establishes capacity, not accuracy across the full context.
- The relocated exact binary/library bundle passed HTTP UI asset hashes and an OpenAI-compatible chat request. The unchanged published projector passed a red-square and a blue-circle image check with MTP disabled. These are functional smoke checks, not a broad vision evaluation.

## Shipping configuration and retained failures

Text defaults: context 262144, batch/microbatch 8192, one slot, F16 target and draft KV, MTP 6, prompt RAM cache 1024 MiB, PLE cache 4096 MiB, 32 checkpoints at an 8192-token minimum step. The 36 published reference kernel variables are enabled with type/shape admission intact. The Q4_1 compatibility code does not enable IQ4-only kernels for incompatible weight types or create whole-expert BF16 weight shadows.

R1 inherited target PLE settings into the separate draft and failed to load it; R2 clears PLE settings only for an external draft. R2 with Q8 draft KV hit the fused QSA F16 requirement. F16 draft plus batch 16384 crossed the 4 GiB memory floor. Batch 8192 with a 4096 MiB prompt cache completed 14 requests but crossed the floor on the final 16K request. The final 1024 MiB prompt cache completed the whole panel and capacity check. All failed runs are retained and excluded from successful performance claims.

Vision mode explicitly disables MTP because image-position handling has not been qualified with the new MTP checkpoint implementation. The source review identified a possible image/text pending-position discontinuity; an MTP-on vision runtime failure is not claimed. Multi-slot MTP is rejected by the launcher. Old v3 slot saves are preserved but not automatically restored. External UI assets ship with the release. The tested Nix binary has HTTPS fetching disabled; the old v3 Nix build also failed OpenSSL detection. Use image data URLs or build with OpenSSL development dependencies.

## Evidence and lineage

The release assets include source and file manifests, exact tested binary identities, raw R1 and R2 evidence, reference control/quality evidence, and this report. The source archive is verified against the release Git tree including executable modes and symlinks. The binary archive contains the same files used by the relocated UI/chat/vision checks. Original benchmark store rows 4416–4433 and separate strict/import-context quality rows are preserved. Historical v3 reports remain historical and are not inherited as new-engine qualification.

Credit: [pwilkin's Strix Halo work](https://pwilkin.github.io/strix-halo/), Qwen, ggml-org/llama.cpp, AMD/ROCm, Ryan Monsurate's MTP work, Daniel Han's QSA work and all retained source contributors. Existing license and component notices remain in the source tree.
