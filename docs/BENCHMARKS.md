# Benchmarks and methodology

All results below were produced on a Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) system with 128 GiB unified memory and the CIRU Qwen3.8 runtime. The released weights are text-only.

## Quality

| Benchmark | Result | Coverage | Protocol note |
|---|---:|---:|---|
| HumanEval | **160/164 (97.561%)** | Full 164 | Local-custom chat, first sample, no retry/repair |
| HumanEval+ | **155/164 (94.512%)** | Full 164 | EvalPlus 0.1.10, first sample |
| ARC-Challenge | **1,143/1,172 (97.526%)** | Full 1,172 | EvalScope local run |
| ToolEval Standard | **115/138 points (83.33%)** | 69 cases | 53 pass, 9 partial, 7 fail, 0 runtime errors |
| ToolEval Hard | **23/30 points (76.67%)** | 15 cases | 10 pass, 3 partial, 2 fail, 0 runtime errors |
| GPQA-Diamond | **46/50 (92.0%)** | Sampled 50/198 | First/default-order local sample; not a full score |
| MMLU-Pro | **61/70 (87.143%)** | Sampled 70 | Five questions from each of 14 subjects |
| GSM8K | **97/100 (97.0%)** | Sampled 100 | Local EvalScope sample |
| IFEval | **92/100 (92.0%)** | Sampled 100 | Prompt-level strict; instruction strict was 94.17% |

### HumanEval protocol

- Full 164 tasks, one sample per task.
- Local custom chat protocol, temperature 0, no retries, repairs, or pass@k expansion.
- 2,048-token output cap.
- HumanEval and HumanEval+ results are functional test pass counts, but this is not represented as an external leaderboard submission.
- Weighted serving metrics across the run: 211.108 prefill tok/s, 31.531 generation tok/s.
- MTP depth-1 acceptance: 19,618/20,098 (97.612%).
- Zero API, validation, pager, or slot-erase failures.

### EvalScope protocol

ARC-Challenge is the complete 1,172-item dataset. GPQA-Diamond, MMLU-Pro, GSM8K, and IFEval are explicitly sampled local runs and must not be compared directly with full-suite leaderboard rows. The sampled suite used temperature 0, top-p 1, top-k 0, min-p 0, seed 42, reasoning disabled, first sample, and natural EOS.

### ToolEval limitations

All 84 local-custom cases completed without transport or runtime errors. The evaluator nevertheless identified material injection/safety misses in standard cases TC33 and TC34 and hard case TC81. The aggregate points should not be read as proof that the model is safe for unsupervised tool execution.

## H121 current-runtime performance

The release-fix row used an exact 8,192-token prompt and 128 generated tokens with MTP depth 3, F16 target KV, Q8_0 draft KV, 4 GiB PLE cache, graphs enabled, batch 2,048, microbatch 512, and one slot.

| Prompt / generation | Cold prefill | Generation | TTFP | MTP acceptance |
|---|---:|---:|---:|---:|
| 8,192 + 128 | **359.4267 tok/s** | **30.7970 tok/s** | 22.807 s | 84/128 (65.625%) |

The H121 stability stress generated 6,009 tokens before intentional termination with no HSA, pager, nonfinite, or server failure.

## Cold context ladder

These rows used the same weights and the pre-fix H111 runtime. H121 only corrects MTP continuation-buffer lifetime and does not change the prefill kernels, but only the matched 8K row above has been repeated on H121. Every row below used an uncached exact-count real-source prompt and 128 decode tokens.

| Prompt tokens | Prefill tok/s | Generation tok/s | TTFP |
|---:|---:|---:|---:|
| 512 | 298.3149 | 29.9648 | 1.729 s |
| 2,048 | 376.9801 | 24.3850 | 5.445 s |
| 4,096 | 378.9664 | 22.8458 | 10.824 s |
| 8,192 | 372.8099 | 32.1142 | 21.987 s |
| 16,384 | 295.0188 | 28.3220 | 55.549 s |
| 32,768 | 228.2670 | 23.2864 | 143.569 s |
| 65,536 | 174.6093 | 14.7909 | 375.350 s |
| 131,072 | 121.0477 | 11.3400 | 1,082.842 s |

The server is configured for 262,144 tokens, but the measured ladder ends at 131,072; this release does not claim a completed 262K benchmark row.

## Windows (WSL2 + ROCDXG) verification

The following rows were measured on a production Windows 11 host in August 2026: Windows 11 Pro 25H2 (build 26200), WSL 2.7.12 (kernel 6.18.33.2), Ubuntu 26.04.1, ROCm 10.0 (`amdgpu-install` 31.50, `--no-dkms`), `rocdxg-roct` 1.2.2, runtime tag `v1.1`, and `CONTEXT_SIZE=131072`. All rows used the same released weights and the public server profile (MTP depth 3, F16 target KV, Q8_0 draft KV, 4 GiB PLE cache, batch 2,048, microbatch 512, one slot) with uncached prompts and 128 generated tokens. The WSL2/ROCDXG path is experimental; treat these as verification numbers, not leaderboard claims.

| Prompt tokens | Prefill tok/s | Generation tok/s | TTFP | Wall | MTP acceptance |
|---:|---:|---:|---:|---:|---:|
| 564 | 86.8 | 21.4 | 6.50 s | 12.6 s | 64/188 (34.0%) |
| 2,100 | 380.1 | 19.7 | 5.53 s | 12.0 s | 59/199 (29.6%) |
| 8,244 | 456.9 | 21.3 | 18.04 s | 24.1 s | 67/179 (37.4%) |
| 16,436 | 371.5 | 17.7 | 44.24 s | 51.5 s | 61/197 (31.0%) |
| 32,820 | 291.0 | 14.4 | 112.78 s | 121.7 s | 56/213 (26.3%) |

Notes:

- The host's large GPU carve-out leaves a ~95.8 GiB GPU pool; the full
  262,144-token profile OOMs when loading the MTP draft on such machines.
  `CONTEXT_SIZE=131072` (28.1 GiB total KV) was used for this ladder.
- Generation held 20-21 tok/s up to 16K context and stayed usable at 32K,
  within ~35% of the native-Linux 30.8 tok/s headline despite the DXG
  translation layer.
- Prefill was prompt-cache-cold; the first request's context-load time is
  included in TTFP. See [BUILD_WINDOWS.md](BUILD_WINDOWS.md) for the full
  recipe and lifecycle notes (WSL 2.6+ idle regression and the
  `vmIdleTimeout=-1` + self-keeper fix, CONTEXT_SIZE guidance for
  carve-out machines).

## BF16 and Q5 comparison panel

This small diagnostic panel compared 64 full-vocabulary next-token distributions across four domains, with no MTP, F16 KV, and positions 48–63.

| Metric | CIRU IU4 | Size-matched Q5 comparator |
|---|---:|---:|
| BF16 top-1 agreement | **59/64 (92.1875%)** | **59/64 (92.1875%)** |
| BF16 top token present in top 5 | 63/64 | **64/64** |
| Forward KL, BF16 → arm | 0.223406 nats | **0.170264 nats** |
| Observed-token PPL, shared 60-token slice | **2.242104** | 2.341209 |
| PPL delta from BF16 2.017988 | **+11.106%** | +16.017% |

The CIRU model tied Q5 on top-token agreement and had lower observed-token perplexity on this tiny slice; Q5 had better aggregate full-distribution KL. This is a diagnostic panel, not a broad perplexity benchmark. The Q5 control also promoted 36 recurrent-convolution tensors to F32 for runtime compatibility.

## Cache disclosure

Headline PP/TTFP rows are deliberately cold and uncached. The public launcher enables prompt caching, an 8 GiB RAM cache, idle-slot caching, and context checkpoints for real service use. It does not carry over the evaluation harness's cache disables or slot erases.

Machine-readable values are in [`results/release-results.json`](../results/release-results.json).
