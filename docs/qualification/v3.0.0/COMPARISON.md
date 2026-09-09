# CIRU Strix runtime v3.0.0 qualification

V3 combines the qualified QSA fixes from the prior READY package with the portable prefill and decode improvements from the retained hybrid checkpoint. Target, PLE and MTP weights are unchanged. The general v3 profile retains MTP 6 and uses b1024/u1024; Halo uses its own best screened stock profile, MTP 3 on Vulkan.

## Measured serving performance

Ciru: Ryzen AI Max+ 395, gfx1151, 128 GB shared memory, NixOS. One model workload ran at a time. These cold coding requests use exactly matching chat-formatted input token IDs, 128 generated tokens, one slot and 262144 context capacity. EOS is honored; every reported request produced all 128 tokens. The sampler is nonthinking, temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, repeat penalty 1, frequency penalty 0 and seed 123.

| Input tokens | Profile | Prompt tok/s | Generation tok/s | First streamed piece (s) | Whole request (s) | Rows |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | Previous CIRU | 392.00 | 22.52 | 10.70 | 16.34 | 2 |
| 4,096 | CIRU v3 | 455.65 | 24.60 | 9.25 | 14.41 | 6 |
| 4,096 | Halo | 381.49 | 35.30 | 11.09 | 14.69 | 4 |
| 65,536 | Previous CIRU | 284.49 | 13.33 | 230.46 | 239.99 | 1 |
| 65,536 | CIRU v3 | 369.81 | 24.22 | 177.32 | 182.57 | 3 |
| 65,536 | Halo | 263.42 | 23.28 | 248.91 | 254.37 | 2 |

| Input tokens | Comparison | Prompt rate change | Generation rate change | Request time reduction |
| ---: | --- | ---: | ---: | ---: |
| 4,096 | v3 vs Previous CIRU | +16.24% | +9.25% | +11.80% |
| 65,536 | v3 vs Previous CIRU | +29.99% | +81.66% | +23.93% |
| 4,096 | v3 vs Halo | +19.44% | -30.32% | +1.93% |
| 65,536 | v3 vs Halo | +40.39% | +4.04% | +28.23% |

Prompt processing and whole-request latency are distinct from generation speed. The table exposes each result; an overall request-time advantage is not a claim that CIRU wins every generation-rate measurement. The fresh MTP 6 confirmation is close to Halo at 4K, so small aggregate short-request differences should not be treated as a robust default-profile win. The qualified MTP 2 option provides the clearer short-request latency benefit on this fixture; the long-context improvement is much larger. Different CIRU/Halo weights and runtime profiles make the external result a serving-package comparison, not a runtime-only or quantization-quality comparison. The previous/current CIRU pair uses identical model weights but each release's own profile, so it measures the combined runtime/profile update.

The table includes three clean v3 MTP 6 loads: the initial C1/C2 pair and a fresh final confirmation. Halo MTP 3 has two selected-profile loads. Between the initial and final v3 runs, the stock MTP configuration screens, selected Halo A1, previous CIRU and optional v3 MTP 2 checks ran. The fresh v3 MTP 6 confirmation was followed by Halo A2 and CIRU full capacity. Each load has one excluded 512/16 warmup. Each speed position has 4K/128, 64K/128, then a second 4K/128 request. A1, previous CIRU and C1 then run quality. No build, weight download or profiling runs alongside measured serving. This is a small repeated experiment, not a confidence interval or every-workload ranking.

Aggregated PP divides total prompt tokens by total native prompt time. TG follows the native N-1 convention: total 127 timed output tokens per request divided by total native generation time. Latencies are arithmetic means from the official API harness; first-piece time is its first streamed content-field event. Loading, warmups, quality request speeds and instrumented requests are excluded. Raw row values, ranges, MTP counts and exact input hashes are in comparison.json.

## Configuration selection and execution evidence

The following are single 4K/128 selection screens using the published sampler, except the batch screen, which used greedy sampling. They selected finalists; the repeated final measurements above carry the release claim.

| Runtime | Maximum draft depth | Prompt tok/s | Generation tok/s | Drafted / accepted |
| --- | ---: | ---: | ---: | ---: |
| CIRU | 2 | 454.94 | 28.88 | 112 / 71 |
| CIRU | 3 | 451.18 | 27.94 | 159 / 73 |
| CIRU | 4 | 436.93 | 26.04 | 200 / 77 |
| CIRU | 6, initial repeated profile | about 459 | about 24.7 | see initial rows |
| Halo | 2, initial repeated profile | about 383 | about 31.4 | see initial rows |
| Halo | 3 | 383.66 | 35.37 | 132 / 82 |
| Halo | 4 | 381.41 | 30.04 | 165 / 85 |
| Halo | 6 | 376.20 | 25.91 | 214 / 90 |
| Halo | 6, native adaptive | 379.69 | 29.31 | 145 / 83 |

Higher maximum depth was tested on Halo without source changes. Its native maximum 3 won this screen; using the largest supported value would have made this workload slower. The CIRU batch/microbatch 512, 1024, 1536, 2048 screen measured PP 408.31, 451.19, 449.25, 390.17 tok/s respectively. 1024 was selected over the close 1536 result for the smaller workspace and qualified at 64K and full capacity. The MTP 2 setting is retained as an option for low-acceptance long requests, rather than replacing the general MTP 6 default. The high-acceptance coding counterexample below prevented that default change. These selections do not establish a universal optimum.

A native ROCm fresh-launch trace confirmed radix TOP_K, indexed decode attention and tiny F32 gather dispatches, alongside the device-resident MTP chain. The trace captured 204 radix-selector, 132 indexed-attention and 132 tiny-gather dispatches; [kernel-path evidence](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v3.0.0/docs/qualification/v3.0.0/kernel-path-summary.json) records the counts and diagnostic timing scope. The trace's served rates are excluded. A focused extension of existing Q5_K weight reuse to three verification columns passed 17 actual GPU reference cases but measured 27.47 TG versus 28.88 for the existing path. It was reverted and parked. The release binaries match the pre-candidate hashes; no benefit from this rejected change is claimed.

The earlier Halo MTP 2 mirrored 64K runs varied from 13.28 to 20.61 TG despite identical output IDs and 111 drafted/71 accepted tokens. PP was 265.42 and 264.01. The cause was not established; those results are retained as preliminary evidence and are not substituted for the selected MTP 3 comparison.

## Quality and correctness

| Profile | HumanEval base | EvalPlus extended tests | Two-key recall, original + cached | Exact cached token replay |
| --- | ---: | ---: | --- | --- |
| Previous CIRU | 20/20 | 20/20 | 4/4 at ~8K and ~64K | Both lengths |
| CIRU v3 | 20/20 | 20/20 | 4/4 at ~8K and ~64K | Both lengths |
| Halo | 20/20 | 20/20 | 4/4 at ~8K and ~64K | Both lengths |

## MTP depth depends on the workload

The same 20 short coding requests provide a counterexample to promoting MTP 2 as a universal default. Their generation work has high draft acceptance. The previous CIRU and v3 MTP 6 runs produced identical token IDs on all 20 tasks; MTP 2 and Halo have different output lengths, shown below. These are diagnostic native compute rates over the bounded quality panel, separate from the fixed-output speed experiment.

| Profile | Generated tokens | Prompt tok/s | Generation tok/s | Sum of request times (s) |
| --- | ---: | ---: | ---: | ---: |
| Previous CIRU | 3179 | 148.53 | 53.33 | 75.49 |
| CIRU v3 MTP 6 | 3179 | 219.51 | 53.24 | 70.67 |
| CIRU v3 MTP 2 | 3212 | 226.25 | 39.63 | 91.57 |
| Halo MTP 3 | 3241 | 176.10 | 49.48 | 79.15 |

MTP 6 retains high-acceptance coding throughput while v3 reduces prompt time. MTP 2 improves the lower-acceptance 4K/64K coding fixture but is slower on these short tasks, so it is an explicit option instead of the general default. Halo MTP 3 is the selected-profile result here; this table is not a claim that it is the optimal Halo depth on every short coding task.

| Input tokens | Optional CIRU MTP 2 prompt tok/s | Generation tok/s | Whole request (s) |
| ---: | ---: | ---: | ---: |
| 4,096 | 453.09 | 29.39 | 13.62 |
| 65,536 | 373.08 | 24.88 | 180.87 |

The panel is canonical HumanEval tasks 0–19, EvalPlus v0.1.10 base and extended tests, first sample only, no retries, 4096-token cap and truncations counted as failure. Generated code is tested inside a filesystem/network sandbox. This bounded nonthinking coding/recall panel does not establish broad model equality, thinking-mode quality, benchmark rankings, tool-use reliability or filled-context accuracy.

The final inference sources passed 69 QSA mapping/state/guard cases with flags on and off, 33 ROCm TOP_K/GET_ROWS CPU-reference cases and 30 batch allocator tests with 198 assertions. Cases include ties, selection/gather admission boundaries and independent sequences with equal cell offsets. An earlier test fixture requested unsupported partial KV copying; the corrected full-copy fixture passes, and the failed attempt is preserved.

The four-prefix same-weight numerical diagnostic matched 15,892,480 F32 logits byte-for-byte against the previous runner. It uses four 64-token prefixes, 64 full-vocabulary vectors, context 512, one-token microbatch and MTP off. It does not exercise long-prefill ordering. Radix selection can change threshold-tie membership and selected-list order; general bitwise equivalence is not claimed.

## Full-capacity serving

The final CIRU profile completed 261,888 real input tokens plus 128 generated tokens within 262,144 capacity: **257.44 prompt tok/s, 18.00 generation tok/s, 1017.38 s first piece and 1024.44 s whole request**. It was a cold request with EOS honored. This verifies serving capacity on this host, not full-context task accuracy. No filled 256K Halo comparison is claimed.

## Included runtime and prior package

The base `9ea2390a71ae9f3d1cab519bbe099eb4ee06380e` already includes the READY package's QSA isolation/copy repairs. V3 adds wider aligned QSA prefill, J32 expert tiling on supported prefill shapes, parallel radix cell selection, indexed F16 decode attention, guarded direct PLE previous-token lookup, F32 derived-history reuse and packed tiny F32 gathers. Derived-cache validity includes sequence identity. Each new environment switch accepts literal 0. Draft-window code is included but disabled and unqualified when enabled.

External-GPU ownership, peer transfer, hot/cold expert placement, scheduler overlap, compact masks and the parked bulk-prefill pooling extension are excluded. The checkpoint's contiguous expert-ID fix belongs to that excluded hybrid routing path. The derived F32 cache adds 192 MiB at 256K over the prior F16 allocation: 12 layers × 128 values × 65,536 blocks × 2 extra bytes. The width 128 is verified from the actual GGUF indexer-normalization tensors.

The original READY.md, package verification, independent package review, source/evidence archives and their checksums are preserved in the prior-package archive and historical qualification directory. v2.0.1 was qualified locally but not separately published as a tag; its fixes ship in v3.0.0. Existing public v2.0 tags, weights and weight checksums remain unchanged.

A clean initial same-profile, same-weight greedy 64K/128 pair measured previous CIRU 290.34 PP / 13.95 TG and v3 341.57 PP / 25.50 TG (+17.64% / +82.73%). All 128 output IDs and 225 drafted/89 accepted counts matched. This one-sequence pair predates the sequence-key guard and uses b2048/u512 and MTP 6; it is separate from the final release-profile comparison.

The initial 4K raw-record continuation ended after four text tokens; the then-enabled ignore_eos forced repeated end tokens. That unsuitable speed row is preserved and excluded. All reported final speed inputs use a real coding request and honor EOS. Initial quality runs overlapping CPU build/download activity and the profiler rates are also excluded from clean performance claims.

Isolated 262144-column TOP_K measured 12,062.33 us to 1,792.08 us for 4 rows (6.73x), and 288,363.00 us to 48,854.72 us for 1536 rows (5.90x), versus heap selection with warp scan. Both use the 16-node graph duplication cap. These are kernel measurements, not served speed multipliers.

## Halo compatibility and settings

Halo source is pristine commit `5f851647fe5ed795dfd6c0a3fba543114879e874`, using its recommended Vulkan backend. The target is Unsloth UD-Q4_K_XL at revision `38bb39ee97821de2c9009abb7e93950eec396e66`, all four shards verified against LFS hashes. It uses full GPU offload, documented --ngram-on-disk, selected maximum MTP 3, and native KV, batch, thread, fitting and cache defaults. Only supported command-line MTP settings were screened.

Unsloth's 2.79 GB shared Q8 head intentionally omits tensors a supporting runtime borrows from its main model. The pinned Halo loader requires token_embd.weight and fails even after loading the main model. Unsloth's self-contained head also fails for missing output_hc_norm.weight. The comparison uses the already published EasiiX Strix Q8 head at revision `6f7900648b1c6b14f067a182c640e47971e9ab35`, SHA256 `9db03a687670608286e99b563fcc86d0ee76c8dd863f64b2afc0b54eb0eb975d`. No Halo source, weights or adapters were modified. Both failed compatibility attempts are preserved.

CIRU retains its 79.40 GB target, 52.43 GB PLE payload and 4.14 GB Q8 draft, ROCm 10, F16 target KV, Q8 draft KV, 32,768-row draft shortlist, eight CPU threads, one slot and full target verification. The general maximum MTP depth is 6; MTP_DEPTH=2 selects the tested long-request option. The optional draft attention window remains off.

Setup sources: [Halo README](https://github.com/halo-box/strix-llama.cpp/blob/5f851647fe5ed795dfd6c0a3fba543114879e874/README.md), [Unsloth guide](https://unsloth.ai/docs/models/qwen3.8-next), [Unsloth MTP card](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/README.md), [EasiiX head](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF/blob/6f7900648b1c6b14f067a182c640e47971e9ab35/README.md).

## Memory, platform and reproducibility

| Input tokens | Profile | Mean RAM before request (GiB) | Peak system RAM (GiB) | Max RAM increase (GiB) | Peak GTT (GiB) | Peak VRAM (GiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | Previous CIRU | 100.73 | 103.43 | 2.15 | 88.58 | 0.52 |
| 4,096 | CIRU v3 | 103.06 | 106.02 | 2.43 | 90.29 | 0.53 |
| 4,096 | Halo | 102.43 | 104.88 | 2.77 | 93.08 | 1.85 |
| 65,536 | Previous CIRU | 100.59 | 101.32 | 0.72 | 88.58 | 0.52 |
| 65,536 | CIRU v3 | 102.91 | 103.66 | 0.74 | 90.29 | 0.52 |
| 65,536 | Halo | 103.14 | 103.72 | 0.60 | 93.17 | 1.99 |

RAM, GTT and VRAM overlap on this unified-memory APU and must not be added. These are sampled host/device telemetry values, not model-file sizes.

New GPU qualification is NixOS/ROCm 10/gfx1151. Historical clean Ubuntu qualification belongs to v2.0.1; no new v3 Ubuntu result is claimed. Source build instructions are retained. The optional binary payload is the tested NixOS server, bench, focused tests and shared libraries; it requires the recorded Nix store and SDK paths. No tested CLI binary is claimed. Vision and simultaneous multi-slot MTP remain outside this qualification.

Raw requests, SSE, output IDs, samplers, memory samples, model/build identities, quality scoring artifacts, failures, official speed row hashes and configuration screens are preserved in the evidence archive. Source archives are checked against the release Git tree including modes and symlinks. The prior READY package is included separately with its original checksums. The main service remains in its original inactive, enabled, unmasked state.
