# QSA sequence-isolation fix: v2.0.1 qualification

Qualification completed on 2026-09-07. Correctness and scoped performance checks passed. The clean Ubuntu build and GPU smoke passed, including exact reference output and MTP counts.

v2.0.1 fixes the QSA conversation-isolation defect reported in [upstream #27994](https://github.com/ggml-org/llama.cpp/issues/27994). It includes the relevant sequence grouping, indexer copying, saved-position metadata and normalization-shape changes from Daniel Han's [#27941](https://github.com/ggml-org/llama.cpp/pull/27941), commit `36b10154383b60eb15baac2c7a40d2a5f784faa7`. Unrelated upstream metadata validation and tensor-parallel policy changes are outside this backport.

QSA now pools cache cells by both sequence set and position bucket, so a second conversation with overlapping token positions cannot replace the first conversation's indexer blocks. Pending stream copies also update the raw indexer keys. A guarded path for one contiguous sequence emits the same QSA inputs with less host work. Shared prefixes, fragmented caches, multiple sequences and repeated spatial positions use the general sequence-aware path.

Weights, HIP kernel sources, sampling defaults, MTP depth and the shortlist policy are unchanged. The retained NixOS HIP binary is unchanged; the clean Ubuntu build has its own binary identities. Multi-slot MTP is still unsupported: use `ENABLE_MTP=0 PARALLEL_SLOTS=2` for two slots.

## Correctness

| Check | Result and scope |
| --- | --- |
| Clean CPU source build | Passed; 66 mapping/state cases and the existing batch-allocation test passed |
| Original runtime mapping reproducer | v2.0 failed 16/16 overlapping-sequence cases; the backport passed all |
| Guarded path versus general reference | All four QSA input tensors matched by SHA-256 across 288 cases, including offsets, tails, two streams and both bias formats |
| Full-model cache copy and restore | Split-KV copying fixed; all 16 reference logit vectors exact; state roundtrips byte-exact in split and unified modes |
| Equal-layout foreign-content control | v2.0 failed; v2.0.1 returned byte-identical full-vocabulary logits for all 16 steps |
| Simultaneous two-sequence decode batches | Changing the second conversation's complete prompt and suffix left the first conversation's 16 full-vocabulary vectors byte-identical in split and unified KV; v2.0 failed unified KV |
| Actual served recall | 16/16 requests passed: eight split, eight unified; 13062-token initial prompts with the private code near the beginning, simultaneous requests, followups, slot reuse, prefix caching, streamed and nonstreamed responses |
| GPU normalization boundary | Passed 65535, 65536 and 131072 pooled rows on ROCm0; all finite, maximum absolute error 3.61e-7 against a double reference |

Full-model concurrency tests used 524288 configured total context, two slots, MTP disabled, F16 target KV and a 4096 MiB PLE cache. Both served slots reported a 262144-token limit. These are bounded isolation and recall tests, not a filled-512K run or a complete agent-workload qualification.

Different physical cache layouts can still produce different floating-point results. The original solo-versus-shared gate failed for both builds and is retained in the evidence. A tensor trace located the first difference in the existing flash-attention calculation when the physical cell window grew from 6400 to 10496; inputs and logical selected cells matched at that point. The separately specified equal-layout control isolates foreign-conversation content from that layout effect. We do not claim bitwise invariance across different physical layouts.

## Performance

Serving comparisons of the final guarded candidate used the retained v2.0 RC2 baseline and the same QSA candidate library, ROCm 10.0.0, 262144 configured context, one slot, F16 target KV, Q8_0 draft KV, maximum draft depth 6, shortlist 32768 and performance CPU governors. Target, draft and PLE identities were held fixed. Sozo and Ciru are separate Strix Halo gfx1151 hosts with Ryzen AI MAX+ 395, Radeon 8060S and 128 GiB shared memory, running NixOS. Comparisons are paired within each host; results from different machines are not pooled.

The short served MTP check on Sozo used four fresh loads (A1/C1/C2/A2), a frozen 57-token coding request and 520 greedy output tokens. Every output token and draft/acceptance count matched. Geometric pooled generation throughput changed by **+0.10%**, passing the prespecified less-than-2% loss and 3-percentage-point mirror-gap screen.

Two subsequent confirmations used exactly one fresh candidate load and one fresh baseline load each, with the same 512-token prompt / 16-output warmup excluded from timing. Both used cold exact-token prompts and 128 greedy output tokens:

| Host / date | Actual prompt tokens | Baseline prefill / generation tok/s | Candidate prefill / generation tok/s | Prefill change | Generation change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sozo / 2026-09-06 | 8192 | 370.5067 / 38.5802 | 387.9168 / 39.0444 | +4.6990% | +1.2032% |
| Ciru / 2026-09-07 | 65536 | 284.1161 / 17.1228 | 290.1413 / 17.0459 | +2.1206% | -0.4488% |

Both confirmations passed the exact-output/work-count and less-than-2% throughput-loss gates. The Ciru pair returned the same 128 tokens, with 225 drafted / 89 accepted tokens and 39 target graph reuses in each arm. The official recorder retained raw SSE, request/settings, token IDs, memory samples, JSONL/SQLite records and verified hash ledgers. These are bounded regression checks, not general speedup claims.

Ciru telemetry sampled GPU clock, power and temperature, CPU scheduling, memory pressure and process I/O every 0.25 seconds. Candidate/baseline mean GPU clocks were 2838.66/2833.54 MHz during prefill and 2654.80/2666.77 MHz during generation; mean powers were 106.45/106.49 W and 105.61/105.25 W respectively. No sustained late-prefill clock collapse, process major faults, swapping or material CPU/I/O anomaly was observed. The candidate's approximately 0.45% lower generation clock is consistent with its approximately 0.45% lower rate, but this does not establish causality.

**Profiler limitation:** ROCProfiler attachment was attempted only after each clean timing completed. Both attempts failed because the server had not been started with `ROCP_TOOL_ATTACH=1`. No cached diagnostic request ran and no kernel/API trace was obtained. The failed-attachment receipts are retained. Clean timing and system telemetry remain valid; there is no traced explanation of the earlier Sozo slowdown.

### Earlier Sozo result retained

The earlier candidate-fast comparison completed one baseline/patched pair at each length below. A duplicated follow-up pair was cancelled and its partial records are retained:

| Work | Prefill change | Generation change | Output tokens |
| --- | ---: | ---: | --- |
| 8192 prompt / 128 output | -1.21% | -1.73% | Exact |
| 65536 prompt / 128 output | -2.61% | -6.86% | Exact |

That 64K pair failed the 2% loss screen and remains unexplained. An earlier general-path-only block also showed the unchanged baseline's 64K generation speed move from 16.56 to 15.40 tok/s. Clock and temperature traces were not captured for those runs. The later passing Ciru pair does not erase the failed Sozo gate, prove thermal throttling or establish parity under every machine condition.

### Focused execution-cost checks

A separate diagnostic used one patched and one baseline load with a synthetic 65536-cell cache and real full-model 1-, 7- and 512-token batches. The target context remained 262144, F16 KV and one slot; MTP was off. Attention/indexer values were zero-filled, sequence metadata was contiguous and recurrent state came from one token. Cache setup was excluded; one warmup and three small samples were taken per shape. These are operation costs, not real-conversation or served MTP rates.

| Tokens in batch at 64K cache size | Median execution-time change (lower is faster) | Full-vocabulary output |
| --- | ---: | --- |
| 1 | +0.13% | Exact |
| 7 | -0.15% | Exact |
| 512 | -1.81% | Exact |

All three cases passed their exact-logit and less-than-2% execution-cost gates. CPU mapping was faster in all 12 cases, with 24-58% less time for the 64K cases. The isolated GPU normalization comparison returned byte-identical outputs and added less than one microsecond per call at the 64K shape.

**Qualification scope:** the sequence-isolation/copy/boundary checks, served recall, short MTP performance, later real 8K and 64K confirmations, and focused execution-cost checks support this isolated correctness release. The old Sozo slowdown and causal attribution remain unresolved. There is no filled-512K, multi-slot MTP, cross-layout bitwise invariance or new full task-quality-suite claim.

The first general-path-only candidate was rejected: tokens matched, but pooled prefill losses were 3.40% at 8K and 4.16% at 64K, with noisy 64K decode timing. Its results remain in the evidence alongside the guarded candidate.

## Build identity and evidence

The NixOS candidate replaced only `libllama.so.0.3.0` in the retained runtime. Three affected translation units were rebuilt, with the guarded helper rebuilt afterward; the other 15 binaries were unchanged. The same candidate library was hash-verified on both Sozo and Ciru. The corresponding source also passed the earlier clean CPU build.

A separate clean Ubuntu 24.04 / ROCm 10.0.0 build completed on Dunamis's Intel i9-14900KF in an isolated container with no GPU access and no inherited build objects. All 3566 input files were verified unchanged after compilation. `llama-server`, `llama-cli` and `llama-bench` built and passed ELF dependency/help checks. The existing tests passed: **66 QSA mapping/state cases** and **30 batch-allocation tests / 198 assertions**.

The documented setup/build helper was used. Because the build machine was Intel, CPU compilation was explicitly configured with `GGML_NATIVE=OFF`, AVX2/F16C/FMA/BMI2 enabled and AVX512/VNNI disabled before the final build; this portability adjustment is recorded. No inference source changed.

The resulting Ubuntu binaries were transferred by verified hashes to an Ubuntu 24.04 container on Ciru, using its NixOS host GPU driver and the matching ROCm 10 SDK. The 57-token coding prompt / 520-token greedy completion matched **all 520 reference token IDs**, with **674 drafted / 400 accepted**, at the production 262144 context allocation. **GPU device identity and all 15 binary hashes were verified.** This is a short active-prompt compatibility check, not an Ubuntu performance comparison or native-Ubuntu-host qualification.

The first smoke verifier reported `gfx1151 identification not found` because it required that literal string in a verbosity-3 server log. The same built binary identifies ROCm0 as Radeon 8060S; the sole GPU KFD node reports `gfx_target_version=110501`, and the actual inference log records RDNA3.5 kernel execution. The reviewed result is PASS. The original verifier failure and raw evidence remain preserved; inference was not repeated to obtain that result.

- NixOS candidate libllama SHA-256: `4a70e051c953617fe6a13d60444073dd2a5e7be1a9f707896841156939d3992c`.
- Retained NixOS HIP library SHA-256: `f60bfe62b7dfab548c5a30dcf7728f47b85f142c39ca9ae7377cc340b71c19d4`.
- Clean Ubuntu libllama SHA-256: `7d4f28ed7578e242a744b27e12d7a2a1d9256a1dbe6bb97a91b9b362be23066e`.
- Clean Ubuntu HIP library SHA-256: `7955750301284227e36a558195fc4830ec1a521fed19e412159f3ac56a9629a9`.
- [Final qualification decision](qualification/v2.0.1/FINAL-QUALIFICATION.json).
- [Later Sozo 8K confirmation](qualification/v2.0.1/short-context-confirmation/).
- [Ciru 64K timing, telemetry and independent review](qualification/v2.0.1/causal64k/).
- [Clean Ubuntu build identities and CPU tests](qualification/v2.0.1/ubuntu24-build/).
- [Reviewed Ubuntu GPU smoke receipt](qualification/v2.0.1/ubuntu24-gpu-smoke/REVIEWED-RESULT.json).
- [All structured summaries and build/link identities](qualification/v2.0.1/).
- [Raw performance records, frozen requests and diagnostic fixtures](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/download/v2.0.1/qsa-v2.0.1-evidence.tar.gz).

Build records distinguish hash-verified retained objects from one vendor-hash static archive omitted from the old sealed index; that archive was captured from its original link path and its identity is recorded. A retained executable's version string alone does not identify this library update. Use the tag/source manifest and library hashes.

The separate HIP host-buffer proposal [#25863](https://github.com/ggml-org/llama.cpp/pull/25863) remains excluded after an unresolved MTP-output difference. It is not the fix for #27994. See [parallel validation history](PARALLEL_VALIDATION.md).
