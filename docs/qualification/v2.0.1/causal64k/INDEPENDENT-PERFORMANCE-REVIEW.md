# Independent performance disposition — 2026-09-07

**Recommendation: the accumulated evidence supports a scoped QSA v2.0.1 release, conditional on the clean Ubuntu GPU launcher smoke and final package checks. No additional 64K run is warranted merely to repair the failed profiler attachment.** This is a release judgment from the combined evidence, not a claim that the original Sozo slowdown has been explained.

The new same-host Ciru pair passes the locked practical performance gate. I independently parsed both raw SSE responses: all 128 generated token IDs and their recorded native timings agree with the summary. The complete request payloads are identical by SHA-256. Both runs evaluated 65,536 uncached prompt tokens, drafted 225 tokens and accepted 89. Target graph reuse was 39 in each run (3 after warmup), with no logged HIP/CUDA/assert errors. The official store reports two verified rows and no problems; cleanup restored the original inactive service state and left no server running.

| Real serving measurement | Released baseline | QSA candidate | Candidate change |
|---|---:|---:|---:|
| Prefill tok/s | 284.1161 | 290.1413 | +2.1206% |
| Generation tok/s | 17.1228 | 17.0459 | −0.4488% |
| Prefill duration | 230,666.227 ms | 225,876.185 ms | −4,790.042 ms |
| Generation duration | 7,417.020 ms | 7,450.458 ms | +33.438 ms |

These are one candidate-first/baseline pair on Ciru with production 262,144 context allocation, one slot, MTP6 and shortlist 32,768. They are not averages across machines or a demonstrated general speedup.

## Telemetry review

The 0.25-second collector captured real GPU clock, power and temperature, plus CPU, pressure and process I/O. The table uses interior phase samples with 0.5-second boundary margins; the prefill/generation boundary is inferred from the request window and native generation duration. Generation has 25/26 samples, so small differences are descriptive rather than causal attribution.

| Mean sensor reading | Baseline prefill | Candidate prefill | Baseline generation | Candidate generation |
|---|---:|---:|---:|---:|
| GPU MHz | 2,833.54 | 2,838.66 | 2,666.77 | 2,654.80 |
| GPU power W | 106.49 | 106.45 | 105.25 | 105.61 |
| GPU temperature °C | 69.14 | 66.95 | 67.15 | 66.16 |
| GPU busy % | 97.69 | 97.90 | 84.04 | 83.56 |

- Both prefill clock medians were 2,849 MHz. Final-quarter means were 2,829.67 and 2,831.13 MHz; there is no sustained late-prefill clock collapse in these samples. Baseline was modestly warmer, so this is not an exact thermal-state match or proof of zero thermal influence.
- Candidate generation clock was approximately 0.449% lower, similar to its 0.449% lower generation rate. This is **consistent with clock variation**, not proof that clock variation caused the difference. No hardware-counter or kernel trace establishes causality.
- Zero process major faults and zero system swap-in/out were observed in both trimmed phases. Prefill process reads were identical at 47,194,112 bytes. Memory full-pressure time was below 0.002%; CPU some-pressure was about 0.49%. Main-thread runqueue waits were 36.6/38.9 ms across approximately 225/230 seconds of prefill. No material scheduling or I/O anomaly invalidates the pair.
- From existing per-2,048-token progress logs, candidate prefill was faster in every compared segment. Beyond the first 8K segment, segment duration reductions were roughly 1.6–2.0%; the original Sozo pattern of increasing late-prefill slowdown did not recur here.

## What remains unestablished

Both ROCProfiler attachments failed because the server was not launched with `ROCP_TOOL_ATTACH=1`. This happened **after clean timing and before any cached diagnostic request**. The receipts and explicit error logs are preserved. There is no usable kernel/API execution breakdown, no cached 16-token result, and no traced explanation of the older Sozo observation. The failed attachment does not invalidate already completed clean timing or independent system telemetry.

The old Sozo pair remains a real observed PP−2.61% / TG−6.86% result with its cause unresolved. A passing Ciru pair must not overwrite it, relabel its failed gate, or prove that Sozo was thermally throttled. The two machines' measurements should be presented separately.

## Why release is reasonable within scope

This latest pair exercises the actual affected 65,536-token serving state and full MTP path, returning identical tokens/work while staying well inside the 2% generation-loss gate. It supplements the passing short MTP comparison, later 8K confirmation, 66 mapping/state cases, 288 canonical-input comparisons, full-model sequence-copy/foreign-content/simultaneous-batch controls, 16 served recall requests and GPU boundary checks. The focused CPU/normalization/synthetic 64K diagnostics independently found no material added cost in their measured paths.

Taken together, this is enough to release the isolated correctness repair with an explicit performance qualification scope. It does not establish every model task, filled 512K operation, every machine condition or concurrent MTP. Keep multi-slot MTP unsupported, HIP host-buffer #25863 excluded, and clean Ubuntu compatibility distinct from NixOS performance evidence. Root retains the final release decision after the pending build/smoke/package checks.

Structured statistics, raw-response identities and phase-analysis method are in `INDEPENDENT-PERFORMANCE-REVIEW.json`; the analysis script is retained at `../../analyze-final-perf.py`.
