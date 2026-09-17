# v4.4 qualification

The exact `final-all-v1-pm4` inference binaries and pinned HIP/ROCr were qualified on a 128 GB Strix Halo / gfx1151, NixOS, stock TheRock10 compiler/math. MTP3, F16 target/draft caches, one slot, 262144 context, batch/microbatch 8192, native production thinking for HA20. These are runtime changes; no model artifacts were changed.

| Matched IU4 workload | v4.3 + S5/D0 control | v4.4 combined | Change |
| --- | ---: | ---: | ---: |
| 12960-token incident replay decode | 37.79 tok/s | 44.53 tok/s | +17.84% |
| Incident prefill | 824.28 tok/s | 889.13 tok/s | +7.87% |
| 245760-token cold decode | 7.62 tok/s | 21.14 tok/s | +177.48% |
| 245760-token prefill | 751.15 tok/s | 761.25 tok/s | +1.35% |
| Deep cold whole request | 394.58 s | 347.29 s | −11.98% |
| Cached append decode | 7.51 tok/s | 20.78 tok/s | +176.93% |
| Cached append whole request | 69.73 s | 25.84 s | −62.95% |
| Branch B revisit first piece | 5.653 s | 0.222 s | −96.08% |

One seven-case final block per arm. The control already includes S5/D0 and is not the plain published v4.3 runtime. No mirrored confidence interval or universal speedup is claimed. Comparable generated texts and draft counters matched; branch A2 is excluded from cross-arm speed claims because the control changed its canonical output. Candidate peak system RAM was 1.275 GiB higher and GTT 91.94 MiB higher.

Numerical evidence: 20 full-vocabulary teacher-forced rows exact, 11 lifecycle rows exact, 11 wide-index oracle cases passed. GDN specialization dispatch was confirmed. These checks qualify the complete IU4 composition; not every included patch has an isolated positive serving effect.

## IU4 HumanEval 0–9

Ten canonical prompts, one cold slot per task, greedy nonthinking native chat adapter, natural EOS, no output cap, seed 123. This is a speed panel, not a graded coding accuracy score. Allocated context is 262144; prompts are short.

| Setting | Pooled timed decode | Generated tokens | Total request time |
| --- | ---: | ---: | ---: |
| Default MTP3 | 60.351 tok/s | 1633 | 36.057 s |
| Opt-in Boost + MTP3 | 64.067 tok/s | 1633 | 33.864 s |

Boost improved pooled decode 6.16% in this one follow-up load. Nine outputs were byte-identical; one differed only by quote style and had the same parsed Python AST. Peak per-task Boost decode was 70.95 tok/s (HE3). This does not qualify long thinking, HA20, or Orca with Boost. Dataset SHA256: `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.

## IU4 Hermes Agent 20

One first trajectory per task, frozen 32-turn adapter, production native xhigh thinking, temperature 1, top-p .95, top-k 20, min-p 0, seed 160916, unlimited generation within context. Result: 19/20 full-score tasks; 98.5/100 arithmetic mean (matching the old card's aggregation), 99/100 canonical weighted score. HA17 scored 70 for the same delegation artifact issue seen in both older IU4 runs.

| IU4 version / run | Arithmetic mean /100 | Full-score tasks |
| --- | ---: | ---: |
| Published v4.3, pass 1 | 95.0 | 17/20 |
| Published v4.3, pass 2 | 98.5 | 19/20 |
| v4.4, one pass | 98.5 | 19/20 |

Same suite and arithmetic aggregation; historical engines/seeds differ. This comparison does not establish statistical equivalence, lower weight error or a causal quality gain. No new failed scenario appeared. Locked protocol SHA256: `6465f9465b78e9ca0b1456a8dfee2fd6df7e04681c7169dfefed9ebf9e025075`. HermesAgent-20 `57d7766bf3db8c40696e3ed937d43c8c85f4cd6c`; Hermes agent `ea74f61d983ebdfd6a863c45761d1b38081f1d08`.

## Capacity and limitations

A separate native target-only 524288-token prefill measured 604.651 tok/s (867.092 s), after an excluded 8192-token warmup, batch/microbatch16384, F16 cache. Minimum available RAM was 9.750 GiB. No 512K MTP, decode, or quality claim. New-build target-only decode was not measured.

Image+MTP returned the same pending-row error in both control and candidate. Target-only red-square/blue-circle fixtures passed. Keep vision target-only. Boost is optional and lacks a new HA20 run. Model weights/scales/PLE are unchanged, so weight reconstruction error is unchanged.

The shared package is released with IU4's completed qualification. Orca keeps its retained MTP4 default and receives a separate HE0–9 / HA20 release report after its requested checks. IU4 speed figures are not Orca measurements.
