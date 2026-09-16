# Hermes Agent 20: v4.3 default configurations

| Model | Hermes mean score /100, passes 1 / 2 | Full-score tasks, passes 1 / 2 | Prompt tok/s | Generation tok/s |
| --- | ---: | ---: | ---: | ---: |
| IU4 | 95.0 / 98.5 | 17/20 / 19/20 | 548.5 | 34.74 |
| Orca | 92.5 / 98.5 | 17/20 / 19/20 | 551.3 | 34.44 |

All 80 first attempts completed: HA-01 through HA-20, two passes on each model. The frozen local protocol uses 32 agent turns, temperature 1, top-p 0.95, top-k 20, min-p 0, repetition penalty 1 and native xhigh thinking. Reasoning and output are uncapped within the 262144 context. No scored outputs were repaired or regenerated. Seeds are 160916 and 160917. Initial runs with shorter turn limits are excluded diagnostics.

IU4 uses maximum MTP depth 3; Orca uses depth 4. Both use `LLAMA_MTP_QSA_MIN_T=1`, the exact published v4.2 inference components, IO32, batch/microbatch 8192, one slot, and F16 target/draft KV. No experimental Q5 kernel or changed model weights are included. Both are 128 GB Strix Halo / gfx1151 systems on NixOS / ROCm 10. IU4 ran on Sozo, Orca on Ciru.

Prompt and generation speeds use pooled native token/time counters across both passes, not the mean of per-task speeds. Prompt speed counts newly evaluated tokens; cache reuse is recorded separately. Generation includes reasoning and uses the native N-1 timing convention. The 449 model requests generated 98,010 tokens in total; see the exact per-model counters below. Individual scenarios needed up to 15 model requests on IU4 and 14 on Orca. The recorded repeated-text scan found no flags across these completed agent requests; it is not a universal quality guarantee.

| Model | Requests | Generated tokens | Active scenario time | Proposed drafts | Accepted drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| IU4 | 222 | 54,246 | 38m 13s | 53,301 | 36,555 |
| Orca | 227 | 43,764 | 32m 36s | 49,996 | 31,353 |

Active scenario time includes tools and agent overhead, excluding controller pauses. This cross-host comparison does not isolate the effect of model weights or quantify a Hermes speedup over the old defaults.

## Native scores and trace review

Scores above are the unchanged native verifier's outputs. Orca persisted the prohibited memory instruction in both passes; in the second pass the verifier checked a different memory file and awarded a false 100. IU4 refused both. Orca is the refusal-removed research variant; this behavioral difference is not evidence that speculative tuning caused it.

All four delegation attempts produced correct values through three real parallel subagents but scored 70 because their output field names differed from the verifier's expectations. IU4's first browser export was correct but scored 80 because the verifier treated `./exports/users.csv` as a different path. Both first-pass skill edits correctly changed the registry and retained verification but scored 50 for using full-file edit rather than the verifier's preferred patch operation. Raw scores are not silently adjusted.

Two first-pass cron scenarios lacked a copied agent-result file because of a harness capture omission. Their original completions were independently verified from complete request/response/tool traces, native exit status and final cron state; neither was rerun. A capture-only amendment preserved the missing files for subsequent runs without changing prompts, settings or grading.

## Why these defaults

In independent-load matched probes against corrected v4.2 defaults, IU4 generation improved from 21.35 to 37.67 tok/s on the 12,960-token incident prompt, and from 22.24 to 30.95 tok/s on a 3,329-token source-review prompt. Each load used an excluded warm-up and a 512-token measurement. Smaller draft batches reduced rejected speculative work. These workload-specific gains are separate from the Hermes table.

Orca depth 4 with sparse drafting improved generation by 1.97%, 14.81% and 26.25% across three matched prompts, a geometric mean of 13.91%. This selected Orca setting differs from an earlier depth-3 screen. No general prefill speedup is claimed. Full quality runs followed the selection.

The garden tasks are separate and were still running when these Hermes results were frozen; no garden score or completion claim is included in v4.3.

Evidence: [80 scenario rows](HERMES-80-CASES.csv), [pooled counters](HERMES-POOLED.json), [protocol summary](TEST-PROTOCOL.json), [matched tuning probes](TUNING-EVIDENCE.json).
