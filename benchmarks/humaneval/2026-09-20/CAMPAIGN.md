# IU4 HumanEval speed tuning and full rerun

**Selected: IU4 v4.4.1, MTP6, Kairic Boost off, 4,096-token context on Ciru.**

The full 164-problem run scored **157/164 (95.73%) base and 154/164 (93.90%) HumanEval+**. Mean complete-request wall time was **5.07 s**, median **4.29 s**. Total HTTP inference time: **831.71 s (13.86 minutes)**. Pooled native decode: **66.46 tokens/s**.

## Draft and Boost screen

Same ten tasks, chat prompts, 1024-token cap, temperature 0, seed 0, thinking off, cold prompt cache, 256K allocation, F16 target/draft KV, and released v4.4.1 binaries. Each row is an independent load with one excluded streamed warmup. Scoring follows generation in a filesystem/network sandbox.

| Configuration | MTP depth | Boost | Sum of HTTP seconds | Decode tok/s | Base / plus |
|---|---:|---|---:|---:|---:|
| screen-d3 | 3 | off | 37.02 | 61.11 | 10/10 / 10/10 |
| screen-d4 | 4 | off | 35.09 | 64.71 | 10/10 / 10/10 |
| screen-d6 | 6 | off | 33.43 | 70.09 | 10/10 / 10/10 |
| screen-d8 | 8 | off | 34.32 | 67.01 | 10/10 / 10/10 |
| screen-d6-boost | 6 | on | 37.91 | 64.83 | 10/10 / 10/10 |

MTP8 was slower than MTP6 in the initial screen, so the predeclared condition for testing MTP12 was not met. Kairic Boost is available in this runtime: it combines ngram-mod with draft-mtp. Its valid MTP6 screen took 37.91 seconds versus 33.43 without Boost (+13.42% wall time). This single screen does not establish its effect across workloads.

An initial measurement-control error was corrected before selecting the winner: HE0 had been used for warmup, and source inspection showed Boost's n-gram table can retain generated text across requests. All initial Boost and confirmation runs were moved to `superseded-he0-warmup/` and excluded. Boost, confirmation, context and full runs were rerun with an unrelated affine-function warmup. Initial non-Boost depth screens were retained as directional evidence; independent confirmation uses the corrected warmup for both sides. Historical performance-store rows remain append-only and are explicitly excluded by the campaign exclusion manifest.

## Independent-load confirmation and context test

| Position | Allocated context | HTTP seconds | Decode tok/s | Generated tokens | Plus |
|---|---:|---:|---:|---:|---:|
| confirm-0-a1 | 262,144 | 37.16 | 61.99 | 1639 | 10/10 |
| confirm-0-c1 | 262,144 | 35.08 | 67.87 | 1642 | 10/10 |
| confirm-0-c2 | 262,144 | 34.34 | 68.64 | 1642 | 10/10 |
| confirm-0-a2 | 262,144 | 36.73 | 62.22 | 1639 | 10/10 |
| context-small-1 | 4,096 | 32.88 | 71.57 | 1642 | 10/10 |

Draft/Boost confirmation reductions against stock MTP3: 5.61%, 6.51%; pooled **6.06%**. Confirmed: **True**.

Bounded 4K-context smoke reduction against the latest confirmation with the same draft/Boost settings at 256K: 4.27%; observed **4.27%**. Lower context selected: **True**. At the user’s request, extra tuning rounds were omitted and this single context comparison selected the faster quality-passing configuration. It is not a precise or independently repeated context effect. All other requested settings stayed fixed; runtime limits may clamp effective batch capacity with the smaller allocation. Actual commands, slots and logs are retained. The small-context choice is for this workload, not a new general serving default.

Output lengths differ slightly across configurations. These are observed task-latency results, not fixed-output kernel speedups. Every repeated position is retained; no slow valid row was dropped. Exact per-task output hashes and raw text are available for checking output equivalence.

## Full-suite comparison

| Run | HumanEval | HumanEval+ | Mean wall seconds | Median wall seconds |
|---|---:|---:|---:|---:|
| **Tuned IU4, this run** | **95.73%** | **93.90%** | **5.07** | **4.29** |
| Earlier Orca MTP4, Ciru, 256K | 95.12% | 93.29% | 5.32 | 4.59 |
| cygnal Heretic2, published | 84.1% | 79.3% | 4.16 | 3.31 |

The earlier Orca run used the same exact messages, dataset version, evaluator, sampler and output cap, but a different model/draft configuration and potentially context. It had no explicit warmup and CPU grading ran alongside generation. This run uses an excluded warmup and scores afterward. Consequently the full-run difference is a deployment/workload comparison; the controlled HE0–9 pairs provide the tuning evidence.

The [author's Heretic2 table](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF) gives mean and median wall times but no exact harness, prompts, timing code or cache policy. Matching its disclosed temperature, thinking setting and cap does not establish identical execution conditions.

HE0–9 was used to select the configuration and is included in the standard 164-task score; this is not a completely held-out evaluation. The full run retains one fresh first answer per task, including **5 capped responses**. No answer repairs or retries. Base failures: HumanEval/32, HumanEval/116, HumanEval/129, HumanEval/132, HumanEval/140, HumanEval/145, HumanEval/147. Extended-suite failures: HumanEval/32, HumanEval/39, HumanEval/76, HumanEval/91, HumanEval/116, HumanEval/129, HumanEval/132, HumanEval/140, HumanEval/145, HumanEval/147.

## Evidence and disposition

- `PROTOCOL.md`, `identity.json`: predeclared sequence, user context amendment, measured hashes and scope.
- `candidate-results.csv`, `results.json`: valid measured loads and their native scoring results. Superseded measurements are retained separately and excluded from selection.
- `winner.json`: mirrored effects and selection decision.
- Each arm: exact argv, frozen requests, complete responses, per-task memory traces, slots/metrics, sandbox scores, and output hashes.
- `/home/crown/bench-results/llama/`: official append-only performance JSONL, SQLite and audit ledger; warmups labeled excluded.
- `restoration.json`: previous model-service active/enabled state restored; no model-selector changes.

Disposition: selected configuration advances only to this completed short-coding full-suite result. Other passing configurations are parked as measured alternatives, not rejected universally. Reopen depth/Boost selection for a materially different prompt length or acceptance regime. No persistent profile settings were changed.
