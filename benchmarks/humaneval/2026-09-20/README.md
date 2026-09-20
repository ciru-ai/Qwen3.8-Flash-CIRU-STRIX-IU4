# HumanEval proof: IU4 and Orca on Ciru

**IU4 v4.4.1, MTP6, Kairic Boost off, 4096 context:** 157/164 HumanEval (95.73%), 154/164 HumanEval+ (93.90%), **66.46 decode tokens/s**, **5.07 s mean / 4.29 s median per problem**. Total measured HTTP time: 831.71 s. Run date: 2026-09-20, Ryzen AI Max+ 395 / gfx1151, 128 GB unified memory, NixOS, ROCm 10.

| Full 164-task run | HumanEval | HumanEval+ | Mean seconds | Median seconds |
|---|---:|---:|---:|---:|
| IU4, MTP6, Boost off, 4K | 95.73% | 93.90% | 5.07 | 4.29 |
| Earlier Orca, MTP4, 256K | 95.12% | 93.29% | 5.32 | 4.59 |
| cygnal Heretic2, published | 84.1% | 79.3% | 4.16 | 3.31 |

The published comparison comes from [cygnal's model card](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF/blob/b7a350f3ba4c29962cefe9e69af225f142cda9bc/README.md) and [Reddit post](https://www.reddit.com/r/StrixHalo/comments/1wl3weh/i_benchmarked_12_quantizations_of_qwen38flashnext/). This is a local protocol matching the disclosed thinking, temperature and output cap. The author has not supplied the exact prompts, raw answers, evaluator version or timing implementation. These are different complete configurations, not an isolated quantization comparison. Orca's earlier run also had no explicit warmup and graded concurrently; IU4 warmed up once and graded afterward.

## What was measured

HumanEvalPlus v0.1.10, EvalPlus 0.3.1, Python 3.11.14. One fresh answer per task in order, thinking off, temperature 0, seed 0, cap 1024, no repair or retry. Request-level prompt caching is disabled. All 164 responses were verified to contain zero cached prompt tokens and zero reasoning characters. Five capped answers remain in the score and all five failed. Native EvalPlus sanitization and base/extended tests were run in a filesystem/network sandbox after generation.

HE0-9 selected the settings and is included in the standard full score. MTP6 reduced matched smoke wall time by 6.06% versus MTP3. A subsequent single 4K smoke took 32.88 s versus 34.34 s at 256K, a 4.27% reduction; both passed 10/10. This bounded context smoke is not a precision estimate. The longest full-suite prompt was 455 tokens, so every prompt plus the 1024-token allowance fits in 4K.

An initial control error used HE0 for warmup, potentially priming Boost's persistent n-gram table. Those Boost/confirmation runs are retained under `superseded-he0-warmup` and excluded. Corrected Boost and confirmation used an unrelated affine-function warmup; the full run did too. Boost did not win the corrected screen. See [the full campaign](CAMPAIGN.md) and [frozen protocol and amendments](PROTOCOL.md).

## Why 66 tok/s can coexist with a slower task time

Our 66.46 tok/s is pooled native **MTP decode throughput on this HumanEval run**. The author's 24-33 tok/s is a separate **serial-decode throughput statement**. They are not matched fixed-output measurements.

Our average answer had **260.20 output tokens**. The measured mean breaks down as:

| Component | Seconds per task |
|---|---:|
| Prompt processing | 1.045 |
| Decode | 3.915 |
| Other request overhead | 0.111 |
| Complete HTTP request | 5.071 |

The complete HTTP output rate is 51.31 tok/s. Faster generation per token does not guarantee a shorter answer or a faster whole request. At 24-33 tok/s, generating our 260-token mean answer alone would require 7.9-10.8 seconds. Thus the author's 4.16-second task mean cannot describe that same token workload at that rate. His output lengths and per-task timing records are needed to identify the difference; this does not establish an error in either result. [Machine-readable breakdown](timing-breakdown.json).

## Audit the evidence

- [Per-task timings and scores](per-task.csv), [all 164 requests, responses and native score records](answers.jsonl).
- [Summary](summary.json), [selection decision](winner.json), [candidate times](candidate-results.csv), [verification](verification.json), [model/runtime hashes](identity.json).
- [Complete evidence archive](evidence.tar.gz): raw IU4 and Orca outputs, commands, scorer, original generation harnesses, memory traces, frozen dataset, runtime profile, model/runtime identities and excluded runs. `evidence/SHA256SUMS.json` covers every archived source file. Operational service snapshots and credentials are not included.
- [Source-method comparison and limits](SOURCE-METHODS.md).

## Reproduce

The measured runtime source is [2de109eea3bc94b566f1c3f23bae806f8045f5af](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/2de109eea3bc94b566f1c3f23bae806f8045f5af), with the matching complete v4.4.1 runtime and [IU4 model package](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4). Match the hashes, MTP head and required PLE sidecar. Use the [Linux build instructions](../../../docs/BUILD_LINUX.md); a different build or hardware can change timings.

From that runtime checkout, with the model and complete runtime installed:

```sh
MODEL_DIR=/path/to/IU4 RUNTIME_DIR=/path/to/runtime \
  CONTEXT_SIZE=4096 MTP_DEPTH=6 KAIRIC_BOOST=0 ENABLE_MTP=1 \
  PORT=18090 SLOT_DIR=/path/to/fresh-slot-state \
  scripts/ciru/run-server.sh --chat-template-kwargs '{"enable_thinking":false}'
```

The exact executed argv is in `evidence/iu4/full-fastest/command.json`; environment and the frozen production profile are also archived. Request parameters override server sampler/cache defaults. Use one slot, F16 target/draft KV, eight threads and the saved kernel settings. Start a fresh server and do not run scoring or other GPU work alongside generation.

Extract the evidence archive and run the standalone input replay helper:

```sh
tar -xzf evidence.tar.gz
python3 replay.py --url http://127.0.0.1:18090 \
  --warmup evidence/iu4/full-fastest/warmup-prompt.txt --out my-iu4-run
```

`replay.py` submits only the stored requests, not the stored answers. It performs one identical excluded warmup, measures complete HTTP responses and does not retry. `--limit 10` selects the smoke subset. This convenience helper uses the same timed request path but omits the original host memory sampler and append-only lab store. The exact original harness and measurement module are archived separately.

For grading, use the archived `score.py` with EvalPlus 0.3.1 and the frozen dataset. `score-sandbox.py` records the exact Bubblewrap mounts and environment from the measured NixOS run; remap its Python/Nix paths for another installation. Keep generated code inside a network/filesystem-isolated environment. Native settings are `fast_check=False`, minimum time limit 1 second, ground-truth time factor 4. Score all first answers, including failures and length stops. Do not present a generation-only replay as a validated pass@1 result.

Dataset and EvalPlus credit: [EvalPlus](https://github.com/evalplus/evalplus/tree/v0.3.1), Apache-2.0 ([license](EVALPLUS-LICENSE)); HumanEval originates from [OpenAI HumanEval](https://github.com/openai/human-eval), MIT. Model weights are not redistributed here. Wall times can vary with clocks, thermals and system load; the recorded numbers apply to the retained run.
