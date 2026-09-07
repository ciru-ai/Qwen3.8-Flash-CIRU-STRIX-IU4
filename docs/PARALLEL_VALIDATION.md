# Parallel serving validation - 2026-09-07

## Prepared release: v2.0.1

The QSA fix is included in the prepared v2.0.1 source. Full-model tests reproduce the released v2.0 unified-KV interference and split-KV indexer-copy defects; v2.0.1 passes the corresponding exact-logit controls. It also passes 16/16 served recall requests, with 13062-token initial prompts, across separate and unified KV. See the [complete qualification and performance report](QSA_BACKPORT_STATUS.md) and [copyable two-slot instructions](RUNNING.md#parallel-requests-and-unified-kv-cache).

MTP with the released shortlist still requires one slot. The community's assertion log (`n_slots = 2`, `n_ctx_slot = 100096`, `kv_unified = false`) identifies this separate configuration limitation. Use `ENABLE_MTP=0 PARALLEL_SLOTS=2` for two slots. The launcher catches unsupported multi-slot MTP before loading the model.

## Historical v2.0 smoke and issue identification

The initial v2.0 short-marker smoke below passed, but it did not validate recall from a conversation's earlier context after another request joined. It did not rule out [#27994](https://github.com/ggml-org/llama.cpp/issues/27994). The subsequently supplied [#27941](https://github.com/ggml-org/llama.cpp/pull/27941) identified the missing sequence-aware QSA grouping and indexer update. Those fixes are included in v2.0.1; the original v2.0 tag and archive remain unchanged.

[Original structured smoke results and identities](parallel-validation-20260906.json)

| Test | Result | Scope |
| --- | --- | --- |
| Full CIRU, released runtime, separate KV | 8/8 requests passed | `-c 524288 -np 2`, MTP off |
| Full CIRU, released runtime, unified KV | 8/8 requests passed | Same settings; fresh load |
| Qwen3.5-4B, released and proposed HIP runtimes | 64/64 requests passed | 16 requests per runtime/KV combination, `-c 16384 -np 2`, MTP off |
| Proposed HIP backport, backend capability probe | Passed | Direct host compute disabled, pinned allocation retained |
| Proposed HIP backport, CIRU MTP regression | Held | Different greedy continuation in both independent candidate loads |
| Launcher configuration tests | Passed | Defaults, environment/CLI overrides, early rejection, target-only launch without draft files |

Hardware was Strix Halo gfx1151 with 128 GiB shared memory, NixOS and ROCm 10. Full-model requests used a 6659-token reference fixture and short prompts, greedy decoding, thinking disabled, concurrent and queued requests, streaming and nonstreaming responses, and prefix caching. Each response contained its requested marker and no other request's marker. No transport or GPU errors were observed. Both slots reported a 262144-token context. This did not fill the 524288-token allocation or run a full agent workload.

## Why the HIP patch is held

We tested the two-commit proposed upstream [#25863](https://github.com/ggml-org/llama.cpp/pull/25863), headed by `ce82541acbaf5c532c0727d6ccb6de2b0b0c948d`. Only its HIP translation unit and shared library changed; the other 15 runtime binaries and all weights were held fixed. It disables an unsafe direct host-buffer compute path reported in [#25992](https://github.com/ggml-org/llama.cpp/issues/25992), while preserving pinned staging buffers. Our baseline did not reproduce that report in the bounded concurrency checks, so those passing checks do not establish that the reported corruption is fixed on CIRU/ROCm 10.

The four-load A1/C1/C2/A2 MTP comparison used the released 262144-token configured context, one slot, depth 6, the 32768-row shortlist, and the retained 57-token coding prompt with 520 greedy output tokens. Baseline loads agreed with each other; candidate loads agreed with each other but first differed from baseline at generated token 147. A target-only follow-up agreed with baseline through 200 tokens, including in the candidate runtime. Explicit GPU placement of the input embeddings did not restore the baseline output.

This is an unresolved numerical or execution difference, not an established task-quality loss. It fails the predeclared exact-output regression gate, so the candidate remains unqualified for promotion. Timing rows from different continuations do not establish a speedup. The original logs, token IDs, request JSON, build identities and official performance-store records are retained for investigation. No new Ubuntu qualification or filled-context stability claim is made for the candidate.

The older fragmented unified-KV/SWA fix [#23981](https://github.com/ggml-org/llama.cpp/pull/23981), commit `236531595584fdb5f63f09bf4306cf982b757e6e`, is already an ancestor of v2.0. It is distinct from the QSA issue corrected in v2.0.1.

## Launcher regression check

```bash
python3 tests/test-ciru-launcher.py
bash -n scripts/ciru/run-server.sh
```

These use a stub executable and temporary model-file placeholders; no GPU or model download is required.
