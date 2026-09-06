# QSA backport: draft status

This branch contains a focused backport of upstream [#27941](https://github.com/ggml-org/llama.cpp/pull/27941), commit `36b10154383b60eb15baac2c7a40d2a5f784faa7`, for [#27994](https://github.com/ggml-org/llama.cpp/issues/27994). It also adds a guarded canonical-stream implementation that preserves the general path's QSA inputs while reducing CPU work.

**Not released: the revised candidate still needs performance qualification.** The model weights, HIP code, MTP policy and sampling defaults are unchanged. `CIRU_RELEASE.json` continues to describe the last published v2.0 release; it is not this candidate's build manifest.

## Completed checks

- Clean CPU source build; 66 mapping/state cases and existing batch-allocation test pass.
- All four QSA input tensors match the sequence-aware reference by SHA-256 across 288 canonical-layout comparisons, including physical offsets, incomplete tails, two streams and per-block/per-cell biases.
- Full CIRU model, 524288 total context, two slots, MTP off: split-KV sequence copying is fixed and returns exact reference logits. State blobs round-trip exactly.
- With identical physical cache layouts, computing a second conversation does not change any of the first conversation's 16 full-vocabulary output vectors. The released build fails this test.
- Actual two-sequence decode batches: changing the other conversation's entire prompt and suffix leaves the first conversation's logits exact, in both split and unified KV. The released build fails the unified-KV case.
- 16/16 served private-code recall requests pass: eight split and eight unified, with 13062-token initial prompts, simultaneous requests, followups, slot reuse, prefix caching and streamed/nonstreamed responses. No GPU errors.
- GPU normalization passes at 65535, 65536 and 131072 pooled rows, including the original launch-dimension boundary.

Different physical cache layouts can still produce different floating-point results. The original solo-versus-shared numerical failure is retained in the evidence. A tensor trace locates its first difference in the existing flash-attention calculation; a separately specified equal-layout control distinguishes that behavior from foreign-conversation interference. These tests do not validate a filled 512K context or every agent workload.

## Performance gate

The initial unoptimized backport preserved exact generated tokens but failed the long-context throughput screen: pooled prefill losses were 3.40% at 8K and 4.16% at 64K. Its 64K decode timing was noisy. That candidate was rejected for release.

The revised candidate's guarded canonical path passes the correctness checks above. Its full-model speed has **not** yet been measured. Remaining work is a fresh four-load MTP comparison at 262144 configured context (57 prompt / 520 output tokens), followed by a balanced cold 8K/64K prompt comparison (128 output tokens). Both use the unchanged official served benchmark harness and exact frozen requests. The gate is matching output tokens, less than 2% throughput loss, and no unresolved timing-order effect; a contested result may require one balanced confirmation block.

The Sozo test processes have stopped and all CPU governors are restored to powersave. No further GPU run is queued. Testing was paused when the user needed the machine for other work. Expected remaining machine time is approximately 25 minutes for the base performance block, longer only if a defined failure needs investigation.

The MTP shortlist still supports one slot. Two-slot operation requires `ENABLE_MTP=0 PARALLEL_SLOTS=2`. The separate HIP host-buffer proposal #25863 remains excluded.
