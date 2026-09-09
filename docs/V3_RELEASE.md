# CIRU Strix runtime v3

V3 transfers the portable Qwen3.8 Flash work from the retained hybrid checkpoint to the Strix-only runner. Target, PLE and MTP weights are unchanged. The base is the qualified v2.0.1 source at 9ea2390a71ae9f3d1cab519bbe099eb4ee06380e.

## Runtime changes

| Switch | Operation | Scope |
| --- | --- | --- |
| `GGML_QSA_PREFILL_WIDE=1` | Wider QSA prefill admission | Aligned tails and batches up to 2048 rows |
| `GGML_CUDA_Q41_MOE_FORCE_J_ALL_M=1` | Existing J32 Q4_1 expert tile | Supported prefill shapes; retains small-row path |
| `CIRU_QSA_RADIX_SELECT=1` | Parallel attention-cell selection | 2051 cells, up to 2048 rows, 32-lane HIP devices |
| `CIRU_QSA_INDEXED_DECODE=1` | Indexed attention | 1-8 query rows and the supported F16 KV layout |
| `CIRU_PLE_PREV_DIRECT=1` | Previous-token lookup | Exact position, sequence and uniqueness guards; legacy fallback |
| `CIRU_QSA_POOL_CACHE=1` | Derived history reuse | F32 pooled/normalized/rotated values; 64-block-aligned suffix refresh for 1-8 rows |
| `CIRU_QSA_TINY_GATHER=1` | Packed F32 gather | 1-8 columns, at least 4096 rows, one stream |
| `CIRU_MTP_ATTENTION_WINDOW=8192` | Optional draft window | Target verification retains full context; disabled by default |

The radix selector preserves the selection budget but changes equal-score tie choices and selected-list order. Model-level qualification is required; a TOP_K reference pass alone does not prove identical answers. The derived cache adds approximately 192 MiB at 256K relative to the old F16 block allocation.

Derived-cache validity includes sequence identity so independent KV streams with the same cell offset cannot reuse each other's derived values.

No external-GPU ownership, expert splitting, peer transfer, prefill staging, parked bulk pooling-cache prefill extension, or compact-mask experiment is included. The contiguous expert-ID fix from the hybrid checkpoint concerns that excluded expert-routing path.

## Qualification

Completed initial gates: 69 QSA state/guard cases with the pool flag on and off; 33 GPU TOP_K/GET_ROWS reference cases including ties, decode/prefill rows and gather admission boundaries; existing batch-allocation tests; 20/20 base and 20/20 plus tests on HumanEval 0-19; key recall with exact cached replay near 8K and 64K; 15,892,480 byte-identical F32 logits versus v2 in the existing four-prefix diagnostic.

These are bounded regression checks. They do not establish broad quality equality, full benchmark rankings, or full-context accuracy. The numerical diagnostic uses four 64-token prefixes, MTP off and context 512; the served quality checks use 262144 context capacity.

The final profile uses maximum MTP depth 6 and batch/microbatch 1024. Selected-profile quality, repeated 4K/64K speed, original-profile before/after and the 261888/128 capacity check are complete. See [absolute speeds, latency, memory and scope](qualification/v3.0.0/COMPARISON.md), [source identity](qualification/v3.0.0/source-identity.json) and [tested binary identity](qualification/v3.0.0/binary-identity.json).

## External comparator

The comparator is pristine halo-box/strix-llama.cpp at 5f851647fe5ed795dfd6c0a3fba543114879e874, using its recommended Vulkan backend, Unsloth UD-Q4_K_XL target, the published EasiiX Strix Q8 draft head and documented n-gram-on-disk support. Native KV, batch, thread, fitting and cache defaults are retained. A stock configuration screen tested MTP 2, 3, 4, 6 and adaptive 6; depth 3 was selected for final qualification. Halo source and model files were not modified.

The 2.79 GB Unsloth shared Q8 head intentionally omits tensors that a supporting runtime borrows from the main model. This pinned Halo loader instead requires `token_embd.weight`, and fails after the main model is loaded. The Unsloth self-contained Q8 head also fails for missing `output_hc_norm.weight`. Both attempts are preserved. The compatible EasiiX head is an existing published artifact, used without conversion.

Different weights and runtime profiles make the external test a serving-package comparison. The v2/v3 same-weight pair separately measures the runtime/profile change. See [the full comparison](qualification/v3.0.0/COMPARISON.md) for absolute prompt and generation rates, request latency, configuration screens and limitations.

Primary setup sources: [Halo README](https://github.com/halo-box/strix-llama.cpp/blob/5f851647fe5ed795dfd6c0a3fba543114879e874/README.md), [Unsloth MTP instructions](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/README.md), [model guide](https://unsloth.ai/docs/models/qwen3.8-next), [published EasiiX Strix head](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF/blob/6f7900648b1c6b14f067a182c640e47971e9ab35/README.md).
