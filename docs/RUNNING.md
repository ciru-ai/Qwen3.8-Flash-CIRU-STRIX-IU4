# Running in production

## 1. Download and verify the complete package

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  --revision v2.0 --local-dir ./model
cd model
sha256sum -c checksums.sha256
cd ..
```

The primary GGUF and the complete `ple/` directory are mandatory. The MTP draft is required for the headline generation profile.

## 2. Launch the audited public profile

From the runtime repository:

```bash
BUILD_DIR="$PWD/build-gfx1151-sdk" \
  MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
```

The setup helper uses `build-gfx1151-sdk/`; for a manual SDK build, select `build-gfx1151/` instead. Keep the SDK directory used at build time in place. The script expands to this release profile (shown with the manual build directory):

```bash
export GGML_CUDA_Q41_MOE_FORCE_J=32
export GGML_QWEN4EXP_PLE_WORKERS=16
export GGML_QWEN4EXP_PLE_STRICT_SHA=0
export ROCBLAS_USE_HIPBLASLT=1
export GGML_QSA_LONG_TOPK=1
export GGML_QSA_RESTORE_FAST=1
export CIRU_MTP_TOPK10=1
export CIRU_MTP_SHORTLIST=32768
unset GGML_HIP_GRAPH_EXEC_UPDATE CIRU_MTP_GPU_CONFIDENCE CIRU_MTP_GPU_ADAPTIVE CIRU_MTP_GPU_CONF_MIN CIRU_MOE_EXPERT_REUSE CIRU_MTP_TRACE CIRU_MTP_CONF_TRACE LD_PRELOAD

./build-gfx1151/bin/llama-server \
  --model /absolute/path/to/model/Qwen3.8-Flash-CIRU-STRIX-IU4.gguf \
  --alias Qwen3.8-Flash-CIRU-STRIX-IU4 \
  --host 127.0.0.1 --port 8080 --jinja \
  --ple-sidecar /absolute/path/to/model/ple \
  --ple-cache-mib 4096 \
  --slot-save-path ./slot-state \
  -ngl all -sm none --fit off \
  -c 262144 -b 2048 -ub 512 --parallel 1 \
  -t 8 -tb 8 -ctk f16 -ctv f16 -fa on \
  --cont-batching \
  --cache-prompt --cache-ram 8192 --cache-idle-slots \
  --ctx-checkpoints 32 --checkpoint-min-step 8192 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --metrics --slots \
  --spec-type draft-mtp \
  --spec-draft-model /absolute/path/to/model/mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf \
  --spec-draft-ngl all --spec-draft-device ROCm0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --spec-draft-threads 8 --spec-draft-threads-batch 8 \
  --spec-draft-n-max 6 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 --spec-draft-p-split 0.10
```

`GGML_QWEN4EXP_PLE_STRICT_SHA=0` avoids hashing the 52.4 GB payload at every launch. Run `sha256sum -c checksums.sha256` after download or transfer before using that setting.

## Confirm the MTP profile and choose a draft depth

`scripts/ciru/run-server.sh` loads and exports `profiles/strix-halo-production.env`, including `CIRU_MTP_SHORTLIST=32768` and `CIRU_MTP_TOPK10=1`. Building v2.0 and passing `--spec-draft-n-max 6` directly to `llama-server` does not enable these environment-controlled optimizations. For a direct launch, use the complete environment and command above.

With MTP enabled, confirm this line appears in the startup log:

```text
CIRU MTP shortlist enabled: 32768 / 248320 vocabulary rows; full target verification retained
```

The shortlist restricts draft output projection; target verification still uses the full vocabulary. An absent line means the run has not confirmed the released shortlist configuration.

**Maximum depth 6 is the released default, not a guarantee of the best speed on every prompt.** A deeper draft can waste work when later tokens are rejected. Depth 3 may be faster for a particular interactive workload with low acceptance. To select it while retaining the rest of the v2.0 profile, restart the server with:

```bash
MTP_DEPTH=3 BUILD_DIR="$PWD/build-gfx1151-sdk" \
  MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
```

The headline 42.28–42.31 tok/s was measured on a non-thinking coding probe with 57 prompt tokens, 520 generated tokens, greedy sampling, one slot, 16K configured context and the performance CPU governor. It does not establish that six is optimal for short chat, long contexts or batched serving. No adaptive depth selection is enabled. The MTP-off context sweep cannot establish an optimal MTP depth.

For a useful comparison, retain the exact source/build identity, launch command and exported profile, shortlist startup line, request JSON and effective sampling settings (including min-p), thinking mode, actual prompt/output counts, cache state, drafted/accepted counts and timing definition. Report generation rate separately from time to first token and end-to-end latency. A single sampled completion is directional evidence; acceptance percentage alone does not determine speed or show that one runtime is generally faster. Compare depths within v2.0 with all other settings held fixed; when comparing versions, record each version's supported profile explicitly.

## Production cache behavior

The public profile intentionally differs from the measurement harness:

- `--cache-prompt` keeps reusable prompt prefixes.
- `--cache-ram 8192` allocates an 8 GiB RAM prompt cache.
- `--cache-idle-slots` allows idle slots to retain useful KV state.
- `--ctx-checkpoints 32 --checkpoint-min-step 8192` restores context checkpoints for long conversations.
- `--slot-save-path` provides explicit slot-state save/restore storage; it is not a replacement for the live prompt cache.
- `--ple-cache-mib 4096` is a separate 4 GiB decoded PLE-page cache.

Do not copy benchmark controls such as `--ctx-checkpoints 0`, slot erases, `cache_prompt:false`, a fixed seed, `temperature:0`, `ignore_eos`, or a small fixed `n_predict` into a general public server profile. Published prefill speeds are cold/uncached measurements so repeated-prefix production traffic can benefit from caching without invalidating the benchmark claim.

## API request examples

The server exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`.

Thinking mode, which is Qwen's default:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-CIRU-STRIX-IU4",
    "messages": [{"role": "user", "content": "Design a robust retry policy."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "cache_prompt": true
  }'
```

Non-thinking mode:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-CIRU-STRIX-IU4",
    "messages": [{"role": "user", "content": "Summarize this in three bullets."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "cache_prompt": true
  }'
```

These follow the [upstream Qwen recommendations](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/README.md#best-practices). The v2.0 launcher explicitly selects `min_p=0`; v1.1 could inherit llama.cpp's `0.05` when no override or GGUF min-p key was present. Zero disables this additional filter; `0.05` can remove candidates below 5% of the most probable remaining token's probability. This can affect sampled output even when top-k and top-p are also enabled.

Keeping zero follows the model author's preset. We have not established a quality advantage over `0.05` in a controlled A/B generation test, and the greedy speed results do not resolve that question. Use `MIN_P=0.05` with the launcher to select the previous inherited value; request-level settings may override server defaults. Tune sampling for your application; do not treat benchmark decoding as a general serving preset.

## Running without MTP

The target and PLE sidecar can run without the MTP draft. Remove all `--spec-*` flags, or set `ENABLE_MTP=0` when using the launcher:

```bash
BUILD_DIR="$PWD/build-gfx1151-sdk" ENABLE_MTP=0 MODEL_DIR=/absolute/path/to/model \
  ./scripts/ciru/run-server.sh
```

This reduces disk and memory pressure but gives up the published speculative-decoding profile.

## Parallel requests and unified KV cache

**The v2.0 MTP shortlist supports exactly one slot.** Setting `PARALLEL_SLOTS=2` while leaving MTP enabled hits a runtime assertion, with either split or unified KV. This is separate from upstream reports about HIP host buffers or unified-cache state. The launcher on `main` now rejects that configuration before loading the model and explains the target-only option; the original `v2.0` tag and archive are unchanged.

For two slots, disable MTP explicitly. This command also works with the original v2.0 launcher:

```bash
ENABLE_MTP=0 PARALLEL_SLOTS=2 CONTEXT_SIZE=524288 \
  BUILD_DIR="$PWD/build-gfx1151-sdk" MODEL_DIR=/absolute/path/to/model \
  ./scripts/ciru/run-server.sh --no-kv-unified
```

With separate KV caches, `524288` is the **total** context allocation: two slots receive `262144` tokens each. It is not 512K per agent. Check `/slots` for the actual per-slot limit. Unified KV shares the pool; this runtime still caps each slot at the model's 262144-token training context.

On 2026-09-06, the released CIRU/ROCm 10 build completed 16 target-only requests across two fresh loads: eight with separate KV and eight with unified KV, both at `-c 524288 -np 2`. These mixed short prompts, a 6659-token reference fixture, overlapping requests, queued requests, streamed responses and prefix caching. All returned their own requested marker, with no foreign markers, transport failures or GPU errors. This is a bounded concurrency smoke test, **not validation of filled 512K contexts or every agent workload**. These marker checks did not validate retrieval from an earlier conversation after another request joined, so they do not rule out the model-specific QSA bug below. Use the explicit separate-KV configuration above for two slots.

The community subsequently identified [upstream #27994](https://github.com/ggml-org/llama.cpp/issues/27994), a **Qwen3.8/QSA unified-cache sequence-isolation bug**, fixed upstream by [#27941](https://github.com/ggml-org/llama.cpp/pull/27941) on 2026-09-01. **The relevant fix is missing from our v2.0 source:** the QSA block map still groups cells by position without separating sequence sets. The PR also repairs indexer-key copying; that update is absent too. The older fragmented-KV/SWA fix #23981 and the separate HIP host-buffer report #25992 are different issues.

**Keep `--no-kv-unified` explicit for multi-slot serving until the QSA backport is integrated and validated.** The new launcher guard addresses the one-slot MTP assertion; it does not repair QSA sequence isolation or implement multi-slot MTP. The source archive, release tags and weights are unchanged. The separate proposed HIP patch #25863 remains held after the MTP regression comparison; it is not the fix for #27994.

[Validation and source-audit details](PARALLEL_VALIDATION.md).

## Network exposure

The launcher binds to loopback. Do not change it to `0.0.0.0` on an untrusted network without authentication, TLS, request limits, and a reverse proxy. Exposing `/metrics` and `/slots` also exposes operational information.

## Health and observability

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/metrics | head
curl -fsS http://127.0.0.1:8080/slots
```

On first load, confirm that the log identifies `gfx1151`, the PLE sidecar, a 4 GiB PLE cache, the Q8_0 MTP draft, and no fallback or pager error.
