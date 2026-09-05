# Running in production

## 1. Download and verify the complete package

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  --local-dir ./model
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
  --metrics --slots \
  --spec-type draft-mtp \
  --spec-draft-model /absolute/path/to/model/mtp/Qwen3.8-Flash-CIRU-STRIX-IU4-MTP-Q8_0.gguf \
  --spec-draft-ngl all --spec-draft-device ROCm0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --spec-draft-threads 8 --spec-draft-threads-batch 8 \
  --spec-draft-n-max 3 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 --spec-draft-p-split 0.10
```

`GGML_QWEN4EXP_PLE_STRICT_SHA=0` avoids hashing the 52.4 GB payload at every launch. Run `sha256sum -c checksums.sha256` after download or transfer before using that setting.

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

These follow the upstream Qwen recommendations. Tune sampling for your application; do not treat benchmark decoding as a general serving preset.

## Running without MTP

The target and PLE sidecar can run without the MTP draft. Remove all `--spec-*` flags, or set `ENABLE_MTP=0` when using the launcher:

```bash
ENABLE_MTP=0 MODEL_DIR=/absolute/path/to/model \
  ./scripts/ciru/run-server.sh
```

This reduces disk and memory pressure but gives up the published speculative-decoding profile.

## Network exposure

The launcher binds to loopback. Do not change it to `0.0.0.0` on an untrusted network without authentication, TLS, request limits, and a reverse proxy. Exposing `/metrics` and `/slots` also exposes operational information.

## Health and observability

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/metrics | head
curl -fsS http://127.0.0.1:8080/slots
```

On first load, confirm that the log identifies `gfx1151`, the PLE sidecar, a 4 GiB PLE cache, the Q8_0 MTP draft, and no fallback or pager error.
