# Running CIRU v4.1.0

Download the unchanged model, all three `ple/` files and the `mtp/` draft from the matching Hugging Face tag. The source archive and repository include the profile, launchers and hashed external `ui/` assets.

```bash
MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
```

The default build location is `build-gfx1151-sdk`. Override `BUILD_DIR` or `SERVER_BIN` for another matching build. The tested Nix binary bundle can be launched using `SERVER_BIN=/absolute/path/to/bundle/bin/llama-server`; it requires the exact Nix and SDK dependencies in its binary identity. The launcher puts the selected binary's directory first in `LD_LIBRARY_PATH`.

The model-side launcher is also supported:

```bash
RUNTIME_DIR=/absolute/path/to/runtime SERVER_BIN=/absolute/path/to/runtime/build-gfx1151-sdk/bin/llama-server   bash /absolute/path/to/model/run-server.sh
```

Open http://127.0.0.1:8080 for the UI. OpenAI-compatible chat uses `/v1/chat/completions`. `HOST`, `PORT`, `MODEL_DIR`, `UI_DIR` and `SLOT_DIR` override locations. `ENABLE_UI=0` supports API-only serving. A missing projector or UI directory fails before model load.

## Defaults and overrides

Context262144; batch/microbatch 8192; target and draft F16 KV; one slot; MTP depth 6; prompt RAM cache 1024 MiB; PLE cache 4096 MiB; 32 checkpoints with 8192-token minimum step. The complete15-request panel retained at least 6.82 GiB available memory on our 128 GB host. Raising batch size or prompt cache exhausted the 4 GiB guard in earlier runs; these settings consume separate memory pools.

The sampler remains temperature 1.0, top-p 0.95, top-k 20 and min-p 0. Thinking follows the embedded template default. Request parameters or `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P` override sampling. Qualification used explicit nonthinking requests with seed 123, temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence 1.5 and repeat 1; those benchmark settings are not production defaults.

`CONTEXT_SIZE`, `BATCH_SIZE`, `UBATCH_SIZE`, `PROMPT_CACHE_MIB`, `PLE_CACHE_MIB`, `CTX_CHECKPOINTS`, `CHECKPOINT_MIN_STEP` and `MTP_DEPTH` are available. `ENABLE_MTP=0` selects target-only serving. Saved-slot state defaults to the new `slot-state/v4.1.0`; old saves are not restored automatically. Prefixes without a compatible MTP state checkpoint reprocess safely.

## Parallel requests and unified KV cache

MTP is qualified only with one slot. The launcher rejects multi-slot MTP, including trailing parallel overrides. `ENABLE_MTP=0 PARALLEL_SLOTS=2` is available for experiments, but new-engine multi-slot serving has not been qualified. The default disables unified KV and context shifting.

## Vision

```bash
MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh --vision
```

`ENABLE_VISION=1` is equivalent. It loads `vision/mmproj-Qwen3.8-Flash-F16.mmproj`; `MMPROJ` overrides the path. **Vision mode disables MTP**, including when `ENABLE_MTP=1` is set. Do not override speculative CLI options in vision mode. Two image smoke checks passed with the existing 904003840-byte projector, SHA256 `db643482521c722ff1074afd5018c060ef6ce9b828421c7cfc27b2f235c2569b`.

Use image data URLs with the tested Nix binary, whose HTTPS fetching is disabled. A source build can enable remote HTTPS image fetching when OpenSSL is found. Image processing uses additional memory/context. The text performance table does not measure vision.

## Orca and IO32

Use the existing Orca model directory with the shared v4.1 runtime:

```bash
MODEL_DIR=/path/to/orca-model bash ./scripts/ciru/run-orca-server.sh
```

The Orca launcher selects its own target, Q8 MTP head, projector and slot directory. Do not pair it with the non-Orca MTP head. The release profile enables 32 bulk PLE I/O workers. Use `GGML_QWEN4EXP_PLE_IO_WORKERS=16` to retain the previous concurrency. `GGML_QWEN4EXP_PLE_WORKERS` remains 16. No model or PLE replacement is required.
