# Running CIRU v4.3.0

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

Context262144; batch/microbatch 8192; target and draft F16 KV; one slot; MTP depth 3 for IU4 or 4 for Orca; sparse draft attention from one token; prompt RAM cache 1024 MiB; PLE cache 4096 MiB; 32 checkpoints with 8192-token minimum step. The historical v4.0 complete 15-request panel retained at least 6.82 GiB available memory on our 128 GB host. Raising batch size or prompt cache exhausted the 4 GiB guard in earlier runs; these settings consume separate memory pools.

The sampler remains temperature 1.0, top-p 0.95, top-k 20 and min-p 0. Thinking follows the embedded template default. Request parameters or `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P` override sampling. Historical v4.0 qualification used explicit nonthinking requests with seed 123, temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence 1.5 and repeat 1; those benchmark settings are not production defaults.

`CONTEXT_SIZE`, `BATCH_SIZE`, `UBATCH_SIZE`, `PROMPT_CACHE_MIB`, `PLE_CACHE_MIB`, `CTX_CHECKPOINTS`, `CHECKPOINT_MIN_STEP` and `MTP_DEPTH` are available. `ENABLE_MTP=0` selects target-only serving. Saved-slot state defaults to the new `slot-state/v4.3.0`; old saves are not restored automatically. Prefixes without a compatible MTP state checkpoint reprocess safely.

## Parallel requests and unified KV cache

MTP is qualified only with one slot. The launcher rejects multi-slot MTP, including trailing parallel overrides. `ENABLE_MTP=0 PARALLEL_SLOTS=2` is available for experiments, but new-engine multi-slot serving has not been qualified. The default disables unified KV and context shifting.

## Vision

```bash
MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh --vision
```

`ENABLE_VISION=1` is equivalent. It loads `vision/mmproj-Qwen3.8-Flash-F16.mmproj`; `MMPROJ` overrides the path. **MTP stays enabled with vision by default.** Set `ENABLE_MTP=0` for target-only generation. Vision with MTP has not been newly inference-qualified; the historical image smoke checks below used target-only generation. Two image smoke checks passed with the existing 904003840-byte projector, SHA256 `db643482521c722ff1074afd5018c060ef6ce9b828421c7cfc27b2f235c2569b`.

Use image data URLs with the tested Nix binary, whose HTTPS fetching is disabled. A source build can enable remote HTTPS image fetching when OpenSSL is found. Image processing uses additional memory/context. The text performance table does not measure vision.

## Orca and IO32

Use the existing Orca model directory with the shared v4.3 runtime:

```bash
MODEL_DIR=/path/to/orca-model bash ./scripts/ciru/run-orca-server.sh
```

The Orca launcher selects its own target, Q8 MTP head, projector and slot directory. Do not pair it with the non-Orca MTP head. The release profile enables 32 bulk PLE I/O workers. Use `GGML_QWEN4EXP_PLE_IO_WORKERS=16` to retain the previous concurrency. `GGML_QWEN4EXP_PLE_WORKERS` remains 16. No model or PLE replacement is required.

## V4.3 tuning and evidence

See [the v4.3 qualification](qualification/v4.3.0/QUALIFICATION.md) for the tuning and completed Hermes Agent 20 results. The v4.2 attention correction is retained unchanged.

## Updating an existing v4.2 installation

V4.3 changes launch settings and metadata; the v4.2 inference binaries are reused. Fetch the new model-directory `run-server.sh` from its Hugging Face `v4.3.0` tag and run it with `RUNTIME_DIR` pointing to your corrected v4.2 installation. No rebuild or model download is needed. Alternatively use the complete v4.3 source or tested binary package. An older v4.0/v4.1 runtime still needs the v4.2 attention correction.

`MTP_DEPTH=6 LLAMA_MTP_QSA_MIN_T=128` restores the previous speculative settings. Explicit overrides, including `LLAMA_MTP_QSA_MIN_T=0`, remain respected.

## Runtime verification in current launchers

The current main-branch launcher checks the HIP shared library selected by the dynamic loader before loading a model. It requires the v4.2 indexed-attention correction and prints the selected server, HIP library and library SHA256. An old RUNTIME_DIR, BUILD_DIR or SERVER_BIN fails with upgrade instructions; downloading a newer model-directory launcher does not replace those binaries. Clear GGML_BACKEND_PATH when using this supported shared-library launcher.

This startup check was added after the v4.3.0 tag and does not change inference kernels, weights, MTP defaults or sampling. The immutable release archives retain their original launchers. Existing users can fetch run-server.sh and launcher-checksums.sha256 from their Hugging Face model repository at revision main, verify the small checksum file, and run the launcher with their corrected v4.2 or v4.3 runtime. Source builds from main include the check directly. The check detects the correction's compiled kernel names; it does not certify a custom build as byte-identical to the published binaries or prove model quality.
