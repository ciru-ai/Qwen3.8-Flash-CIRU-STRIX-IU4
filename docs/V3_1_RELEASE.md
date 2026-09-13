# v3.1.0 release draft

This is staging material, not a published or fully qualified release. Measurements and final batch defaults are pending the exact own-weight serving campaign. Do not publish the placeholder values as performance claims.

The proposed update uses the reference prefill engine with an isolated CIRUPLE1 bridge, exact-format Q4_1 routed-expert/GLU support and MTP pending-state correctness fixes. It keeps the existing target GGUF, three PLE files, separate Q8 MTP head and optional vision projector. No requantization or model redownload is required.

| Release result | Pending value |
|---|---|
| Exact source and binary identity | Await final source/build seal |
| Cold served PP and context/depth | Await qualified matched measurement |
| TG, first-piece and whole-request latency | Await qualified matched measurement |
| Own-weight quality, cache and MTP gate | Await final per-case results |
| Capacity and memory floor | Await final qualified serving settings |
| Optional vision smoke | Await this engine's projector/image check |

The draft launcher retains context 262144, one-slot MTP6, target F16/draft Q8 KV, 4096 MiB PLE cache and production prompt caching. Proposed batch/microbatch 16384 and prompt-cache RAM 4096 MiB remain subject to actual serving results. It preserves the public sampler and embedded thinking default. Request-specific benchmark settings are not server defaults.

The old CIRU shortlist and specialized QSA/TOPK10 environment switches are not the reference engine's controls. The new profile enumerates the complete published reference bundle; a flag activates only its supported type/shape path. Our unchanged Q5 dense weights do not become IQ4 or receive unimplemented Q5 BF16 shadows.

The web UI is packaged externally under `ui/`, built from the pinned reference source and lockfile. `--path` serves these assets with the matched server. UI-PROVENANCE.json identifies every asset and the local build commands. Keep these assets with the runtime package; the diagnostic inference binary contains no embedded UI.

Start a new `slot-state/v3.1.0` directory. Older saved slots remain untouched and are not claimed compatible across engines. Live prompt caching and MTP reuse depend on matching checkpoints; a safe full reprocess can occur when none exists. Explicit disk slot persistence is not complete MTP-session persistence.

The retained tested Nix v3 binary and current diagnostic binary both effectively lack HTTPS support. Local model files and data-URL image requests remain the intended path. Source builds can enable HTTPS when OpenSSL development files are available. Optional vision stays opt-in and requires the existing matching projector.

Credit: Qwen for the model; pwilkin and llama.cpp contributors for the reference engine and prefill work; CIRU/Crown for the unchanged weight package, integration, focused Q4 support and qualification. Preserve the original licenses, notices and source provenance when assembling the release.
