# CIRU runtime v4.2.0

V4.2 fixes selected-key attention on the small-batch fallback used by the v4.0/v4.1 engine. QSA3 remains enabled. The existing IU4 and Orca model files are reused, and long thinking and production reasoning settings are unchanged.

The correction passes 48/48 independent numerical checks and completed the captured medium-effort tool-use replay on both variants. On one matched IU4 workload, decode measured 16.9% lower while prefill remained within the 5% parity band. Both mirrored comparisons agreed, but the figure is workload-specific and includes changed draft acceptance; it is not a universal or Orca speed claim.

We are investigating new GPU kernels and profiling the corrected attention path to increase performance further while preserving correctness.

[Patch notes and evidence](docs/qualification/v4.2.0/QUALIFICATION.md) | [Run and upgrade](docs/RUNNING.md) | [Build](docs/BUILD_LINUX.md) | [Release downloads](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v4.2.0)

## Previous v4.1.0 release

V4.1 enables 32-worker bulk PLE loading for the existing IU4 and Orca packages, with a 16-worker override. Model files and GPU arithmetic are unchanged. Non-Orca native 16k prefill improved 4.44%; a quick matched MTP6 check found no observed TG loss and identical paired outputs. Orca uses the same runtime; no new Orca speed claim is made.

[Run and upgrade](docs/RUNNING.md) | [Build](docs/BUILD_LINUX.md) | [v4.1 validation](docs/qualification/v4.1.0/QUALIFICATION.md) | [Release downloads](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v4.1.0)

## Previous v4.0.0 release

The pwilkin Strix Halo reference stack now runs with the existing CIRU Qwen3.8 Flash IU4 weights and CIRUPLE1 files. On our Strix Halo host, five cold 30.8K coding prompts measured approximately **992–1,001 prompt tokens/s**; cold 65K recall measured **948 PP**. These are diagnostic serving observations, not a clean repeated before/after comparison. Model weights are unchanged.

Credit for the original fast-prefill breakthrough goes to Halogen's creator, [Peonist.ai (`peonist-ai`)](https://github.com/peonist-ai/halogen-flash-server). [pwilkin](https://pwilkin.github.io/strix-halo/) reproduced that performance in an open-source llama.cpp implementation and published the kernels and configuration that CIRU adapted for these existing weights. CIRU's contribution is the compatibility integration and validation.

[Qualification and measured tradeoffs](docs/qualification/v4.0.0/QUALIFICATION.md) · [Run](docs/RUNNING.md) · [Build](docs/BUILD_LINUX.md) · [Model download](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v4.0.0)

The text profile uses context 262144, batch/microbatch 8192, F16 target/draft KV, MTP 6, one slot and a 1024 MiB prompt cache. It passed the 15-request serving panel, cache/recall checks and 261888-input+128-output capacity check. The original standalone coding score is 7/8; a uniform prompt-provided-import rescore of the same answers is 8/8. Both scopes and the original failure are retained. Decode gains are workload-dependent; full-context decode remains slow.

The historical v4.0 vision smoke used the existing projector with MTP disabled; current launcher behavior is documented in docs/RUNNING.md. The reused inference binaries and UI application assets passed UI, OpenAI chat and two image smoke checks; v4 updates the displayed UI version. The tested binary requires its recorded NixOS/ROCm10 dependencies. Other hosts should build this matching source. The included external UI matches the engine; keep the complete executable/shared-library set together.

The integration retains the CIRUPLE1 pager, adds type-correct Q4_1 expert paths to the reference kernels and repairs MTP pending-state and external-draft parameter handling. Some reference optimizations remain unavailable for incompatible quantization types. The remaining gap to the author's different reference weights is not fully attributed.

Thanks to [pwilkin](https://pwilkin.github.io/strix-halo/), Qwen, ggml-org, AMD/ROCm and all contributors credited in [provenance](docs/PROVENANCE.md), [upstream README](docs/UPSTREAM-README.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Runtime code retains MIT/component licenses; model artifacts use Qwen Community License 1.0. Previous v3 qualification remains under docs/qualification/v3.0.0 and the v3.0.0 tag.
