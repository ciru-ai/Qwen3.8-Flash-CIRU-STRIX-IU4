# Release provenance

## v2.0.1 source and build identity

v2.0.1 is the qualified QSA source release. The named archive and external source identity record identify its exact Git tree.

At publication, the [v2.0.1 Git tag](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0.1) and named `ciru-runtime-v2.0.1-source.tar.gz` release asset identify the same file contents, executable modes and symlinks. The [release](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v2.0.1) supplies the archive, SHA256SUMS and `git-source.json` with the exact commit/tree IDs and archive hash. [Compare v2.0 to v2.0.1](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/compare/v2.0...v2.0.1).

The focused backport preserves Daniel Han's authorship for upstream [#27941](https://github.com/ggml-org/llama.cpp/pull/27941), commit `36b10154383b60eb15baac2c7a40d2a5f784faa7`. Ciru adds the guarded canonical-stream implementation and regression fixtures. `CIRU_RELEASE.json` records the five changed core files and tested library identities. Weights and HIP sources are unchanged. The retained NixOS HIP binary is unchanged; the clean Ubuntu 24.04 / ROCm 10 build has a separate binary hash map. See [qualification](QSA_BACKPORT_STATUS.md) for the 66 QSA / 30 batch-test results, later Sozo 8K and Ciru 64K serving comparisons, and Ubuntu GPU output/count match with verified device and binary identities. The earlier Sozo slowdown and failed profiler attachment remain documented.

Original v1.1.1/v2.0 tags and the original v2.0 source archive retain their identities. The historical source record follows.

## v2.0 source in Git

The [GitHub `v2.0` tag](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0) points to [`3e21240ec793`](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/commit/3e21240ec7935b18fd39e1f07fd80f8d905ba968), a direct child of `v1.1.1` (`764ee491d4bc765cb8414d9bb17c24a5b364e097`). [Compare v1.1.1 to v2.0](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/compare/v1.1.1...v2.0).

All **3,537 tracked file contents, executable modes and symlink targets** at that tag match the original published `ciru-runtime-v2.0-source.tar.gz`, after removing its enclosing directory. The original archive SHA-256 is:

```text
938bd98629fc25014973082eef5a3ea4d45dd5acb6732b608e05e0ecde08f044  ciru-runtime-v2.0-source.tar.gz
```

The original archive is available from the [GitHub release](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v2.0) and [Hugging Face](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/resolve/v2.0/runtime/v2.0/ciru-runtime-v2.0-source.tar.gz). GitHub's automatically generated source archives may have different packaging and hashes; the checksum above identifies the named original asset. The tag records the existing release; publishing it does not introduce new runtime code or weights.

Documentation clarifications are subsequent commits on `main`. The GitHub tag, original Hugging Face `v2.0` revision and source archive retain their original release contents. Use the [current running instructions](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/main/docs/RUNNING.md) for the profile and draft-depth guidance.

## Model lineage

| Component | Source | Frozen revision |
|---|---|---|
| Text model and tokenizer lineage | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| Exact FP8 PLE sidecar source | [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` |

Both upstream repositories may have moved since the release was built; the revisions above are the reproducibility anchors.

The public release is text-only and does not include a vision projector.

## Runtime lineage

- Clean public base: [`ggml-org/llama.cpp@f5e85d43a048f3d5adefb4c5e29867d8077fba62`](https://github.com/ggml-org/llama.cpp/commit/f5e85d43a048f3d5adefb4c5e29867d8077fba62).
- Qwen experimental/MTP support was adapted from the integration represented by `1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`; it is a port onto the pinned base, not a claim that the base commit already contained that integration.
- CIRU additions include the optimized Q4_1/IU8 gfx1151 kernels, PLE paging/banking, protected-core loading, speculative controls, and serving changes required by this artifact.
- H121 release fix: persistent MTP continuation inputs in `src/models/qwen4exp.cpp`.

The exact H121 patch has SHA-256:

```text
cd20f28b1939d93d4d24226f3acbdba82fe43c578a69b251db3aa14bce684c94  h121-mtp-persistent-inputs.patch
```

The corrected `src/models/qwen4exp.cpp` used for the release source had SHA-256 `90fa4b8ddbbfbff74b683c8ec3c4dd1bf311f8a340f2975c241d9d9d25f21520` before the public documentation commit.

## Evaluated artifact to public-name mapping

The final evaluated tensors were not requantized for publication. The two GGUF headers were mechanically rewritten to replace their internal names and descriptions with the exact public model names, which changed the file length and SHA-256 while leaving tensor arrays unchanged.

| Component | Evaluated file identity | Public file identity |
|---|---|---|
| Target | 79,397,818,656 B; `0c9cb11d34f9ae241180798a22de160cedc493600c9ae9d1a35b34e330d93eb8` | 79,397,818,720 B; `c0ea11e4e24d0f909720b6c4e7462aa1e6fbf5e0f6acc796063f2aed4cf46ed0` |
| MTP Q8_0 | 4,135,893,152 B; `cf6054d50ad260ba5b9be03b6d6d15b5100e9727bfd16f81734eba57c657f798` | 4,135,893,248 B; `e6743badef1f2619fcb5addfa4344a2a3368cb75214735117e3af80c70b80642` |
| PLE payload | 52,429,053,952 B; `687fc742efb6888c6cd7cf9c80cb4b1ac8cb4707b9409c206699c43363e239b2` | Unchanged |

Public GGUF metadata:

```text
general.name = Qwen3.8-Flash-CIRU-STRIX-IU4
general.name = Qwen3.8-Flash-CIRU-STRIX-IU4-MTP
```

## Composition

The target contains 1,223 tensors:

| Storage type | Tensor count |
|---|---:|
| F32 | 388 |
| Q5_K | 328 |
| Q8_0 | 290 |
| Q4_1 | 144 |
| Q5_1 | 48 |
| BF16 | 25 |

The 144 Q4_1 routed-expert tensors occupy 75,497,472,000 bytes. The remaining protected core occupies 3,900,335,968 bytes. No tensor is stored on disk as a custom `IU4_A640` type; the standard launcher uses ordinary Q4_1 storage, expands packed values to byte lanes for the IU8 WMMA matrix path, and does not activate the separate native IU4/E3 bank path.

## Validation identities

The post-fix H121 binary used for the 8K row and stability stress had SHA-256 `3eff75a0e6276d8ebef284f981bfbc7bf04de89de5231df33df5ad609420c`.

The raw matched-row capture had SHA-256 `3c25acd705bb02a76f6e0ae71642955f1f97e190e78b0eb1d5394f6511d6b126`.

The clean public source export was separately configured and built on CPU through all `llama-server`, `llama-cli`, and `llama-bench` targets before publication. GPU loading of the public-name GGUFs is recorded in the release verification notes once the benchmark host is free.
