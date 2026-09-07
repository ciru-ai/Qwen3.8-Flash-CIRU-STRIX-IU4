# CIRU runtime v2.0.1

Correctness, scoped performance and clean Ubuntu GPU qualification completed on 2026-09-07.

This repository contains the runtime for [Qwen3.8-Flash-CIRU-STRIX-IU4](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0.1). v2.0.1 fixes QSA conversation isolation with unified KV, indexer cache copying and a large-cache normalization boundary. The target, PLE and MTP weights are unchanged.

[Release downloads](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v2.0.1) · [Git tag](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0.1) · [Diff from v2.0](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/compare/v2.0...v2.0.1) · [Qualification](docs/QSA_BACKPORT_STATUS.md) · [Source identity](docs/PROVENANCE.md)

To obtain the exact released source with Git:

```bash
git clone --branch v2.0.1 --single-branch https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git ciru-runtime-v2.0.1
cd ciru-runtime-v2.0.1
```

On Ubuntu/Debian, from the cloned or extracted runtime directory:

~~~bash
./scripts/ciru/setup-linux-amd.sh --install-host-deps
BUILD_DIR="$PWD/build-gfx1151-sdk" MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
~~~

The private SDK is pinned to ROCm 10.0.0 with gfx1151 libraries. The production profile uses 262144-token context, F16 target KV, Q8_0 draft KV, maximum MTP depth 6, a 32768-row draft shortlist and normal prefix caching.

**Parallel serving:** the released MTP shortlist requires one slot. For two slots, set `ENABLE_MTP=0`; [the two-slot instructions](docs/RUNNING.md#parallel-requests-and-unified-kv-cache) explain the total context allocation, tested KV modes and upstream bug status. The v2.0.1 launcher catches unsupported multi-slot MTP configurations before model load.

Use the launcher to load the complete environment, including `CIRU_MTP_SHORTLIST=32768` and `CIRU_MTP_TOPK10=1`. Setting draft depth alone does not enable these optimizations. Depth 6 is workload-dependent; `MTP_DEPTH=3` selects a shallower draft while keeping the rest of the profile. See [profile verification and depth selection](docs/RUNNING.md#confirm-the-mtp-profile-and-choose-a-draft-depth) for the startup check, copyable command and evidence limits.

v2.0.1 passed the isolation/copy checks, the later Sozo 8K and Ciru 64K serving gates, and a clean Ubuntu 24.04 / ROCm 10 build with 66 QSA cases and 30 batch-allocation tests. The Ubuntu GPU completion matched all 520 reference tokens and MTP counts; device identity and 15 binary hashes were verified. The earlier Sozo 64K slowdown remains unexplained, and profiler attachment failed before any diagnostic request. Performance measurements use NixOS; the Ubuntu check uses container userspace on Ciru’s NixOS GPU driver. See [Linux build instructions](docs/BUILD_LINUX.md), [running instructions](docs/RUNNING.md), and [historical v2.0 evidence](docs/V2_RELEASE.md).

The model retains its existing IU4 name. The standard launcher uses ordinary GGUF tensor types; the Q4_1 matrix path expands packed values into byte lanes and uses IU8 WMMA. It does not activate the separate native IU4/E3 bank path. This release corrects the description without changing weights or the measured execution path.

The runtime retains the upstream MIT license and [third-party notices](THIRD_PARTY_NOTICES.md). Models use Qwen Community License 1.0. Credits and lineage remain in [PROVENANCE.md](docs/PROVENANCE.md).
