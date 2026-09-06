# CIRU runtime v2.0

This repository contains the runtime for [Qwen3.8-Flash-CIRU-STRIX-IU4 v2.0](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0). The target, PLE and MTP weights are unchanged.

The [GitHub `v2.0` tag](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0) matches every file and Git mode in the original published source archive. [Release downloads](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v2.0) · [Diff from v1.1.1](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/compare/v1.1.1...v2.0) · [Source identity](docs/PROVENANCE.md#v20-source-in-git). Later commits on `main` clarify the documentation.

To obtain the exact released source with Git:

```bash
git clone --branch v2.0 --single-branch https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git ciru-runtime-v2.0
cd ciru-runtime-v2.0
```

On Ubuntu/Debian, from the cloned or extracted runtime directory:

~~~bash
./scripts/ciru/setup-linux-amd.sh --install-host-deps
BUILD_DIR="$PWD/build-gfx1151-sdk" MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
~~~

The private SDK is pinned to ROCm 10.0.0 with gfx1151 libraries. The production profile uses 262144-token context, F16 target KV, Q8_0 draft KV, maximum MTP depth 6, a 32768-row draft shortlist and normal prefix caching.

Use the launcher to load the complete environment, including `CIRU_MTP_SHORTLIST=32768` and `CIRU_MTP_TOPK10=1`. Setting draft depth alone does not enable these optimizations. Depth 6 is workload-dependent; `MTP_DEPTH=3` selects a shallower draft while keeping the rest of the profile. See [profile verification and depth selection](docs/RUNNING.md#confirm-the-mtp-profile-and-choose-a-draft-depth) for the startup check, copyable command and evidence limits.

All three binaries built in a clean Ubuntu 24.04 container and the resulting build completed a real GPU/MTP smoke. Full comparison metrics refer to the retained NixOS RC2 binaries. See [Linux build instructions](docs/BUILD_LINUX.md), [running instructions](docs/RUNNING.md), and [current release evidence](docs/V2_RELEASE.md).

The runtime retains the upstream MIT license and [third-party notices](THIRD_PARTY_NOTICES.md). Models use Qwen Community License 1.0. Credits and lineage remain in [PROVENANCE.md](docs/PROVENANCE.md).
