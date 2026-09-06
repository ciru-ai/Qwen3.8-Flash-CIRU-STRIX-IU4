# CIRU runtime v2.0

This is the exact source package for [Qwen3.8-Flash-CIRU-STRIX-IU4 v2.0](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v2.0). The target, PLE and MTP weights are unchanged.

On Ubuntu/Debian, from this extracted directory:

~~~bash
./scripts/ciru/setup-linux-amd.sh --install-host-deps
BUILD_DIR="$PWD/build-gfx1151-sdk" MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
~~~

The private SDK is pinned to ROCm 10.0.0 with gfx1151 libraries. The production profile uses 262144-token context, F16 target KV, Q8_0 draft KV, maximum MTP depth 6, a 32768-row draft shortlist and normal prefix caching.

All three binaries built in a clean Ubuntu 24.04 container and the resulting build completed a real GPU/MTP smoke. Full comparison metrics refer to the retained NixOS RC2 binaries. See [Linux build instructions](docs/BUILD_LINUX.md), [running instructions](docs/RUNNING.md), and [current release evidence](docs/V2_RELEASE.md).

The runtime retains the upstream MIT license and [third-party notices](THIRD_PARTY_NOTICES.md). Models use Qwen Community License 1.0. Credits and lineage remain in [PROVENANCE.md](docs/PROVENANCE.md).
