# CIRU runtime v3.0.0

V3 brings the portable hybrid-campaign improvements to the Strix-only Qwen3.8 Flash runner. The target, PLE and MTP weights are unchanged from v2.0.1. It adds wider prefill, parallel QSA selection, indexed decode attention, cached derived attention history and faster PLE lookups. The default batch and microbatch are 1024.

At 4,096 input tokens, v3 measured 455.65 prompt / 24.60 generation tok/s and 14.41s per request; unmodified Halo Vulkan measured 381.49 / 35.30 tok/s and 14.69s. At 65,536 input tokens, v3 measured 369.81 prompt / 24.22 generation tok/s and 182.57s per request; unmodified Halo Vulkan measured 263.42 / 23.28 tok/s and 254.37s. Each request generated 128 tokens. [Full comparison and limitations](docs/qualification/v3.0.0/COMPARISON.md).

[Changes and qualification](docs/V3_RELEASE.md) | [Run instructions](docs/RUNNING.md) | [Build instructions](docs/BUILD_LINUX.md) | [Release manifest](CIRU_RELEASE.json)

Use the [v3.0.0 GitHub release](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v3.0.0) or the matching named source archive. The published v2.0 model weights remain compatible. See [download and verification](docs/RUNNING.md#1-download-and-verify-the-complete-package).

On Ubuntu/Debian, from the cloned or extracted runtime directory:

~~~bash
./scripts/ciru/setup-linux-amd.sh --install-host-deps
BUILD_DIR="$PWD/build-gfx1151-sdk" MODEL_DIR=/absolute/path/to/model ./scripts/ciru/run-server.sh
~~~

The private SDK is pinned to ROCm 10.0.0 with gfx1151 libraries. The production profile uses 262144-token context capacity, F16 target KV, Q8_0 draft KV, maximum MTP depth 6, a 32768-row draft shortlist and prefix caching. Use the launcher to load the complete environment. `MTP_DEPTH=2` is the tested option for the longer, lower-acceptance coding requests; MTP 6 retains the higher throughput on short coding tasks. Each new v3 optimization can be disabled with its environment switch set to `0`; the optional draft attention window remains off.

**Parallel serving:** the released MTP shortlist requires one slot. For multiple slots, set `ENABLE_MTP=0`; the launcher checks this before loading the model. The [parallel instructions](docs/RUNNING.md#parallel-requests-and-unified-kv-cache) explain context allocation and the inherited v2.0.1 isolation fix. V3 also keys derived-history reuse by sequence identity.

The comparison uses the unmodified Halo Vulkan runtime in a documented working configuration and reports its model package separately from CIRU's. The same-weight v2/v3 measurements isolate the runtime change. Small coding and recall panels are regression checks, not broad model rankings. See [qualification and limitations](docs/V3_RELEASE.md).

V3 GPU qualification is on Ciru's NixOS host with ROCm 10 / gfx1151. The earlier clean Ubuntu build belongs to v2.0.1; it is historical evidence, not a new v3 Ubuntu qualification. Source build instructions are retained. The supplied tested binary payload requires the recorded Nix store and SDK paths.

The model retains its IU4 name. The standard launcher uses ordinary GGUF tensor types; the Q4_1 matrix path expands packed values into byte lanes and uses IU8 WMMA. It does not activate the separate native IU4/E3 bank path.

The runtime retains the upstream MIT license and [third-party notices](THIRD_PARTY_NOTICES.md). Models use Qwen Community License 1.0. [Credits and lineage](docs/PROVENANCE.md), [v2.0 history](docs/V2_RELEASE.md), and [v2.0.1 qualification](docs/QSA_BACKPORT_STATUS.md) are retained.
