# Community Qwen3.8 Flash fidelity benchmark

Compare your Qwen3.8-Flash-Next GGUF with CIRU IU4 v4.3, Agention AP-Q4_K_XL and AP-Q5_K_XL on **the exact same frozen corpus and original BF16 reference**. Intended for Linux on AMD Ryzen AI Max+ 395 / Strix Halo (gfx1151), with enough unified RAM for your model. Other AMD hardware is not qualified by these results.

This measures numerical fidelity, not agent/task accuracy. HermesAgent-20 is not included. The dataset and baseline numbers are small and included in Git. The saved reference is a separate **5.84 GiB download**. No BF16 model download or reference inference is needed. Candidate logits use another **15.16 GiB**; allow at least 25 GiB of free disk beyond your model and runtime, and roughly 4 GiB of RAM for scoring once the capture process exits.

## Setup

Use Python 3.11+ and a matching CIRU v4.3 runtime **source and shared-library build**. Stock llama.cpp lacks this capture program's PLE/load-mode API. Quant formats supported by this runtime can be tested; a matching vocabulary alone does not establish that an arbitrary model architecture is supported. Fine-tunes must preserve the original tokenizer exactly. The NixOS binary archive is not a portable Ubuntu binary.

From the repository root:

```bash
python3 -m venv .venv-fidelity
. .venv-fidelity/bin/activate
python -m pip install -r benchmarks/fidelity/requirements.txt
python benchmarks/fidelity/fidelity.py fetch --cache "$HOME/flash-fidelity-reference"
```

Build the v4.3 runtime using [the Linux AMD instructions](../../docs/BUILD_LINUX.md). Use a separate source checkout if your current checkout is newer:

```bash
git clone --depth 1 --branch v4.3.0 \
  https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4.git runtime-v4.3
cd runtime-v4.3
./scripts/ciru/setup-linux-amd.sh --install-host-deps
# Follow the helper output; set ROCM_ROOT to the installed SDK.
ROCM_ROOT=/path/to/rocm-sdk BUILD_DIR="$PWD/build-amd" ./scripts/ciru/build-linux-amd.sh
cd ..
```

Keep the matching SDK libraries available in `LD_LIBRARY_PATH` as required by your build. A C++17 compiler is needed; set `CXX` if necessary. The runner compiles its small capture executable against the source headers and shared libraries you supply.

## Run your model

Close other GPU workloads first. The script never stops or changes system services. Ensure `/dev/kfd` and render-node access works. Large model files should be on SSD. The complete target must fit the available system memory with runtime overhead; “128 GB” is not a promise that every GGUF fits.

```bash
python benchmarks/fidelity/fidelity.py run \
  --model /models/My-Qwen3.8-Flash.gguf \
  --source "$PWD/runtime-v4.3" \
  --runtime "$PWD/runtime-v4.3/build-amd/bin" \
  --label 'My Flash quant' \
  --cache "$HOME/flash-fidelity-reference" \
  --out "$PWD/my-flash-result"
```

For a CIRU package with external PLE, add `--ple /models/my-model/ple`. For a split GGUF, provide the first shard as `--model` and each other shard with `--extra-model-file`; the runtime must support loading that split package. All target shards and sidecar files contribute to the package size and identity. Do not include an MTP head or vision projector: neither is used here.

Use `--device 0` to select the first HIP device (default). Ambient `LLAMA_*`, `GGML_*`, `HSA_*`, and retained-PM4 overrides are removed, then the measured PLE I/O worker count is set to 32. SDK library search paths are retained. Exact runtime library hashes and selected environment are recorded. A different runtime/build is a configuration comparison, not an isolated weight-quantization comparison.

Output:

- `result.json`: means, counts, confidence intervals, perplexity, per-window results and source hashes.
- `comparison.png`, `.svg`, `.pdf`: your result with the three measured baselines.
- `run.json`, `capture.log`, `capture/manifest.json`: model/runtime identity and raw execution evidence.
- `capture/logits.f32le`: full raw candidate logits; keep for audit, or archive after verifying the report.

Existing output directories are refused. Failed captures stay intact. No partial scores or automatic inference retries are produced. Download retries are independent of scored inference.

## Python chart source and reference outputs

[`plot.py`](plot.py) is included. Recreate just the current baseline chart without a GPU or reference download:

```bash
python benchmarks/fidelity/plot.py --out baseline-comparison
```

Add a completed community result:

```bash
python benchmarks/fidelity/plot.py --result my-flash-result/result.json --out my-comparison
```

The original baseline aggregate outputs are in [`data/baselines.json`](data/baselines.json); per-window baseline outputs in [`data/baseline-per-window.json`](data/baseline-per-window.json), and the frozen source revisions, hashes and window assignments are in [`data/panel.json`](data/panel.json). PNG/SVG baseline charts are included. The full historical candidate logit files are not bundled. All 16 teacher shards in the release are losslessly packed BF16: every discarded low 16 bits of the original F32 storage was checked to be zero. SHA-256 checks cover compressed and expanded shards; the original F32 reference hash is retained.

## Frozen protocol

16 windows, each 2,048 input tokens plus the next-token label; score positions 1,024 through 2,047, giving 16,384 full-vocabulary distributions. Eight public repositories, two adjacent windows each. No chat wrapper, special tokens, sampling, generation, reasoning budget, MTP or answer repair. Fresh context per window. Candidate capture uses F16 KV, flash attention, batch/microbatch 512, eight CPU threads, GPU target layers, and native CPU/sidecar PLE.

Primary metrics: strict argmax agreement and forward KL(reference || candidate), in nats. Secondary: exact-BF16-tie-aware agreement and observed-token NLL/perplexity. Confidence intervals use 10,000 bootstrap resamples over **eight repository clusters**, seed 20260916, keeping each repository's windows together. This does not treat 16,384 positions as independent samples.

Reference: `Qwen/Qwen3.8-Flash-Next`, revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`, original Transformers BF16 logits. Teacher arithmetic differs from candidate llama.cpp arithmetic. The numbers describe the complete configurations. The engineering corpus is narrow and its recent dates do not establish absence from training. **Do not compare these percentages directly to Agention's published chart**, which uses another corpus/reference. A lower KL does not guarantee a higher agent score.

## Validation and sharing

Run the small CPU integrity tests with `python benchmarks/fidelity/test_fidelity.py`. They cover identity, tied argmax, nonfinite logits, dataset hashes, corrupt reference files and incomplete captures.

The capture C++ source is unchanged from the measured runs. The community scorer is replay-tested against retained raw IU4/Q4/Q5 outputs and must reproduce the saved metrics. See `VALIDATION.json` for the actual completed checks; no new Ubuntu or arbitrary-model GPU qualification is implied.

To report a result, share `result.json`, `run.json`, chart and capture log, plus your OS, driver/ROCm, hardware/RAM, model revision and runtime commit. Check the files for local path/hostname information before sharing. There is no automatic upload or leaderboard submission.

Corpus excerpts retain their upstream licenses. See [THIRD_PARTY.md](THIRD_PARTY.md) and `licenses/`. Benchmark code uses the repository MIT license. No model weights are redistributed by this kit.
