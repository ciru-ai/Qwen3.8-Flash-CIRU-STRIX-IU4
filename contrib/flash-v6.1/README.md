# Flash v6.1.0 hybrid on AMD Strix Halo

Download the packaged Linux runtime and the matching hybrid GGUF. The v6.1.0 profile uses the downloadable C1 hybrid, Unsloth’s matching Q8_0 MTP draft, and the original donor shard containing the per-layer embedding table. The original v6.0.0 IU4 package has its own model and runtime instructions. [Runtime download](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v6.1.0) · [Hybrid model and setup](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4-Hybrid/tree/v6.1.0) · [Original v6.0.0](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/v6.0.0)

## Start the server

Follow the model card’s download commands for the three pinned GGUF files. Extract the runtime archive and run these commands inside the extracted folder. Substitute the actual downloaded file paths.

```bash
python3 install-rocm.py
python3 run-server.py /path/to/halo-c1.gguf /path/to/draft.gguf /path/to/donor-shard.gguf
```

The server listens on port 8081 on localhost and serves model ID `main`. The validated configuration uses one slot, a 262,144-token context, F16 KV and MTP2. The binary is built for Linux x86-64 and AMD gfx1151. It requires Python 3.11 or newer, glibc 2.43 or newer and a libstdc++ exposing GLIBCXX_3.4.32. NixOS users can pass a matching library directory and loader via `--compat-lib-dir`. Use `python3 run-server.py --check` to verify archive contents and native loader resolution before loading the model. The package was tested on a 128 GB Sozo system. Plan for about 155 GB of disk space for the three GGUFs.

[Release notes](RELEASE-NOTES.md) · [Qualification](qualification.json) · [Runtime archive checksum](runtime/runtime-archive.sha256)

## Source and reproduction

The release installs prebuilt binaries. Developers can reconstruct the full patched runtime from HaloBox b1064 and rebuild it with the pinned ROCm SDK. The optional weight recipe independently assembles the exact released GGUF from the published upstream weights. [Runtime source and build](source/README.md) · [Weight provenance and assembly](weights/README.md)
