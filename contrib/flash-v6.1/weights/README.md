# Rebuild the Flash v6.1 hybrid GGUF

This optional Python 3.11+ recipe creates the same `halo-c1.gguf` supplied by the hybrid model release. It copies 144 compact C1 expert weight tensors and 288 C1 helper tensors from the Ciru IU4 v5.0 GGUF. It copies the remaining 1,080 tensors and model metadata from the pinned Unsloth donor. The C1 packet IDs are mapped to 150 and 151 for the matching runtime. [Ciru input](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4/tree/8a2f3caffb5c7f0a05c7285b7dac2c9e4f765434) · [Unsloth donor](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66)

## Download the source files

Allow about 289 GB of disk space for the six downloads and the assembled model. Use the Hugging Face CLI with the exact revisions below. Readers who already have the source files can link to them from the same directory layout.

```bash
assets="$PWD/flash-v6.1-models"
hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4 \
  Qwen3.8-Flash-CIRU-STRIX-IU4-v5.0.gguf \
  --revision 8a2f3caffb5c7f0a05c7285b7dac2c9e4f765434 --local-dir "$assets/compact"

hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
  MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 --local-dir "$assets/donor"
```

## Assemble or verify

From this recipe’s directory, run `python3 build-hybrid.py "$assets"`. It verifies the source files against their SHA-256 pins and publishes `model/halo-c1.gguf` under the asset directory after the assembled bytes match the expected digest. The expected output is 102,528,733,984 bytes and SHA-256 `4ff8156aabfa69ac8962a9bbbfded4e23add20f9246f5939cf4915c16e4fb8cb`.

Run `python3 build-hybrid.py "$assets" --verify-assembly` to hash the assembled stream and verify the draft without storing a second GGUF. A lightweight `--headers-only` check verifies the source sizes, header hashes and output layout. [File, header and PLE range pins](pins.json)

To serve, pass these paths to the release’s `run-server.py` in order.

- `$assets/model/halo-c1.gguf`
- `$assets/donor/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
- `$assets/donor/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf`

The original donor shard has the byte layout expected by the lazy PLE reader. The recipe gives an ordinary GGUF with the same bytes as the tested extent-backed image. The storage layout can affect loading behavior.

The weights and derivatives are covered by the included Qwen Community License 1.0. [License](LICENSE.qwen)
