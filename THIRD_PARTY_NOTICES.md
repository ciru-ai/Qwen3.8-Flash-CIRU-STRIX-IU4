# Third-party notices

This source repository is a derivative of `ggml-org/llama.cpp` and is distributed under the MIT license in the root [`LICENSE`](LICENSE). Existing source-file notices remain in place.

## Included code

- **llama.cpp / ggml** — Copyright the llama.cpp contributors; MIT License.
- **Qwen experimental and MTP runtime support** — adapted from work by Ryan Monsurate and the llama.cpp community; MIT License as contributed to llama.cpp.
- **AMD OPUS helper headers** under `ggml/src/ggml-cuda/e3-qr05-locked/` — copyright carlus.huang@amd.com, 2025–2026; MIT License in the source headers.
- **AMD `hip_minimal.hpp` helper** — copyright Advanced Micro Devices, Inc., 2024–2026; SPDX-License-Identifier: MIT.
- **Gufo Q5 kernels and expert prefill architecture** — the v5.0 Q5 short-batch path adapts Gufo's `SmallBatchKQuantExactFp32GEMMKernel` and Q5_K decoder. The wave64 path includes the pinned `WKQuantA8BlockedWmmaGEMMKernel` excerpt from commit `990fdce45640facffeb6d7a8268cb7d0f536d94a`. Gufo's MIT copyright and permission notice are retained in `ggml/src/ggml-cuda/q5_short.cu` and `ggml/src/ggml-cuda/donor_wkquant_kernel.cuh`.
- **Halogen and pwilkin Strix Halo work** — the fast-prefill breakthrough came from [Halogen / Peonist.ai](https://github.com/peonist-ai/halogen-flash-server); [pwilkin](https://github.com/pwilkin/strix-halo) published the open-source llama.cpp implementation and custom HIP/ROCr base adapted here.
- **HaloBox chunked GDN lead** — the v5.0 chunked GDN gate availability follows [HaloBox](https://github.com/halo-box/strix-llama.cpp) commit `c750386bb`, credited at the adaptation site in `src/models/qwen4exp.cpp`.

## Concurrency update

The original-batch-order fix for speculative hidden states backports [hthadicherla's llama.cpp PR #29019](https://github.com/ggml-org/llama.cpp/pull/29019), commit `4453b535fd15cd5b9d5ccb956ebd38dc325c98fc`, under the existing MIT terms. CIRU supplied the sequence-safe QSA integration, synchronized UMA input reuse, conservative optional context pool, and concurrent GDN dispatch qualification.

## Model artifacts

This GitHub repository does not contain the model weights. The model, draft, and PLE files on Hugging Face are governed by the Qwen Community License 1.0 included in that model repository. Runtime source remains under its own MIT terms.

## Trademarks and affiliation

Qwen and AMD names and logos are the property of their respective owners. CIRU is an independent community research project and is not sponsored or endorsed by Qwen or AMD.
