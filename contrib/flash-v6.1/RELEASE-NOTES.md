# Flash v6.1.0 for Strix Halo

Flash v6.1.0 adds a hybrid C1 profile for AMD Strix Halo. It uses the compact Ciru routed expert banks with Unsloth’s UD-Q4_K_XL Flash-Next tensors, the HaloBox b1064 runtime base and the donor Q8_0 MTP draft head. The existing v6.0.0 package and IU4 weights remain available.

The runtime includes tiled C1 weights, improved expert routing and gate/up scheduling, the hot-expert down tile, and fusion of the weighted expert merge with the following hyperconnection. It also includes the qualified recurrent, sparse attention, MTP catch-up, output readback, and prompt-cache changes for this profile.

On a Ryzen AI Max+ 395, the research build increased uncached prefill from **1,363.63 to 1,626.27 tokens/s (+19.26%)** versus the initial HaloBox b1064 C1 hybrid retested after the same Sozo kernel update. Both tests used the same target and draft weights, one slot, F16 KV, a 262,144-token context and MTP2. On one scored 4,096-input, 512-output request after the prescribed warmup, prefill finished 485.09 ms earlier and first text arrived 487.64 ms earlier. The 512-piece stream matched byte for byte and both runs accepted 331 of 360 drafts. Generation measured 42.22 → 42.33 tokens/s.

The reported speed applies to this hybrid profile and workload. v6.0.0's original IU4 profile used a different weight package and MTP depth. Use the hybrid GGUF and the matching runtime and draft head from the release instructions.

Thanks to Qwen for Flash-Next, Unsloth for the quantized donor and draft distribution, HaloBox for the Strix Halo runtime, and the Gufo contributors for recurrent and routed-compute work that informed this release. [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66) · [HaloBox](https://github.com/halo-box/strix-llama.cpp/tree/b9f3c0e878f82f80ea1528969f305017a909b73e) · [Gufo](https://github.com/gufo-org/gufo)
