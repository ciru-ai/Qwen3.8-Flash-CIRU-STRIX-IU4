# Flash-attention mask bounds regression (HIP)

This standalone GPU test includes the production `fattn-common.cuh` kernel.
It checks 840 shapes and 795990 integer tile limits against an independent
scalar reference. Cases cover partial/full query tiles, all-masked input,
sequence broadcasting, tight allocations, poisoned padding and row strides.
The test requires an AMD GPU and the same compatible HIP toolchain as GGML.
It does not load model weights or start a server.

From the source root, using the compiler from the HIP SDK:

```sh
/path/to/sdk/lib/llvm/bin/clang++ -O3 -std=c++17 \
  -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 \
  -Iggml/src -Iggml/include -Iggml/src/ggml-cuda \
  --offload-arch=gfx1151 --hip-link --rtlib=compiler-rt -unwindlib=libgcc \
  -x hip tests/test-fattn-mask-bounds.cu \
  -L/path/to/sdk/lib -lamdhip64 -o test-fattn-mask-bounds
./test-fattn-mask-bounds
```

Use the compatible SDK runtime loader/library path required by your system.
Replace `gfx1151` only when testing another supported AMD target.
Expected result: `PASS: 840 cases; 795990 exact KV tile limits`.

The observed production failure had 2044 live query rows in a tile width of16.
The old scanner read the four rounded-up rows beyond the view. A preserved
negative control uses safely allocated finite poison rows to demonstrate the
old numerical failure without intentionally triggering another GPU fault.
The source fix does not change attention values, visibility or model weights.
