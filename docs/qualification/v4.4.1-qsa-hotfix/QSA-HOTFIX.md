# QSA vision/MTP source hotfix for v4.4.1

Fixes `qsa: cell position runs past the cell window` when MTP draft cache positions extend beyond its physical cell window after image input. MTP and sparse attention remain enabled. Ordinary position lookup is unchanged; overflow logical buckets use a separate ordered map, including tail lookup.

This is a **source hotfix requiring a rebuild**. Existing v4.4.1 prebuilt runtime archives do not contain it. Model weights and projectors are unchanged. The prior v4.4.1 tag remains unchanged.

## Apply

Download `vision-mtp-qsa-gap.patch` and, from the v4.4.1 source checkout, run:

```bash
patch --dry-run -p1 < vision-mtp-qsa-gap.patch
patch -p1 < vision-mtp-qsa-gap.patch
cmake --build build-gfx1151-sdk --target llama -j 4
```

Use your existing SDK build environment and build directory, then restart the server with the rebuilt library. MTP stays enabled. A fresh checkout of tag `v4.4.1-qsa-hotfix` already contains the fix and should not be patched again.

## Checks

- Released v4.4.1 reproduces the exact assertion with synthetic position gaps; the fix passes.
- 340 ASan/UBSan mapping cases and 48 native runtime mapping cases pass.
- All output tensor bytes match the original in 1,584 valid in-window cases, including fragmentation, position gaps, sequence separation and ranked image positions.
- The translation unit compiles with the production HIP build flags. Relinking the unchanged release objects reproduces the old library byte-for-byte; a candidate changes only libllama.
- Quick CPU metadata benchmark: nine alternating rounds for each of six shapes, 4K/32K/256K cells and 4/128 queries. No slowdown observed; median time was 3.1-4.4% lower in this run. This is not an end-to-end tokens/sec or GPU performance result.

The reporter's exact request/image was not provided. No new served-model or quality benchmark was run for this source hotfix. The regression tests establish the indexing fix and unchanged existing metadata outputs, not a zero-regression guarantee.
