# v4.4.1 qualification

PASS: 30 served checks on Ciru, NixOS/ROCm10, 262144 context, one slot, F16 target/draft KV, IU4 MTP3 and Orca MTP4. Three greedy text controls per variant match v4.4.0 exactly. Both variants pass single image, text after image, slot reuse with a second image, mid-conversation image plus earlier-number recall, two images, a 1024x512 image, cached-image repeat, and final text. Logs contain no speculative processing errors. These are quick compatibility checks, not a full quality or performance benchmark.

State tests also pass with ASan/UBSan: split pinned images, rectangular spatial bounds, checkpoint roundtrip, reset and invalid positions. Launcher checks accept the patched common library and reject a stale v4.4.0 runtime. GPU work ran under the exclusive host lock; the pre-existing failed main service was left unchanged.

The fix carries the dense target hidden row across image batches. Qwen4Exp MTP requires token IDs, so pinned image rows are skipped in the draft KV as in DFlash. The next text position must match one plus the maximum temporal/spatial position from the image; arbitrary forward gaps remain rejected. Split image batches retain their temporal anchor. Checkpoint state includes that anchor with a new format marker; release launchers use a separate slot-state directory.

Only Qwen4Exp's embedding path and its pending-state format change. Text processing, target verification, sampling, inference kernels and HIP/ROCr are retained. The common library was rebuilt from the release source; other package binaries match v4.4.0.

This is an independent implementation based on the supplied diagnosis. The author's three patch files and commits were not available.
