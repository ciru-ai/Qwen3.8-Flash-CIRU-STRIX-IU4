# Runtime v5.5.8: graph-pool lifetime correction

The legacy HIP/CUDA temporary-buffer pool can physically free memory when its
cache is full, or clear cached buffers after an allocation failure before
retrying. Previously, captured graphs owned by that backend context could
retain references to the freed memory. The pool now clears its owning context's
graph cache at both physical-free paths, so subsequent eligible graph execution
captures again. Returning a buffer to the cache without freeing it preserves
the graphs. Other contexts' graphs are preserved.

This backports Bokros Balint's upstream llama.cpp PR #26771, commit
`4fd70228390839dba220928904f87b18ed9ec69f`. The original author, date and patch
are retained in [UPSTREAM.patch](UPSTREAM.patch). The source changes are limited
to `ggml/src/ggml-cuda/common.cuh` and `ggml/src/ggml-cuda/ggml-cuda.cu`.
The v5.5.7 literal-zero recurrent-state reset, OpenVINO compatibility changes
and recurrent-state regression remain unchanged. Model weights, draft, PLE and
projector files are unchanged. Rebuild the runtime and restart the server to
apply the correction; updating documentation or a launcher alone is insufficient.
HIP graphs remain enabled. No new serving option is required.

## Validation

- The included portable CPU fixture extracts the actual legacy pool and factory
  from the selected source tree and substitutes only the device-allocation
  boundary. On published v5.5.7, cached reuse passes but both physical-free
  invalidation checks fail with graphs enabled. On this release all three checks
  pass. All three graph-disabled controls pass on both versions.
- A native HIP lifecycle check using tiny owned buffers and graph handles passes
  cached reuse, allocation-failure flush and full-cache physical free, including
  owner-context isolation and cleanup. This check launches no graphs or kernels.
- The normally rebuilt gfx1151 HIP library passes public GGML F16 and F32
  arithmetic against a CPU-double reference over 32 graph computations each.
  Both cases reach graph warmup with graphs enabled; all outputs are finite,
  with zero mismatches. Maximum absolute errors are 2.98023224e-08 and zero.
- The complete HIP target and its prerequisites were rebuilt from source using
  the previously qualified compiler and configure flags, including gfx1151,
  HIP graphs, NO_VMM and MMQ/MFMA. All 156 HIP objects were freshly built,
  including every private pool-factory dependent. Public C exports, SONAME,
  direct dependencies and GNU runtime requirements remain compatible. C1 device
  text is unchanged; other device-text differences are address literals.

These are targeted pool-lifetime, backend and build checks. The expanded CPU
suite and OpenVINO checks documented for v5.5.7 were not repeated for v5.5.8;
their source fixes are retained. No new broad model-quality or performance
benchmark is claimed, and other GPU architectures are untested here.

The defect is present in checked own release tags from v1.0 through v5.5.7.
Source exposure does not prove that every release encountered it in operation.
It is not specific to shared KV or multiple request slots. This correction
does not establish the cause of every repetition incident or identify the
first producer of nonfinite state in the earlier reported incident.

## Reproduce the CPU boundary check

From the repository root, with Python 3 and a C++17 compiler named `c++` in PATH:

```bash
python3 docs/qualification/v5.5.8/test-pool-graph-invalidation.py . /tmp/flash-pool-check
```

The output directory receives the extracted test, two CPU executables and a
JSON receipt. It makes no GPU calls. To observe the failing baseline, run the
same script against a separate v5.5.7 checkout, using a different output directory.
See [CHECKS.json](CHECKS.json) for the retained, sanitized result summary.
