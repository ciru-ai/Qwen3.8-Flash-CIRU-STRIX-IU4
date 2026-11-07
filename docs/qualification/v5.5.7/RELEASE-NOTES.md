# Runtime v5.5.7: recurrent-state reset correction

v5.5.7 fixes a reset defect that can preserve poisoned recurrent state across
slot reuse. `build_rs` previously multiplied the old state by zero; NaN and
infinity do not become zero through that operation. It now writes literal
zeros with `ggml_fill_inplace`. Model, draft, PLE and projector files are unchanged.
Rebuild the runtime and restart the server to apply the correction. Updating
only the launcher or model card does not install it.

The OpenVINO decoder and FILL translator also recognize this reset, preserve
neighboring slots, and compute the reset slot from byte strides. This keeps
the new operation compatible with that backend's cached-state bookkeeping.
The existing rollback test now includes poisoned recurrent buffers and fails
if an architecture declares recurrent storage but the fixture cannot find it.
Architectures without recurrent storage explicitly skip that poisoning case.

## Validation

- Original-versus-corrected real-model CPU regression: the original library
  produces nonfinite logits after poisoned-state reuse; the correction passes.
- Actual reset graph on the release HIP backend: the original multiply reset
  fails the poisoned-state check; literal FILL passes all nine checks,
  including neighboring slots, nonzero offsets and an empty reset view.
- Full expanded CPU suite: all 64 tests ran without exclusions. There were
  61 substantive passes and three baseline-matched exceptions: a model-dependent
  backend-sampler assertion, unsupported PLaMo3 inside the vocabulary script,
  and an HTTP 401 for unavailable Nemotron metadata. The available quantization
  checks pass 330 cases. CTest exits 8; this is not an all-green suite.
  No additional CPU regression was observed on the matched fixtures.
- OpenVINO 2026.4.1 CPU plugin: the actual backend builds; 24 reset cases,
  12 legacy SCALE compatibility cases and eight generic FILL cases pass.
  Coverage includes static/dynamic shapes, zero/nonzero slot offsets,
  nonfinite state, neighbor preservation and F32/F16 generic FILL.
- A corrected core library was checked on the existing Strix Halo HIP runtime
  with four unified slots, text, thinking, images, cancellation/reuse and a
  resumed long OMP session. This is an incremental-library qualification,
  not a new complete prebuilt runtime package or a repeat of performance and
  broad model-quality benchmarks.

The reported incident included repeated output and nonfinite probabilities.
The defective reset is independently reproduced and repaired, but the first
operation that produced nonfinite state in that incident remains unidentified.
This release fixes reset recovery; it does not claim to prevent every source
of NaN/Inf or every repetition loop. Intel GPU/NPU and other untested backends
have no new qualification from these checks.

Machine-readable scope and results: [CHECKS.json](CHECKS.json).
