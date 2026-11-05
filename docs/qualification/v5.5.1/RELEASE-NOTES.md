# CIRU Flash runtime v5.5.1 candidate

Status: prepared source candidate, not a published release. This tree is the verified v5.5.0 source package plus the small-batch C1 dispatch change and its production-profile switch. Model weights remain v5.0. Source builds use the existing `scripts/ciru/build-linux-amd.sh` and `scripts/ciru/run-server.sh` workflow. New source archives, binary packages and model-card publication are separate pending release work; no new downloadable artifact is claimed here.

## Change

The AMD C1 dispatcher can use its existing R4 kernels for eligible joint batches of 9 through 16 rows. Previously those batches selected the padded grouped expert path. Both the grouped bypass and vector admission use one predicate. Kernel instruction bodies are unchanged.

Eligibility requires the AMD backend, the existing R4 or R4-LUT route, 512 experts, a 2560-to-640 or 640-to-2560 projection, compatible activation channels and the original contiguity, stride and alignment guards. Unsupported layouts retain fallback behavior. Batches up to eight rows and batches above sixteen retain their existing dispatch.

The shared production profile enables `LLAMA_C1_VECTOR_R4_SMALL_BATCH=1`. Set `LLAMA_C1_VECTOR_R4_SMALL_BATCH=0` before the existing launcher to restore the prior grouped selection. Native direct callers remain opt-in unless they load that profile. MTP depth, context, slot count, model weights, sampling and vision defaults are unchanged. The default remains one slot; this change does not turn the release into a four-slot default.

## Retained evidence and limits

On one captured four-sequence, sixteen-row expert operation, complete-operation time changed from 4.701845 ms to 1.536682 ms (3.06x). This included routing, packing, projection, activation, weighting and reduction. It is a component result, not a whole-model speed ratio.

A captured three-worker conversation showed foreground native decode changing from 7.93 to 13.21 tokens/s and workers from 7.10-7.82 to 11.71-12.26 tokens/s. Outputs differed, so elapsed completion times do not constitute an equal-output throughput comparison.

T8 output was exact; complete T16 operation relative L2 was approximately 5.49e-6. A limited four-sequence full-model panel retained 16/16 top choices, with mean KL approximately 0.0001973 and maximum total variation approximately 2.76%. Twelve positions were predictable JSON endings. These are narrow numerical checks, not broad task-quality or stochastic-acceptance equivalence.

This source candidate excludes application changes and foreground-prefill scheduling. Full-workflow timings from combined deployments are not attributed to this patch. The retained measured C1 source SHA256 is `fa59e9848f5c120e6a7deb9d2e2729669536fffaba990d3068eb71a21e35842f`.

## Candidate validation

Run `python -m pytest tests/test-c1-small-batch-dispatch.py -q` with pytest and a C++17 compiler. Set `CXX` to choose the compiler. The tests compile the actual production host predicates against tensor-metadata fixtures and exercise both projection directions, admission boundaries, invalid layouts, R4 selection, profile default and opt-out. They do not execute GPU kernels or validate their arithmetic.

The unchanged baseline fails seven cases and passes 34; the candidate passes all 41. `CHECKS.json` records these CPU checks and the existing AMD build configuration preflight. A full candidate HIP build, clean-package serving validation and remaining release-boundary quality/performance qualification are pending. No service or installed profile has been replaced by staging this source.

`SOURCE_IDENTITY.json` now includes both production profiles and the dispatch regression test in addition to the previous compiled/build inputs. `checksums/v5.5.1-source.sha256` covers those files and current release documents. The unversioned historical `checksums/SHA256SUMS` is retained as prior model evidence; it is not the candidate source inventory.
