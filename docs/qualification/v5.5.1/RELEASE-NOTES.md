# CIRU Flash runtime v5.5.1 candidate

Status: built candidate with scoped text functional checks, not a published release. This tree is the verified v5.5.0 source package plus automatic small-batch C1 dispatch. Model weights remain v5.0. Source builds use the existing `scripts/ciru/build-linux-amd.sh` and `scripts/ciru/run-server.sh` workflow. Private source and native transfer archives have been built. Public source/binary publication and model-card updates remain pending; no new public downloadable artifact is claimed here.

## Change

The AMD C1 dispatcher can use its existing R4 kernels for eligible joint batches of 9 through 16 rows. Previously those batches selected the padded grouped expert path. Both the grouped bypass and vector admission use one predicate. Kernel instruction bodies are unchanged.

Eligibility requires the AMD backend, the existing R4 or R4-LUT route, 512 experts, a 2560-to-640 or 640-to-2560 projection, compatible activation channels and the original contiguity, stride and alignment guards. Unsupported layouts retain fallback behavior. Batches up to eight rows and batches above sixteen retain their existing dispatch.

The dispatcher selects the improved path automatically when those requirements apply. There is no new environment option or profile setting for this correction. MTP depth, context, slot count, model weights, sampling and vision defaults are unchanged. The default remains one slot.

## Retained evidence and limits

On one captured four-sequence, sixteen-row expert operation, complete-operation time changed from 4.701845 ms to 1.536682 ms (3.06x). This included routing, packing, projection, activation, weighting and reduction. It is a component result, not a whole-model speed ratio.

A captured three-worker conversation showed foreground native decode changing from 7.93 to 13.21 tokens/s and workers from 7.10-7.82 to 11.71-12.26 tokens/s. Outputs differed, so elapsed completion times do not constitute an equal-output throughput comparison.

T8 output was exact; complete T16 operation relative L2 was approximately 5.49e-6. A limited four-sequence full-model panel retained 16/16 top choices, with mean KL approximately 0.0001973 and maximum total variation approximately 2.76%. Twelve positions were predictable JSON endings. These are narrow numerical checks, not broad task-quality or stochastic-acceptance equivalence.

This source candidate excludes application changes and foreground-prefill scheduling. Full-workflow timings from combined deployments are not attributed to this patch. The earlier switch-enabled implementation used for the retained arithmetic measurements had source SHA256 `fa59e9848f5c120e6a7deb9d2e2729669536fffaba990d3068eb71a21e35842f`.

## Candidate validation

Run `python -m pytest tests/test-c1-small-batch-dispatch.py -q` with pytest and a C++17 compiler. Set `CXX` to choose the compiler. The tests compile the actual production host predicates against tensor-metadata fixtures and exercise both projection directions, admission boundaries, invalid layouts, R4 selection and automatic behavior with both clean and legacy environments. They do not execute GPU kernels or validate their arithmetic.

The automatic-selection regressions failed 14 cases against the prior switch-based candidate and now pass all 47. The original package baseline's earlier 7-failure/34-pass result remains historical. `CHECKS.json` records current checks. [Automatic-dispatch validation](AUTOMATIC-DISPATCH.json) records the native incremental rebuild and exact device-text comparison: all 150 HIP device images match the previously qualified build.

The earlier full native build completed from commit `51dd9ffac09c78794e45f9a321c50c60cda3c21e`. Its immutable [build and functional evidence](BUILD-AND-FUNCTIONAL.json) records one single-slot panel and one four-concurrent-request panel: all five responses returned HTTP 200, natural stop and nonempty content. That build used the prior switch-enabled host dispatcher. The current revision removes only that host switch and does not repeat those GPU requests. Contexts were 262144 and four times 65536.

These text checks retained deployment microbatch 2048 and two checkpoints per slot, with no image projector. They do not requalify published single-slot defaults (microbatch8192 and32 checkpoints), vision, broad task quality or speed. Two setup attempts ended before inference. A later cleanup command against an already-unloaded temporary unit returned nonzero; independent readback confirmed the original working environment and source hashes restored. No permanent runtime deployment or public release was made.

The earlier compiled source and transfer archives remain immutable. The automatic-dispatch revision changes the host predicate, restores the original production profile and updates regressions; its separate validation record identifies the rebuilt binaries. No new GPU timing or model-quality result is claimed.

`SOURCE_IDENTITY.json` now includes both production profiles and the dispatch regression test in addition to the previous compiled/build inputs. `checksums/v5.5.1-source.sha256` covers those files and current release documents. The unversioned historical `checksums/SHA256SUMS` is retained as prior model evidence; it is not the candidate source inventory.
