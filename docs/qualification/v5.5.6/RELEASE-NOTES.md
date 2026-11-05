# CIRU Flash runtime 5.5.6

Runtime 5.5.6 automatically selects the existing R4 expert kernels for eligible 9-16-row C1 batches. These batches previously selected padded grouped kernels. Existing backend, geometry, layout, stride and alignment checks determine applicability; no new environment switch or profile option is required. Model weights remain v5.0, and existing serving defaults are retained.

The source is available from the canonical GitHub repository's main branch and the v5.5.6 tag. The matching Hugging Face source archive is source-tree-flash-v5.5.6.tar.gz. Follow the current README and INSTALL instructions, rebuild the runtime after updating source, and keep its executable and libraries together.

Validation reuses the recorded automatic-dispatch correction: 47 CPU dispatch checks pass, and the native incremental build preserves all 150 HIP device-text images exactly against the earlier text-qualified build. The earlier single-slot and four-concurrent text panels completed all five requests with retained deployment settings. These are scoped functional checks; published-default, vision, broad quality and matched performance results were not newly measured for this patch. [Automatic-dispatch validation](../v5.5.1/AUTOMATIC-DISPATCH.json) and [earlier build/text checks](../v5.5.1/BUILD-AND-FUNCTIONAL.json) preserve their exact original identities.

The retained complete expert-operation comparison was 4.701845 to1.536682ms on one captured T16 operation. This is a component measurement, not a whole-model speed ratio. A separate small-prefill decoder candidate produced only 0.37% complete-stage improvement and is excluded. The optimization method is documented in [OPTIMIZATION.md](../../../OPTIMIZATION.md).
