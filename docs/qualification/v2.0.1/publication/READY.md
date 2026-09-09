# QSA v2.0.1 release readiness

The focused QSA release is qualified and packaged. No public upload or production deployment has been performed. Ciru's GPU is released; Dunamis's build containers are stopped. Sozo's unrelated model service was left alone.

| Check | Result |
| --- | --- |
| QSA mapping/state and canonical equivalence | 66 cases; 288 exact four-tensor comparisons |
| Two-slot target-only serving | 16/16 recall requests across separate/unified KV, plus full-model isolation/copy controls |
| Final 8K matched pair on Sozo | Prefill +4.70%, generation +1.20%; exact tokens/counts |
| Final 64K matched pair on Ciru | Prefill +2.12%, generation -0.45%; exact tokens/counts |
| Clean Ubuntu 24.04 / ROCm 10 build | Passed; 66 QSA cases and 30 batch tests / 198 assertions |
| Ubuntu GPU smoke | All 520 tokens and 674 drafted / 400 accepted counts match; all 15 binary hashes verified |
| Packaging | 3,593 Git paths verified, 606 evidence-manifest entries, full and runtime-only checksums pass |

The clean build ran on Dunamis's CPU. Ciru performed one candidate/baseline 64K pair and one clean-build coding completion. No additional model run was made to correct the overly strict GPU-label verifier. Its original failure and independent adjudication are preserved.

The earlier Sozo 64K loss remains unexplained. Final timing and hardware telemetry support this scoped release; the profiler attachment failed, so no kernel-level explanation was obtained. Multi-slot MTP remains unsupported. The HIP host-buffer patch, Kairic and HC-mix are excluded. No filled-512K, universal-agent, new full-quality-suite or native-Ubuntu-driver claim is made.

## Source and delivery

- Local branch: `codex/qsa-sequence-fix`.
- Qualified source commit: `9ea2390a71ae9f3d1cab519bbe099eb4ee06380e`.
- Git tree: `e060b9fce2a1ad3abee390fffe1c33ba02945e3b`.
- Local annotated tag: `v2.0.1`, pointing to that commit; not pushed.
- GitHub release assets: [release-ready/](release-ready/).
- Hugging Face publication stage: [hf/](hf/).
- [Package verification](PACKAGE-VERIFICATION.json), [qualification](../runtime/docs/QSA_BACKPORT_STATUS.md), [release notes](github-release.md), [community reply draft](COMMUNITY-REPLY.md).
- [Independent package review](INDEPENDENT-PACKAGE-REVIEW.json): 25 checks passed, zero blockers.
- Source archive SHA-256: `28b2194323ba1105921f2c52639cc28cce8b8336f4f0a3a54349c249823fa7c5`.
- Evidence archive SHA-256: `a032e4731db988b2b787563a4094ae662e73b3ad44488fda089c998998e1db66`.

Only publication remains: push the source commits and new tag, create the GitHub release with these verified assets, upload the staged Hugging Face changes and pin its new v2.0.1 revision, then verify remote commit identities and downloaded source/checksum bytes. Preserve all v2.0 tags, archives, checksums and weights. The older prepared scripts are superseded by the final staging/packaging workflow and must not be rerun.

No further GPU test is required for these unchanged, qualified source bytes. A later inference-code change or materially different combined patch requires its own qualification.
