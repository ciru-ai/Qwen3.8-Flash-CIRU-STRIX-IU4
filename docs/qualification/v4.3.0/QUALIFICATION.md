# CIRU runtime v4.3.0

V4.3 makes the tested speculative-decoding settings the defaults: **MTP depth 3 for IU4, depth 4 for Orca, and `LLAMA_MTP_QSA_MIN_T=1` for both**. Shorter draft batches reduce discarded speculative work; the lower threshold enables sparse draft attention for small batches. The corrected v4.2 inference binaries and existing model files are retained.

| Model | Hermes mean score /100, passes 1 / 2 | Full-score tasks, passes 1 / 2 | Prompt tok/s | Generation tok/s |
| --- | ---: | ---: | ---: | ---: |
| IU4 | 95.0 / 98.5 | 17/20 / 19/20 | 548.5 | 34.74 |
| Orca | 92.5 / 98.5 | 17/20 / 19/20 | 551.3 | 34.44 |

Hermes Agent 20: two passes per model, 80 completed scenario attempts and 449 model requests, a 32-turn allowance, production sampling and native xhigh thinking. Speeds are pooled across each model's two passes; generation includes reasoning. Both hosts are 128 GB Strix Halo / gfx1151, NixOS / ROCm 10; IU4 ran on Sozo and Orca on Ciru. These are observed tuned-configuration results, not a matched Hermes speedup over v4.2. Native verifier scores are retained; the detailed report records grading limitations and Orca's memory-persistence behavior.

Existing v4.2 users can update the launcher without rebuilding or downloading weights. `MTP_DEPTH` and `LLAMA_MTP_QSA_MIN_T` remain overridable; `MTP_DEPTH=6 LLAMA_MTP_QSA_MIN_T=128` restores the previous speculative defaults. Earlier runtimes must first obtain the v4.2 attention correction. Context remains 262144, batch/microbatch 8192, with F16 target/draft KV and unchanged sampling and thinking.

[Hermes protocol, task results and grading notes](HERMES-RESULTS.md).
