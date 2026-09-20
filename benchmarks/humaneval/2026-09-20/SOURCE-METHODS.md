# What the supplied sources actually disclose

The main coding benchmark is **EvalPlus HumanEval+**, covering all **164 HumanEval problems**. The model card specifies `enable_thinking=false`, `temperature=0.0`, and `max_tokens=1024`, and reports base pass@1, plus pass@1, and mean/median seconds per problem. Base and plus test the same generated answers with different test sets.

[Reddit post](https://www.reddit.com/r/StrixHalo/comments/1wl3weh/i_benchmarked_12_quantizations_of_qwen38flashnext/)

[Heretic2 model card](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF/blob/b7a350f3ba4c29962cefe9e69af225f142cda9bc/README.md)

[Halogen assessment](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF/blob/b7a350f3ba4c29962cefe9e69af225f142cda9bc/halogen-assessment.md)

## Published llama-server setup

- Ryzen AI Max+ 395, Radeon 8060S/gfx1151, 128 GB unified memory, Ubuntu.
- Vulkan, nicoboss llama.cpp `qwen4exp` branch; the card identifies build 10656 / commit 035e22731 for quantization, without explicitly pinning every inference run to that build.
- Published launch: `-ngl 99 -fa on -c 65536 -np 1 --jinja -dev Vulkan0`, with the relevant model and optional projector.
- Heretic2 IQ4_XS NGQ4 result: **84.1% base / 79.3% plus**, mean **4.16 s**, median **3.31 s**.
- Separate throughput table: prefill 312/382/363/327 tokens/s at approximately 512/2K/8K/16K tokens; serial decode approximately 24 tokens/s. Prompts, repetitions, cache state, and precise timing method are not supplied.

## Reproduction limits

The linked repository's complete file list contains the model, projector, README, assessment, and Git attributes. It provides no evaluation script, dataset version, exact messages/system prompt, sanitizer settings, test timeouts, raw answers, per-task scores, sampling seed, or timing code. Accordingly, matching the visible settings does not establish identical prompting or evaluation implementation.

This run fills those missing choices with EvalPlus 0.3.1's standard OpenAI chat prompt, its `sanitize` routine, full HumanEvalPlus v0.1.10, and native correctness checks. Every choice is recorded in `PROTOCOL.md`, the scripts, and raw evidence. The exact upstream prompt implementation is in [codegen.py](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/codegen.py), [provider/openai.py](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/provider/openai.py), and [openai_request.py](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/gen/util/openai_request.py).

## Inconsistencies that matter to comparison

1. The Halogen assessment describes **forced thinking and a 2048-token cap** for its later runs; the initial 1024-token BYO run reportedly scored 48.8%. Those are different conditions from the nonthinking 1024-token llama-server headline, and do not isolate an overlay effect by themselves. Treat the overlay interpretation as the author's explanation, not a controlled causal result established by the table.
2. The Q6_K 88.4/81.7, Q4_K_M 82.3/75.6, and ROCmFP6 73.8/67.7 rows also appear verbatim in the author's separate [27B dense model card](https://huggingface.co/cygnal/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP-GGUF). The [Uncensored Flash card](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Uncensored-IQ4XS-NGQ4-GGUF) explicitly identifies its other reference quants as 27B dense, and describes its own run as greedy with a **4096-token** cap. This makes the claim that the entire combined table is twelve matched Flash-Next quantizations unclear. It does not establish which unpublished runs the author intended to reference.
3. The supplied model card table has seven rows; the Reddit excerpt has nine. Neither linked file contains the promised full twelve-variant raw result set.

Use the Heretic2 headline as a published comparison point, and describe the new Orca result as a fresh run under the disclosed settings plus documented standard EvalPlus choices. Do not claim an exact cross-author reproduction or attribute differences solely to quantization.
