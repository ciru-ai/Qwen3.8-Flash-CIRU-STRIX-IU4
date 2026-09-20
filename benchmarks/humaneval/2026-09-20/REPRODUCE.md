# Reproducing the selected IU4 configuration

Use the same Ciru Strix Halo hardware and the released IU4 v4.4.1 model, matching MTP head, PLE sidecar and runtime. `identity.json` records model/draft/runtime SHA256 hashes. `full-fastest/command.json` is the exact executed argv, and `full-fastest/runtime-env.json` records the performance-relevant environment. Remap filesystem paths only when moving machines. CPU/GPU clocks, thermals and other processes can change wall times.

Selected configuration: MTP maximum 6, Kairic Boost off, one slot, 4096 context, F16 target and draft KV, 8 CPU threads, full GPU offload. Requested batch and microbatch are 8192; the runtime may clamp these to context capacity. See the actual server log. Keep the recorded QSA and MTP shortlist settings.

On the existing Ciru installation, the launcher invocation is:

```sh
bash launch.sh 6 0 /srv/llm/work/iu4-reproduction/slot-state 4096
```

The exact full-suite inputs are `full-fastest/generation/HumanEval-N/request.json` for N=0 through 163, in order. Submit each once to `/v1/chat/completions` on localhost port 18090, waiting for the complete response before the next request. If authentication is configured, obtain the API key through the normal private environment; no key is included in this evidence bundle.

Use the exact unrelated warmup saved in `full-fastest/warmup-messages.json` once after a fresh server load. Exclude it from all reported measurements. The harness uses a streamed native completion for warmup and nonstream chat requests for measured tasks; `campaign.py` contains the exact implementation. Do not warm up on benchmark tasks or reuse prior answers.

Test parameters: HumanEvalPlus v0.1.10, EvalPlus 0.3.1, Python 3.11, standard retained OpenAI-chat instruction, thinking off, temperature 0, seed 0, maximum 1024 new tokens, n=1, no answer retry or repair. Request-level `cache_prompt:false` and slot 0 override the server's prompt-cache default. Verify cached token counts are zero. The complete request files also fix top-p, top-k and penalties.

Wall time is measured with a monotonic clock immediately around HTTP submission through receipt of the complete response. It includes request processing, prefill and generation, but excludes server loading, the warmup, memory-sampler shutdown and grading. Run the model alone during generation; grade afterward with `score-sandbox.py` and `score.py` using the saved public dataset. Grading sanitizes code with native EvalPlus and uses base plus extended tests, minimum time limit 1 second and ground-truth factor 4. The sandbox has no network or private files.

`campaign.py` plus `launch.sh`, the scoring scripts, and `sources/` are the complete campaign implementation. Existing completed arm results are resumed; for a fresh campaign use a new empty output directory containing these scripts and dataset, not this completed evidence directory. HE0–9 selected the settings and is included in the standard 164-task score; disclose that selection. Do not mix superseded warmup measurements into the reported results.

The original comparison author published aggregate wall times but did not release exact timing code, prompts or the harness. These artifacts reproduce our local protocol; they cannot establish undisclosed details of the author's execution.
