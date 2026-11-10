# Runtime v6.0.0: faster multi-slot serving, disk-backed prompt cache, image fix

v6.0.0 is a runtime release for both Flash IU4 and Orca. Model weights, MTP
drafts, PLE files and projectors are unchanged. Rebuild the runtime and restart
the server to apply it; updating a launcher or documentation alone is
insufficient. One slot remains the default.

## 1. Image input in long prompts (fix)

Reported by Hugging Face user jack3149813146 in
[Orca discussion #5](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-Orca/discussions/5): with v5.5.0, a request that carried a
photograph and eight or more tool definitions returned tool calls that went
looking for the image instead of describing it.

The trigger is prompt length, not tools. Once the prompt reaches about 2,048
tokens, image rows select their attention cells through a 2,051-wide sort.
v5.5.0 added that sort and a GPU kernel able to run it, but the HIP backend
still declared sorts wider than 1,024 columns unsupported, so the step fell
back to the CPU in the middle of the image pass and the selection that came
back was wrong. The model then behaved as if the image were blank. Plain text
padding with no tools reproduces it: at 2,542 prompt tokens v5.5.8 answers
that the image is "completely blank".

v6.0.0 lets the HIP backend run the wide sort on the GPU whenever the padded
row fits shared memory. The kernel output was checked on the host for every
row of the failing requests (2,051 columns, 512/512/56-row passes): each row
was a complete permutation in ascending order.

Check: four real photographs, one question ("what is in this picture?"),
temperature 0, thinking off, Orca v5.0 weights, one slot, MTP4, Kairic Boost
on. A request passes when the reply names the subject and makes no tool call.

| Tool definitions | Prompt tokens | v5.0.1 | v5.5.0 | v5.5.8 | v6.0.0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,110 | 4/4 | 4/4 | 4/4 | 12/12 |
| 4 | 1,853 | 4/4 | 4/4 | 4/4 | 12/12 |
| 8 | 2,509 | 4/4 | 0/4 | 0/4 | 12/12 |
| 16 | 3,821 | 4/4 | 0/4 | 0/4 | 12/12 |
| 32 | 6,415 | 2/4 | 0/4 | 0/4 | 6/12 |

v6.0.0 was run three times per photograph (12 requests per row); the other
columns once. At 32 tool definitions the two photographs that miss on v6.0.0 are the
same two that miss on v5.0.1: the model chooses a file-search or image-reading
tool that the list offers. That is model behaviour with this tool list, not
the defect. Flash IU4 on v6.0.0 passed 4/4 at every tier including 32.
With text padding only, v6.0.0 describes the photograph correctly at
2,542, 4,042 and 24,002 prompt tokens.

Affected releases: v5.5.0 through v5.5.8. v5.0.1 is not affected.
The harness is included as [vision-tools-sweep.py](vision-tools-sweep.py).

## 2. Multi-slot decode speed

In a shared pool, v5.5.x decoded one slot per pass, so total throughput with
four busy slots was about the same as one slot, and every pass also scanned
index rows that belonged to the other slots.

- **Per-sequence index cache.** Each pooled sequence keeps its own incremental
  index instead of refreshing against the whole pool.
- **Shared passes.** Slots that are generating are decoded in one pass, each
  with its own visibility. Matrix products, the sparse-attention decode and the
  embedding row sums are kept on the same kernels and thread layouts as a
  single-sequence pass, so the arithmetic does not change with the number of
  slots in the pass.
- **Own rows only.** Each slot scores only its own cached index rows.
- **Indexed rollback.** Removing rejected draft tokens looks up the affected
  cells in the per-sequence position index instead of scanning every cell of
  the pool. On the four-slot test this was 19% of the server's busy thread.

Measured on Ciru (Ryzen AI Max+ 395 / Radeon 8060S, NixOS, TheRock 10.0), IU4
v5.0 weights, shared 393,216-token pool, 262,144 per request, microbatch 2,048:

| Case (four pooled slots, MTP3, thinking on) | v5.5.8 | v6.0.0 |
| --- | ---: | ---: |
| 4 slots generating, about 240k tokens pooled, total | 21.4 tok/s | 49.9 tok/s |
| 4 slots generating, about 328k tokens pooled, total | 11.3–12.1 tok/s | 42.0–42.6 tok/s |
| 2 slots generating, about 328k tokens pooled, total | 11.5 tok/s | 37.3 tok/s |
| 1 slot generating, about 328k tokens pooled | 12.6 tok/s | 24.5 tok/s |

The 240k row is the final v6.0.0 code. The 328k rows were measured on the
v6.0.0 code before the indexed-rollback change; that change only reduces host
time. Each row is one run of 30–75 seconds after prefill, not a confidence
interval.

**Same output.** With greedy sampling, four concurrent slots on v6.0.0
produced the same tokens and the same top-5 log-probabilities as the
one-slot-per-pass decode: 300/300 tokens on each of four slots at 20,000
tokens per slot (maximum log-probability difference 0), and 300/300 tokens on each of four slots (maximum log-probability difference 0) at
100,000 tokens per slot.

**One slot is unchanged.** The default single-slot profile measured
34.9 tok/s at 20,000 tokens and 33.4 at 100,000 on v5.5.8, and
35.5 and 34.1 on v6.0.0. A lone request in a multi-slot pool is still
slower than the single-slot profile, partly because the single-slot draft
shortlist stays disabled when more than one slot is configured.

The busy server thread stays near 100% of one core in both versions. In v6.0.0
most of it is the GPU runtime waiting for kernels to finish; engine code is
about 10% of that thread on the four-slot test, down from about 27% before the
indexed rollback.

## 3. Prompt cache on disk (optional)

A saved conversation state is about 30 KB per token, so states larger than the
RAM cache (`PROMPT_CACHE_MIB`, 1,024 MiB by default) were dropped and the next
request for that conversation re-read its whole prompt. With more conversations
than slots this dominated a real agent workload: 3.3 hours of a six-hour run
were spent re-reading 2.98 million prompt tokens.

`PROMPT_CACHE_DISK_MIB=<MiB>` adds a disk tier. States that do not fit in RAM
are written to `PROMPT_CACHE_DISK_DIR` (default: `prompt-cache/` under the slot
directory) and restored when the conversation returns. The oldest files are
removed when the budget is exceeded, and the directory is emptied when the
server exits. It is off unless the variable is set.

Check: five conversations of 65,000–92,000 tokens over four slots (states of
2.2–3.0 GB). Returning to an evicted conversation re-read 55 tokens and took
2.9–3.6 s. Without the disk tier the same pattern re-read the whole prompt:
37–56 s for 29,000–44,000 tokens. Continuations after a restore matched a run
in which the conversation was never evicted (48/48 tokens, four conversations).

## 4. Smaller changes

- **Prompt reads share the GPU.** While another slot is generating, a prompt
  read is limited to 2,048 tokens per iteration (`CIRU_PROMPT_SHARE_TOKENS`).
  In an eight-minute churn test with six clients on four slots, the longest
  stall seen by a generating slot fell from 26 s to 3.2 s.
- **Rollback after a restore.** A restored recurrent state does not carry the
  per-token rollback snapshots. A small rollback right after a restore is now
  refused and the server falls back to a checkpoint instead of reading stale
  snapshots.
- **Pool admission (experimental, off).** `CIRU_POOL_ADMISSION=1` makes a new
  request wait or evict idle history when the pool cannot hold it, instead of
  failing running requests. It did not trigger in any test here.

## Settings

With more than one slot the launcher now sets `CIRU_QSA_BATCH_SEQS=4`,
`CIRU_EXACT_BATCH_MAX=32`, `CIRU_QSA_OWN_ROWS=1` and
`CIRU_PROMPT_SHARE_TOKENS=2048`. Setting any of them to `0` restores the
v5.5.8 behaviour for that feature. One-slot launches do not use them.

## Limits of these checks

- No broad model-quality benchmark was rerun. The exactness checks compare
  v6.0.0 with the one-slot-per-pass decode of the same build under greedy
  sampling.
- Outputs of long passes depend on where a sequence sits in the pool, in
  v5.5.8 as well; compare concurrent runs with concurrent runs.
- Speed was measured through the raw completion endpoint with text prompts.
  Image requests were checked for correctness, not speed.
- The largest state restored from disk was 3 GB, read back shortly after it
  was written. Cold-read time from NVMe is unmeasured.
- The rollback-after-restore guard and pool admission have no targeted test.
- More than four slots is untested. Other GPU architectures are untested.

See [CHECKS.json](CHECKS.json) for the recorded results.
