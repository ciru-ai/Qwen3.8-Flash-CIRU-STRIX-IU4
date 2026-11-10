> Runtime 5.5.6 automatically selects the improved C1 route for eligible small joint batches. The serving settings and historical measurements below are unchanged. See [5.5.6 patch notes](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v5.5.6/docs/qualification/v5.5.6/RELEASE-NOTES.md).

> **Runtime 6.0.0** changes how pooled slots decode. The section below is current; everything after "Historical record" keeps the v5.5.0 text and measurements unchanged. See the [6.0.0 notes](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/blob/v6.0.0/docs/qualification/v6.0.0/RELEASE-NOTES.md).

# Multi-slot serving in v6.0.0

One slot is still the default and still the fastest way to serve a single
request. Use several slots when requests overlap.

```bash
PARALLEL_SLOTS=4 CONTEXT_SIZE=393216 PROMPT_CACHE_DISK_MIB=65536 \
  MODEL_VARIANT=IU4 MODEL_RELEASE=5.0 ENABLE_UI=0 \
  MODEL_DIR="$MODEL_DIR" RUNTIME_DIR="$RUNTIME_DIR" \
  bash "$RUNTIME_DIR/scripts/ciru/run-server.sh" --kv-unified-per-slot 262144
```

This is the profile measured with the IU4 weights (MTP3): four slots sharing a
393,216-token pool, each request limited to 262,144 tokens, microbatch 2,048.
IU4 uses MTP3. `PARALLEL_SLOTS=2`
with the default 262,144-token pool also works. The slots share the pool; they
do not each get the full context at the same time.

What v6.0.0 does differently with more than one slot:

- Slots that are generating are decoded together in one pass. v5.5.x decoded
  one slot per pass, so four busy slots produced about the same total as one.
- Each slot scores only its own index rows instead of the whole pool.
- A long prompt read is cut into 2,048-token pieces while another slot is
  generating, so it no longer stalls that slot for the whole read.
- With `PROMPT_CACHE_DISK_MIB` set, conversation states too large for the RAM
  cache are kept on disk and restored, instead of being dropped and re-read.

| Case (four pooled slots, MTP3, thinking on) | v5.5.8 | v6.0.0 |
| --- | ---: | ---: |
| 4 slots generating, about 240k tokens pooled, total | 21.4 tok/s | 49.9 tok/s |
| 4 slots generating, about 328k tokens pooled, total | 11.3–12.1 tok/s | 42.0–42.6 tok/s |
| 2 slots generating, about 328k tokens pooled, total | 11.5 tok/s | 37.3 tok/s |
| 1 slot generating, about 328k tokens pooled | 12.6 tok/s | 24.5 tok/s |

Measured on Ciru (Ryzen AI Max+ 395 / Radeon 8060S, NixOS) with the IU4 v5.0
weights through the raw completion endpoint; each row is one run. Greedy output
and log-probabilities matched the one-slot-per-pass decode on all four slots.
The single-slot profile measured 35.5 tok/s at 20,000 tokens and 34.1 at
100,000 on v6.0.0, so a lone request is still best served by one slot. The
single-slot draft shortlist remains disabled with more than one slot, and
Kairic Boost remains limited to one slot.

| Variable | Default with several slots | Meaning |
| --- | --- | --- |
| `CIRU_QSA_BATCH_SEQS` | 4 | Slots decoded per pass; `0` decodes one slot per pass as in v5.5.8 |
| `CIRU_EXACT_BATCH_MAX` | 32 | Keeps shared passes on single-sequence arithmetic; leave on with the line above |
| `CIRU_QSA_OWN_ROWS` | 1 | Score only the slot's own index rows |
| `CIRU_PROMPT_SHARE_TOKENS` | 2048 | Prompt tokens per iteration while another slot generates; `0` removes the limit |
| `PROMPT_CACHE_DISK_MIB` | unset (off) | Disk budget for saved conversation states |
| `PROMPT_CACHE_DISK_DIR` | `prompt-cache/` in the slot directory | Where those files go; use fast local storage |
| `CIRU_POOL_ADMISSION` | unset (off) | Experimental: wait or evict idle history instead of failing when the pool is full |

Limits: more than four slots is untested; pool admission never triggered in
testing; the disk tier was tested with states up to 3 GB; no broad quality
benchmark was rerun. Two requests whose combined demand exceeds the pool can
still fail unless pool admission is enabled.

---

# Historical record: v5.5.0 multi-slot serving and image-memory update


Runtime **v5.5.0** retains the v5 model and runtime features and keeps **one slot as the default**. Use two slots when accepting a second request matters more than the best single-request speed. The target GGUF, matching MTP draft, projector and PLE files are unchanged.

## Launch settings

After following [INSTALL.md](INSTALL.md), start Flash IU4 with:

```bash
MODEL_VARIANT=IU4 MODEL_RELEASE=5.0 ENABLE_UI=0 \
  MODEL_DIR="$MODEL_DIR" RUNTIME_DIR="$RUNTIME_DIR" \
  bash "$RUNTIME_DIR/scripts/ciru/run-server.sh"
```

To opt into a two-slot pool, add `PARALLEL_SLOTS=2`:

```bash
PARALLEL_SLOTS=2 MODEL_VARIANT=IU4 MODEL_RELEASE=5.0 ENABLE_UI=0 \
  MODEL_DIR="$MODEL_DIR" RUNTIME_DIR="$RUNTIME_DIR" \
  bash "$RUNTIME_DIR/scripts/ciru/run-server.sh"
```

In the Orca package use `MODEL_VARIANT=Orca` and its own target, MTP draft and projector. The normal MTP depths remain 3 for IU4 and 4 for Orca. Keep `ROCM_ROOT` configured as described in the installation guide. These commands require a complete rebuild from the updated source; replacing only a launcher is insufficient.

| Setting | Default one slot | Opt-in two slots |
| --- | --- | --- |
| Context | 262,144 tokens | One shared 262,144-token pool |
| Logical batch / physical microbatch | 16,384 / 8,192 | 16,384 / 2,048 |
| Target and draft KV | F16 | F16 |
| MTP | On; single-slot draft shortlist available | On; full-vocabulary draft head |
| Retained checkpoint allowance | 32 | 4 per slot, 8 total |
| Checkpoint minimum spacing | 8,192 tokens | 8,192 tokens |
| Prompt RAM cache / PLE cache | 1,024 / 4,096 MiB | Same global budgets |
| Concurrent GDN option | Off | Off unless explicitly enabled |

**Multi-slot serving has a performance penalty.** It disables the single-slot speculative draft-head shortlist: the draft head expands from 32,768 to 248,320 vocabulary rows even if the other configured slot is idle. MTP itself continues to propose and verify tokens. The shared shortlist state is not safe to reuse across independent sequences. Pooling and scheduling may also contribute to the measured slowdown; the whole loss has not been attributed to the shortlist alone. Kairic Boost remains limited to one slot.

`CONTEXT_SIZE` sets the total context budget. With pooling, one request can borrow nearly the whole pool when its peer is idle; two active requests share that finite capacity. This is **not 262,144 tokens guaranteed to each request simultaneously**. The allocator can evict an idle history under pressure. There is no new admission policy that queues two active requests whose combined demand exceeds the pool; exhausting it can fail active requests.

`CONTEXT_POOL=0` selects fixed partitions; `CONTEXT_POOL=1` explicitly selects pooling. The default `auto` uses fixed storage for one slot and pooling for multiple slots. `CTX_CHECKPOINTS_TOTAL` changes the aggregate default checkpoint count allowance, divided equally across slots. `CTX_CHECKPOINTS` overrides the per-slot count. This checkpoint division is static; the KV pool is dynamically shared. Fewer checkpoints may require more replay when a conversation branches. `UBATCH_SIZE`, `BATCH_SIZE`, `CHECKPOINT_MIN_STEP` and `PROMPT_CACHE_MIB` retain explicit overrides. Four slots are an advanced option, not a recommended faster default.

## Matched 256K pool measurements

Recorded on October 6, 2026, on Ciru: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB unified memory, local NVMe, NixOS, ROCm/TheRock 10.0, IU4 v5.0 target, matching Q8 MTP3, F16 target/draft KV, eight threads, projector loaded but **no images in these requests**. Both arms used total context 262,144, batch 16,384, microbatch 2,048, fenced single-copy UMA, 4,096 MiB PLE and 1,024 MiB prompt cache. The optional concurrent GDN path was off. One fixed slot had 32 checkpoints; the two-slot pool had four per slot.

| Configured slots | Active requests | Aggregate prefill, tok/s | Aggregate decode, tok/s |
| --- | ---: | ---: | ---: |
| One fixed slot | 1 | 1,063.10 | 47.00 |
| Two pooled slots | 1 | 1,042.27 | 42.20 |
| Two pooled slots | 2 | 1,034.67 | 40.06 |

The idle-second-slot short decode change is **47.00 to 42.20 tok/s, -10.21%**. All 256 output tokens and the 240 drafted / 175 accepted counts match in that pair. This fails the experiment's 5% maximum single-request slowdown gate, which is why one slot stays the default. This is one paired observation, not a confidence interval. Because both arms used microbatch 2,048, it does not measure the difference between the launcher's 8,192 and 2,048 defaults.

Prefill used 4,096 uncached text tokens and one output per request. Decode used 1,024 uncached text tokens and 256 forced outputs per request. Each configuration had its prescribed unscored warmup and one scored run, with no retries. Raw token arrays used temperature 1, top-p .95, top-k 20, min-p 0, seed 12345 and ignore_eos. Prefill divides fresh tokens by simultaneous release through the last first token. Aggregate decode counts actual output token IDs in the common interval when all requests decode, excluding prefill. Individual rates use different intervals and need not sum to the common-window rate.

The same cold 258,048-token request completed at 891.34 versus 876.48 tok/s (-1.67%). After six 64-token appends and 20 seconds idle, both reused 258,428 tokens, processed 68 new tokens and generated 128. Resumed decode was 26.59 versus 25.37 tok/s (-4.58%). All **672 output tokens across 20 paired single-active groups** matched. Eight distinct text known-answer questions scored **7/8 in every configuration**, preserving the same JSON-question miss.

Two fresh 122,880-token histories were retained concurrently. After appends and idle, occupancy reached 123,583 tokens per slot, 247,166 total; aggregate resumed decode was 30.48 tok/s. A later 254,000-token request successfully reclaimed the idle peer's 123,583-token history. Long prefill can delay another request's decode: the first cold-fill decode overlapped the peer's lengthy prefill and took about 80.6 seconds. The long synthetic filler is a capacity test, not semantic accuracy evidence.

All 59 planned request groups completed once without swap, OOM, memory-limit or guard events. Minimum sampled available RAM was 23.104 GiB. This does not guarantee safety for arbitrary image or embedding working sets.

| Memory allocation | One fixed slot | Two pooled slots |
| --- | ---: | ---: |
| Total raw target/indexer/draft KV | 7.3125 GiB | 7.3125 GiB |
| Required recurrent MTP state | 450.28 MiB | 900.56 MiB |
| Checkpoint storage ceiling | 3.519 GiB | 0.880 GiB total |
| Loaded-idle available host RAM | 32.173 GiB | 29.727 GiB |

Checkpoints allocate as they are created. The old 32-per-slot setting would allow 7.038 GiB across two slots. Whole-host available-memory readings include workspace and other activity and are not isolated allocation measurements. GPU-visible UMA and process/host RAM overlap and must not be added.

## Earlier fixed-partition concurrency sweep

This separate diagnostic used **16,384 context tokens per slot**, total 16,384/32,768/65,536, and `CIRU_GDN_MULTI_TILE=1`. It is not the 256K default profile comparison and does not define a selectable 16K serving profile. The other short PP/TG shapes, sampling, eight threads, F16 KV, MTP3, batch 16,384 and microbatch 2,048 match the descriptions above.

| Active requests | Aggregate PP, tok/s | Aggregate TG, tok/s | Mean individual TG, tok/s | Mean prefill first-token latency | Minimum available RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,054.69 | 47.49 | 47.49 | 3.882 s | 37.724 GiB |
| 2 | 904.72 | 50.90 | 25.87 | 9.052 s | 35.787 GiB |
| 4 | 898.06 | 38.16 | 9.96 | 18.240 s | 33.215 GiB |

Each level ran once after its prescribed unscored warmup. Two requests gave +7.17% aggregate decode and four gave -19.64% relative to one. Longer sampled output streams differed across concurrency levels, as did MTP acceptance (draft/accept: 240/175, 548/326, 1,085/655). These rates include those differences; they are not a same-output measurement of kernel scaling or a semantic quality score. There were no OOM/swap/guard events.

## Optional concurrent GDN prefill

`CIRU_GDN_MULTI_TILE=1` enables the separately qualified tiled recurrent kernel for eligible two-to-four-sequence shapes. It leaves the single-sequence and short MTP paths unchanged. The earlier matched four-request **vision** experiment ran on Sozo, another Strix Halo system; it must not be combined with the Ciru text sweep above.

| Preserved paired observation | Core PP, tok/s | Optional GDN PP, tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 583.898 | 739.560 | +26.66% |
| 2 | 578.030 | 749.693 | +29.70% |
| 3 | 478.411 | 740.038 | +54.69% |

The third baseline was unusually slow for an unresolved reason and inflates the gain. Decode remained around 38 tok/s with changes below 0.7%. The panel used four fixed slots, total context 188,416, batch 16,384, microbatch 2,048, IU4/Q8 MTP3, F16 KV, two images per request and matched cached resumes. All 512 output token IDs and 343/489 accepted/drafted counts matched across the three core, three GDN and three ROCm 10.1 runs. GDN also preserved 241,741,824 operator output/state floats and 69,529,600 model logits bitwise. The optional path remains off by default. ROCm 10.1 had no repeatable serving-speed benefit and is not promoted by this update.

## Image-memory pressure correction

The fix bounds dense post-image host input storage with an 8,192-token default microbatch and a single UMA input copy, synchronized before reuse. At the measured 90,624-position shape, changing microbatch 16,384 to 8,192 reduced the host input allocation from **9,415,303,232 to 4,708,376,640 bytes**. This is workspace memory, not a halving of model weights, KV storage, projector or total RAM.

The completed IU4 synthetic replay used **226,712 tokens and two 720-token images**, at offsets 73,265 and 110,252. Minimum sampled available RAM was **7.788 GiB with the original microbatch-8,192 path and 13.162 GiB with the final fenced single-copy path** (+5.374 GiB). The final run completed without swap or OOM. The earlier microbatch-16,384 attempt was stopped by its memory guard at 4.040 GiB available. It was not retried as a success.

This addresses the demonstrated post-vision replay pressure. The reported incident showed flat allocations during idle and a global kernel OOM during a large replay; it did not establish a growing idle leak. Its exact binary was unavailable, so the synthetic reproduction does not prove every possible image-memory failure is fixed. Cached-versus-fresh generation can still differ; the existing configuration-sensitive image-answer miss remains documented and was not relabeled correct.

The new shared runtime code is included for both IU4 and Orca. **The historical long-image result above is IU4 evidence only.** Orca release validation is recorded separately; do not infer Orca measurements from the IU4 table.

## Current release checks on both variants

The integrated runtime passed eight simple image checks on Ciru: two checks each for IU4 one slot, IU4 two-slot pool, Orca one slot and Orca two-slot pool. Each returned the expected color with a natural stop. The functional sampler was an explicit greedy diagnostic (temperature 0, seed 12345, thinking off, maximum 96 output tokens); it is not a speed or broad vision-quality score.

Orca also completed the 179,848-token warmup and **226,712-token, two-image replay** at MTP4, microbatch 8,192 and the normal 256K one-slot context. After the prescribed 60-second idle, the replay generated all 16 forced output tokens. Minimum available RAM was **13.490 GiB**, with no swap, OOM or memory-guard event. This is an Orca capacity check of the integrated correction, not an Orca before/after gain measurement.

The first Orca long-warmup request received HTTP 400 before generation because the old fixture used a fixed media marker. The corrected attempt replaced only its two markers with the server's `/props.media_marker`; text, images and model settings stayed unchanged. The rejection is preserved, and the six image checks already completed were not repeated.

This release was built with GNU 16.2.1 on Arch Linux and exercised on Ciru's NixOS system using the existing TheRock 10.0 and custom HIP/ROCr. Its [validation record](benchmarks/release-20261006/validation.json) separates these new checks from the older sweep results.

## Provenance and reproduction

The hidden-state batch-order correction backports [hthadicherla's llama.cpp PR #29019](https://github.com/ggml-org/llama.cpp/pull/29019), commit `4453b535fd15cd5b9d5ccb956ebd38dc325c98fc`. Existing llama.cpp, Ryan Monsurate, Gufo, HaloBox, Halogen, pwilkin and AMD notices remain in the source. The sequence-safe attention, synchronized input reuse and conservative pool integration are CIRU work. No Strata optimization is part of these historical results.

The accompanying [structured results](benchmarks/concurrency-20261006/results.json) preserve settings, scope, source hashes and the measurements summarized here. They are a public summary of the retained first attempts, not a replacement for the earlier six-stack benchmark report. No broader model-quality suite was rerun for this documentation update.
