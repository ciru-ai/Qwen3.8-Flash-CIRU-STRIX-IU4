Combined inference/runtime update; existing target, draft, PLE and projector files are unchanged.

- Integrates MTP shortlisting, duplicate-attention cleanup, separate graph arenas, incremental QSA caching, fused Q8 gate work, masked attention tile skipping, prompt branch reuse, attention occupancy, wide-context indexing and GDN vector/decay paths.
- Uses stock TheRock10 compiler/math with pinned pwilkin HIP/ROCr (`7dda3ac6`). MTP stays default: IU4 depth 3, Orca depth 4. Kairic Boost is opt-in (`--kairic-boost` / `KAIRIC_BOOST=1`).
- IU4 matched control → combined: 12.96K decode **37.79 → 44.53 tok/s**; 245760-token cold decode **7.62 → 21.14 tok/s**. Control already includes v4.3 + S5/D0; these are not plain-v4.3 comparisons.
- IU4 HE0–9: **60.35 tok/s** with MTP3, **64.07 tok/s** with opt-in Boost; one short panel per setting. HA20: **19/20 full-score tasks**, **98.5/100 arithmetic mean**, **99/100 official weighted score**.

Full 256K serving defaults are retained. A separate target-only 512K prefill reached 604.65 tok/s; it does not establish 512K decode or quality. Image+MTP remains incompatible in both retained and combined engines; use target-only vision. Boost has no new HA20 qualification. Weight reconstruction error is unchanged.
