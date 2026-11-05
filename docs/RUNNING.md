# Running the October 2026 runtime update

See [installation and API setup](../INSTALL.md) and [multi-slot instructions and evidence](../CONCURRENCY.md).

## Parallel requests and unified KV cache

One slot remains the default. `PARALLEL_SLOTS=2` opts into a shared total context pool; MTP remains on but the single-slot draft shortlist is disabled. Kairic Boost is limited to one slot. Read the concurrency guide for capacity and performance tradeoffs.
