#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

struct qwen4exp_ple_geometry {
    static constexpr uint64_t source_shards         = 128;
    static constexpr uint64_t rows_per_source_shard = 2500012;
    static constexpr uint64_t total_rows            = 320001536;
    static constexpr uint64_t row_bytes             = 160;
    static constexpr uint64_t rows_per_page         = 25;
    static constexpr uint64_t row_region_bytes      = 4000;
    static constexpr uint64_t trailer_bytes         = 96;
    static constexpr uint64_t page_bytes            = 4096;
    static constexpr uint64_t page_count            = 12800062;
    static constexpr uint64_t payload_bytes          = 52429053952;
    static constexpr uint64_t final_page_valid_rows = 11;
    static constexpr uint64_t scale_bytes            = 2;
    static constexpr uint64_t lookup_heads           = 16;
};

struct qwen4exp_ple_pager_stats {
    uint64_t requests                = 0;
    uint64_t cache_hits              = 0;
    uint64_t cache_misses            = 0;
    uint64_t physical_bytes          = 0;
    uint64_t read_errors             = 0;
    uint64_t checksum_failures       = 0;
    uint64_t page_validation_failures = 0;
    uint64_t cache_capacity_pages    = 0;
    uint64_t cache_resident_pages    = 0;
    uint64_t read_latency_samples    = 0;
    double   read_latency_p50_ms     = 0.0;
    double   read_latency_p95_ms     = 0.0;
    double   read_latency_p99_ms     = 0.0;
    double   read_latency_p99_9_ms   = 0.0;
    double   read_latency_max_ms     = 0.0;
    uint64_t issued_requests          = 0;
    uint64_t unique_pages_submitted   = 0;
    uint64_t ready_before_join        = 0;
    uint64_t join_waits               = 0;
    uint64_t deadline_misses          = 0;
    uint64_t synchronous_fallbacks    = 0;
    uint64_t max_request_pages        = 0;
    bool     direct_io               = false;
};

// One decode-M=1 PLE request. Its implementation owns fixed storage for no
// more than 16 distinct CIRUPLE1 pages and the 16 decoded rows in original
// head order. The pager owns the file, cache, and single reusable worker.
class qwen4exp_ple_request {
public:
    ~qwen4exp_ple_request();

    qwen4exp_ple_request(const qwen4exp_ple_request &) = delete;
    qwen4exp_ple_request & operator=(const qwen4exp_ple_request &) = delete;

    size_t row_count() const noexcept;
    size_t unique_page_count() const noexcept;

private:
    friend class qwen4exp_ple_pager;
    struct state;

    explicit qwen4exp_ple_request(std::shared_ptr<state> state);
    std::shared_ptr<state> state_;
};

class qwen4exp_ple_pager {
public:
    static constexpr uint64_t default_cache_bytes = 512ULL * 1024 * 1024;
    static constexpr uint64_t default_join_deadline_us = 500;

    explicit qwen4exp_ple_pager(
            const std::string & manifest_or_directory,
            uint64_t            cache_bytes = default_cache_bytes);
    ~qwen4exp_ple_pager();

    qwen4exp_ple_pager(const qwen4exp_ple_pager &) = delete;
    qwen4exp_ple_pager & operator=(const qwen4exp_ple_pager &) = delete;

    const std::filesystem::path & manifest_path() const noexcept;
    const std::filesystem::path & payload_path()  const noexcept;
    const std::filesystem::path & scale_path()    const noexcept;

    uint16_t scale_bf16_bits() const noexcept;
    float    scale_f32()        const noexcept;

    void gather_rows(
            const int32_t * row_ids,
            size_t          row_count,
            float *         output,
            size_t          output_count) const;

    // Decode-M=1 early issue. The returned object is the pager's only live
    // request and must be joined before another request is submitted.
    std::unique_ptr<qwen4exp_ple_request> issue_rows(
            const int32_t * row_ids,
            size_t          row_count);

    // Wait only to the soft deadline first. If the request is still pending,
    // synchronously finish the already-issued work and count the fallback.
    // I/O, trailer, or decode failures remain fatal.
    void join_rows(
            qwen4exp_ple_request & request,
            float *                output,
            size_t                 output_count,
            uint64_t               deadline_us = default_join_deadline_us);

    qwen4exp_ple_pager_stats stats() const;

    // OCP/PyTorch signed E4M3FN. The two NaN encodings return signed NaNs;
    // gather_rows rejects them before values reach the graph.
    static float decode_e4m3fn(uint8_t code) noexcept;

private:
    struct impl;

    std::filesystem::path manifest_path_;
    std::filesystem::path payload_path_;
    std::filesystem::path scale_path_;
    uint16_t              scale_bf16_bits_ = 0;
    float                 scale_f32_       = 0.0f;
    std::unique_ptr<impl> pimpl_;
};
