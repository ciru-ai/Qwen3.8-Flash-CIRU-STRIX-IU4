#include "qwen4exp-ple-pager.h"

#include "hash/hash.h"

#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char * manifest_filename = "ple.manifest.json";
constexpr const char * payload_filename  = "ple.payload.bin";
constexpr const char * scale_filename    = "ple.scale.bf16";
constexpr const char * manifest_schema   = "ciru.qwen38-flash-next.ple-sidecar-manifest.v1";
constexpr const char * official_source_revision = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce";
constexpr const char * official_source_index_sha256 =
        "0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6";
constexpr const char * official_payload_sha256 =
        "687fc742efb6888c6cd7cf9c80cb4b1ac8cb4707b9409c206699c43363e239b2";
constexpr const char * official_scale_sha256 =
        "c7c58bd6007672362da2106fdbfaf9f50629e4bdf8598169c598027394ef9791";
constexpr uint16_t official_scale_bf16_bits = 0x3951;
constexpr const char * scale_tensor_name =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";

static_assert(qwen4exp_ple_geometry::source_shards * qwen4exp_ple_geometry::rows_per_source_shard ==
        qwen4exp_ple_geometry::total_rows, "locked PLE row geometry is inconsistent");
static_assert(qwen4exp_ple_geometry::rows_per_page * qwen4exp_ple_geometry::row_bytes ==
        qwen4exp_ple_geometry::row_region_bytes, "locked PLE page geometry is inconsistent");
static_assert(qwen4exp_ple_geometry::row_region_bytes + qwen4exp_ple_geometry::trailer_bytes ==
        qwen4exp_ple_geometry::page_bytes, "locked PLE trailer geometry is inconsistent");
static_assert(qwen4exp_ple_geometry::page_count * qwen4exp_ple_geometry::page_bytes ==
        qwen4exp_ple_geometry::payload_bytes, "locked PLE payload geometry is inconsistent");
static_assert((qwen4exp_ple_geometry::page_count - 1) * qwen4exp_ple_geometry::rows_per_page +
        qwen4exp_ple_geometry::final_page_valid_rows == qwen4exp_ple_geometry::total_rows,
        "locked PLE final-page geometry is inconsistent");

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("PLE sidecar: " + message);
}

const json & at(const json & root, const char * pointer) {
    try {
        return root.at(json::json_pointer(pointer));
    } catch (const json::exception & e) {
        fail(std::string("missing or invalid manifest field ") + pointer + ": " + e.what());
    }
}

void require_string(const json & root, const char * pointer, const std::string & expected) {
    const json & value = at(root, pointer);
    if (!value.is_string() || value.get<std::string>() != expected) {
        fail(std::string("manifest field ") + pointer + " must equal '" + expected + "'");
    }
}

void require_uint(const json & root, const char * pointer, uint64_t expected) {
    const json & value = at(root, pointer);
    if (!value.is_number_integer()) {
        fail(std::string("manifest field ") + pointer + " must be an integer");
    }
    if (value.is_number_integer() && !value.is_number_unsigned() && value.get<int64_t>() < 0) {
        fail(std::string("manifest field ") + pointer + " must be non-negative");
    }
    if (value.get<uint64_t>() != expected) {
        fail(std::string("manifest field ") + pointer + " has incompatible geometry");
    }
}

void require_bool(const json & root, const char * pointer, bool expected) {
    const json & value = at(root, pointer);
    if (!value.is_boolean() || value.get<bool>() != expected) {
        fail(std::string("manifest field ") + pointer + " has an incompatible boolean value");
    }
}

std::string require_sha256(const json & root, const char * pointer) {
    const json & value = at(root, pointer);
    if (!value.is_string()) {
        fail(std::string("manifest field ") + pointer + " must be a SHA-256 string");
    }
    const std::string digest = value.get<std::string>();
    if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos) {
        fail(std::string("manifest field ") + pointer + " must be 64 lowercase hexadecimal characters");
    }
    return digest;
}

void require_shape(const json & root, const char * pointer, uint64_t dim0) {
    const json & value = at(root, pointer);
    if (!value.is_array() || value.size() != 1 || !value[0].is_number_integer() ||
            value[0].get<uint64_t>() != dim0) {
        fail(std::string("manifest field ") + pointer + " has an incompatible tensor shape");
    }
}

void require_shape(const json & root, const char * pointer, uint64_t dim0, uint64_t dim1) {
    const json & value = at(root, pointer);
    if (!value.is_array() || value.size() != 2 || !value[0].is_number_integer() ||
            !value[1].is_number_integer() || value[0].get<uint64_t>() != dim0 ||
            value[1].get<uint64_t>() != dim1) {
        fail(std::string("manifest field ") + pointer + " has an incompatible tensor shape");
    }
}

void require_file_size(const fs::path & path, uint64_t expected, const char * label) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        fail(std::string(label) + " file is missing or not regular: " + path.string());
    }
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) {
        fail(std::string("cannot stat ") + label + " file " + path.string() + ": " + ec.message());
    }
    if (size != expected) {
        std::ostringstream message;
        message << label << " file length is " << size << " bytes; expected " << expected;
        fail(message.str());
    }
}

json read_manifest(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        fail("cannot open manifest: " + path.string());
    }
    try {
        json root;
        input >> root;
        if (!root.is_object()) {
            fail("manifest root must be a JSON object");
        }
        return root;
    } catch (const json::exception & e) {
        fail("invalid JSON in " + path.string() + ": " + e.what());
    }
}

void validate_weight_records(const json & root) {
    const json & weights = at(root, "/tensors/weights");
    if (!weights.is_array() || weights.size() != qwen4exp_ple_geometry::source_shards) {
        fail("manifest tensor weight inventory must contain exactly 128 shards");
    }

    const uint64_t tensor_bytes = qwen4exp_ple_geometry::rows_per_source_shard *
            qwen4exp_ple_geometry::row_bytes;
    for (uint64_t i = 0; i < qwen4exp_ple_geometry::source_shards; ++i) {
        const std::string base = "/tensors/weights/" + std::to_string(i);
        std::ostringstream expected_name;
        expected_name << "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
                      << i << ".weight";
        require_string(root, (base + "/name").c_str(), expected_name.str());
        require_string(root, (base + "/dtype").c_str(), "F8_E4M3");
        require_shape(root, (base + "/shape").c_str(), qwen4exp_ple_geometry::rows_per_source_shard,
                qwen4exp_ple_geometry::row_bytes);
        require_uint(root, (base + "/data_length").c_str(), tensor_bytes);
        require_sha256(root, (base + "/payload_sha256").c_str());
    }
}

void validate_manifest(const json & root) {
    require_string(root, "/schema", manifest_schema);
    require_string(root, "/source/revision", official_source_revision);
    if (require_sha256(root, "/source/index/sha256") != official_source_index_sha256) {
        fail("source index SHA-256 does not match the official Qwen3.8 FP8 checkpoint identity");
    }

    require_uint(root, "/counts/source_shards", qwen4exp_ple_geometry::source_shards);
    require_uint(root, "/counts/rows_per_source_shard", qwen4exp_ple_geometry::rows_per_source_shard);
    require_uint(root, "/counts/total_rows", qwen4exp_ple_geometry::total_rows);
    require_uint(root, "/counts/page_count", qwen4exp_ple_geometry::page_count);

    require_string(root, "/format/name", "CIRUPLE1");
    require_string(root, "/format/magic_ascii", "CIRUPLE1");
    require_uint(root, "/format/version", 1);
    require_uint(root, "/format/flags", 0);
    require_uint(root, "/format/page_size_bytes", qwen4exp_ple_geometry::page_bytes);
    require_uint(root, "/format/rows_per_page", qwen4exp_ple_geometry::rows_per_page);
    require_uint(root, "/format/row_bytes", qwen4exp_ple_geometry::row_bytes);
    require_uint(root, "/format/row_region_bytes", qwen4exp_ple_geometry::row_region_bytes);
    require_uint(root, "/format/trailer_bytes", qwen4exp_ple_geometry::trailer_bytes);
    require_string(root, "/format/trailer_struct", "<8sIIQHHI32s32s");
    require_string(root, "/format/checksum", "SHA-256 over the complete row region including zero padding");
    require_uint(root, "/format/reserved_trailer_bytes", 32);
    require_bool(root, "/format/row_straddling", false);
    require_uint(root, "/format/final_page_valid_rows", qwen4exp_ple_geometry::final_page_valid_rows);
    require_bool(root, "/format/final_unused_row_slots_zero_filled", true);
    require_string(root, "/format/scale_storage", "separate exact two-byte BF16 scalar");

    validate_weight_records(root);
    require_string(root, "/tensors/scale/name", scale_tensor_name);
    require_string(root, "/tensors/scale/dtype", "BF16");
    require_shape(root, "/tensors/scale/shape", 1);
    require_uint(root, "/tensors/scale/data_length", qwen4exp_ple_geometry::scale_bytes);
    const std::string tensor_scale_sha = require_sha256(root, "/tensors/scale/payload_sha256");

    require_string(root, "/outputs/payload/name", payload_filename);
    require_uint(root, "/outputs/payload/bytes", qwen4exp_ple_geometry::payload_bytes);
    if (require_sha256(root, "/outputs/payload/sha256") != official_payload_sha256) {
        fail("payload SHA-256 does not match the official Qwen3.8 CIRUPLE1 artifact identity");
    }
    require_string(root, "/outputs/scale/name", scale_filename);
    require_uint(root, "/outputs/scale/bytes", qwen4exp_ple_geometry::scale_bytes);
    const std::string output_scale_sha = require_sha256(root, "/outputs/scale/sha256");
    require_string(root, "/outputs/manifest/name", manifest_filename);

    if (tensor_scale_sha != output_scale_sha) {
        fail("scale tensor and output SHA-256 identities disagree");
    }
    if (tensor_scale_sha != official_scale_sha256) {
        fail("scale SHA-256 does not match the official Qwen3.8 PLE scalar identity");
    }
}

uint16_t load_le16(const unsigned char * data) {
    return static_cast<uint16_t>(data[0]) |
            (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t load_le32(const unsigned char * data) {
    return static_cast<uint32_t>(data[0]) |
            (static_cast<uint32_t>(data[1]) << 8) |
            (static_cast<uint32_t>(data[2]) << 16) |
            (static_cast<uint32_t>(data[3]) << 24);
}

uint64_t load_le64(const unsigned char * data) {
    uint64_t result = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        result |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return result;
}

std::string bytes_to_hex(const unsigned char * data, size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(2 * size, '0');
    for (size_t i = 0; i < size; ++i) {
        result[2 * i]     = digits[data[i] >> 4];
        result[2 * i + 1] = digits[data[i] & 0x0f];
    }
    return result;
}

uint16_t f32_to_bf16_rne(float value) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_f32(uint16_t value) noexcept {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

struct qwen4exp_ple_request::state {
    using page_data = std::array<unsigned char, qwen4exp_ple_geometry::page_bytes>;

    std::array<int32_t, qwen4exp_ple_geometry::lookup_heads> row_ids = {};
    std::array<uint64_t, qwen4exp_ple_geometry::lookup_heads> page_ids = {};
    std::array<uint8_t, qwen4exp_ple_geometry::lookup_heads> page_map = {};
    std::array<uint8_t, qwen4exp_ple_geometry::lookup_heads> row_slots = {};
    std::array<page_data, qwen4exp_ple_geometry::lookup_heads> pages = {};
    std::array<float, qwen4exp_ple_geometry::lookup_heads * qwen4exp_ple_geometry::row_bytes> output = {};

    size_t row_count = 0;
    size_t unique_page_count = 0;
    std::chrono::steady_clock::time_point issued_at;
    std::mutex mutex;
    std::condition_variable ready;
    bool done = false;
    std::exception_ptr error;
};

qwen4exp_ple_request::qwen4exp_ple_request(std::shared_ptr<state> state) : state_(std::move(state)) {}
qwen4exp_ple_request::~qwen4exp_ple_request() = default;

size_t qwen4exp_ple_request::row_count() const noexcept {
    return state_ ? state_->row_count : 0;
}

size_t qwen4exp_ple_request::unique_page_count() const noexcept {
    return state_ ? state_->unique_page_count : 0;
}

struct qwen4exp_ple_pager::impl {
    using page_data = qwen4exp_ple_request::state::page_data;
    static constexpr size_t read_latency_histogram_us = 10000;
    static constexpr size_t bulk_worker_count = 16;

    struct cache_slot {
        uint64_t  tag;
        bool      referenced;
        page_data data;

        cache_slot(uint64_t tag, const page_data & data) : tag(tag), referenced(true), data(data) {}
    };

#ifdef __linux__
    struct bulk_page_plan {
        uint64_t page_id = 0;
        page_data data = {};
        bool cache_hit = false;
        std::streamsize read_bytes = -1;
        int read_error = 0;
        uint64_t read_latency_ns = 0;
    };

    struct bulk_batch {
        std::mutex mutex;
        std::condition_variable ready;
        size_t remaining = 0;
    };

    struct bulk_task {
        bulk_page_plan * page = nullptr;
        bulk_batch * batch = nullptr;
    };
#endif

    explicit impl(const fs::path & payload_path, uint64_t cache_bytes, float scale_f32) : scale_f32(scale_f32) {
        if (cache_bytes % qwen4exp_ple_geometry::page_bytes != 0) {
            fail("cache size must be a multiple of 4096 bytes");
        }
        const uint64_t pages = cache_bytes / qwen4exp_ple_geometry::page_bytes;
        if (pages > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            fail("cache page count does not fit in size_t");
        }
        cache_capacity_pages = static_cast<size_t>(pages);
        slots.reserve(cache_capacity_pages);
        locations.reserve(cache_capacity_pages);

#ifdef __linux__
        payload_fd = ::open(payload_path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (payload_fd < 0) {
            fail("cannot open payload with O_DIRECT: " + payload_path.string() + ": " + std::strerror(errno));
        }
        if (::posix_memalign(&direct_buffer, qwen4exp_ple_geometry::page_bytes,
                    qwen4exp_ple_geometry::page_bytes) != 0) {
            ::close(payload_fd);
            payload_fd = -1;
            fail("cannot allocate an aligned O_DIRECT page buffer");
        }
        if (::posix_memalign(&bulk_direct_buffers, qwen4exp_ple_geometry::page_bytes,
                    bulk_worker_count * qwen4exp_ple_geometry::page_bytes) != 0) {
            std::free(direct_buffer);
            direct_buffer = nullptr;
            ::close(payload_fd);
            payload_fd = -1;
            fail("cannot allocate the aligned P16 O_DIRECT buffers");
        }
#else
        payload.rdbuf()->pubsetbuf(nullptr, 0);
        payload.open(payload_path, std::ios::binary);
        if (!payload) {
            fail("cannot open payload: " + payload_path.string());
        }
#endif
        try {
            worker = std::thread([this]() { worker_loop(); });
#ifdef __linux__
            bulk_workers.reserve(bulk_worker_count);
            for (size_t i = 0; i < bulk_worker_count; ++i) {
                bulk_workers.emplace_back([this, i]() { bulk_worker_loop(i); });
            }
#endif
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                stopping = true;
            }
            queue_ready.notify_all();
            if (worker.joinable()) {
                worker.join();
            }
#ifdef __linux__
            {
                std::lock_guard<std::mutex> lock(bulk_queue_mutex);
                bulk_stopping = true;
            }
            bulk_queue_ready.notify_all();
            for (auto & bulk_worker : bulk_workers) {
                if (bulk_worker.joinable()) {
                    bulk_worker.join();
                }
            }
            std::free(bulk_direct_buffers);
            bulk_direct_buffers = nullptr;
            std::free(direct_buffer);
            direct_buffer = nullptr;
            if (payload_fd >= 0) {
                ::close(payload_fd);
                payload_fd = -1;
            }
#endif
            throw;
        }
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stopping = true;
        }
        queue_ready.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
#ifdef __linux__
        {
            std::lock_guard<std::mutex> lock(bulk_queue_mutex);
            bulk_stopping = true;
        }
        bulk_queue_ready.notify_all();
        for (auto & bulk_worker : bulk_workers) {
            if (bulk_worker.joinable()) {
                bulk_worker.join();
            }
        }
        std::free(bulk_direct_buffers);
        std::free(direct_buffer);
        if (payload_fd >= 0) {
            ::close(payload_fd);
        }
#endif
    }

#ifdef __linux__
    void bulk_worker_loop(size_t worker_index) noexcept {
        auto * const buffer = static_cast<unsigned char *>(bulk_direct_buffers) +
                worker_index * qwen4exp_ple_geometry::page_bytes;
        for (;;) {
            bulk_task * task = nullptr;
            {
                std::unique_lock<std::mutex> lock(bulk_queue_mutex);
                bulk_queue_ready.wait(lock, [this]() { return bulk_stopping || !bulk_queue.empty(); });
                if (bulk_stopping && bulk_queue.empty()) {
                    return;
                }
                task = bulk_queue.front();
                bulk_queue.pop_front();
            }

            const auto started = std::chrono::steady_clock::now();
            ssize_t result;
            do {
                result = ::pread(payload_fd, buffer, qwen4exp_ple_geometry::page_bytes,
                        static_cast<off_t>(task->page->page_id * qwen4exp_ple_geometry::page_bytes));
            } while (result < 0 && errno == EINTR);
            task->page->read_latency_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count());
            task->page->read_bytes = result;
            task->page->read_error = result < 0 ? errno : 0;
            if (result > 0) {
                std::memcpy(task->page->data.data(), buffer,
                        std::min<size_t>(static_cast<size_t>(result), task->page->data.size()));
            }

            {
                std::lock_guard<std::mutex> lock(task->batch->mutex);
                --task->batch->remaining;
                if (task->batch->remaining == 0) {
                    task->batch->ready.notify_one();
                }
            }
        }
    }
#endif

    void worker_loop() noexcept {
        for (;;) {
            std::shared_ptr<qwen4exp_ple_request::state> request;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_ready.wait(lock, [this]() { return stopping || queued_request != nullptr; });
                if (stopping && queued_request == nullptr) {
                    return;
                }
                request = std::move(queued_request);
            }

            try {
                execute_request(*request);
            } catch (...) {
                request->error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(request->mutex);
                request->done = true;
            }
            request->ready.notify_all();
        }
    }

    void execute_request(qwen4exp_ple_request::state & request) {
        // The cache and O_DIRECT bounce buffer are pager-owned and serialized.
        // The request copies each distinct page into one of sixteen fixed slots
        // before decoding rows back into original head order.
        std::lock_guard<std::mutex> lock(cache_mutex);
        counters.requests += request.row_count;

        for (size_t p = 0; p < request.unique_page_count; ++p) {
            request.pages[p] = page(request.page_ids[p]);
        }

        for (size_t i = 0; i < request.row_count; ++i) {
            const auto & staged = request.pages[request.page_map[i]];
            const unsigned char * const encoded = staged.data() +
                    request.row_slots[i] * qwen4exp_ple_geometry::row_bytes;
            float * const decoded = request.output.data() + i * qwen4exp_ple_geometry::row_bytes;
            for (size_t j = 0; j < qwen4exp_ple_geometry::row_bytes; ++j) {
                const float value = qwen4exp_ple_pager::decode_e4m3fn(encoded[j]);
                if (std::isnan(value)) {
                    fail("FP8 E4M3FN NaN code at row " + std::to_string(request.row_ids[i]) +
                            ", column " + std::to_string(j));
                }
                decoded[j] = bf16_to_f32(f32_to_bf16_rne(value * scale_f32));
            }
        }
    }

    [[noreturn]] void page_failure(uint64_t page_index, const std::string & message, bool checksum = false) {
        ++counters.page_validation_failures;
        if (checksum) {
            ++counters.checksum_failures;
        }
        fail("page " + std::to_string(page_index) + " validation failed: " + message);
    }

    void validate_page(const page_data & page, uint64_t page_index, bool strict_sha = true) {
        constexpr size_t trailer = qwen4exp_ple_geometry::row_region_bytes;
        const unsigned char * const fields = page.data() + trailer;

        if (std::memcmp(fields, "CIRUPLE1", 8) != 0) {
            page_failure(page_index, "magic is not CIRUPLE1");
        }
        if (load_le32(fields + 8) != 1) {
            page_failure(page_index, "version is not 1");
        }
        if (load_le32(fields + 12) != 0) {
            page_failure(page_index, "flags are not zero");
        }
        if (load_le64(fields + 16) != page_index) {
            page_failure(page_index, "absolute page index mismatch");
        }

        const uint16_t expected_rows = page_index + 1 == qwen4exp_ple_geometry::page_count
                ? qwen4exp_ple_geometry::final_page_valid_rows
                : qwen4exp_ple_geometry::rows_per_page;
        if (load_le16(fields + 24) != expected_rows) {
            page_failure(page_index, "valid-row count mismatch");
        }
        if (load_le16(fields + 26) != qwen4exp_ple_geometry::row_bytes) {
            page_failure(page_index, "row width mismatch");
        }
        if (load_le32(fields + 28) != expected_rows * qwen4exp_ple_geometry::row_bytes) {
            page_failure(page_index, "valid-byte count mismatch");
        }

        if (strict_sha) {
            const std::string actual_hash = hash_sha256_hex(
                    page.data(), qwen4exp_ple_geometry::row_region_bytes);
            if (actual_hash != bytes_to_hex(fields + 32, 32)) {
                page_failure(page_index, "row-region SHA-256 mismatch", true);
            }
        }
        if (!std::all_of(fields + 64, fields + 96, [](unsigned char value) { return value == 0; })) {
            page_failure(page_index, "reserved trailer bytes are nonzero");
        }

        if (expected_rows < qwen4exp_ple_geometry::rows_per_page) {
            const size_t valid_bytes = expected_rows * qwen4exp_ple_geometry::row_bytes;
            if (!std::all_of(page.begin() + valid_bytes, page.begin() + qwen4exp_ple_geometry::row_region_bytes,
                    [](unsigned char value) { return value == 0; })) {
                page_failure(page_index, "unused final-page row slots are nonzero");
            }
        }
    }

    void read_page(uint64_t page_index, page_data & page) {
        const uint64_t byte_offset = page_index * qwen4exp_ple_geometry::page_bytes;
        if (byte_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            fail("payload offset does not fit in streamoff");
        }

        const auto read_start = std::chrono::steady_clock::now();
#ifdef __linux__
        ssize_t result;
        do {
            result = ::pread(payload_fd, direct_buffer, qwen4exp_ple_geometry::page_bytes,
                    static_cast<off_t>(byte_offset));
        } while (result < 0 && errno == EINTR);
        if (result < 0) {
            ++counters.read_errors;
            fail("payload read failed at page " + std::to_string(page_index) + ": " + std::strerror(errno));
        }
        if (result > 0) {
            std::memcpy(page.data(), direct_buffer, static_cast<size_t>(result));
        }
        const std::streamsize read_bytes = result;
#else
        payload.clear();
        payload.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
        payload.read(reinterpret_cast<char *>(page.data()), page.size());
        const std::streamsize read_bytes = payload.gcount();
#endif
        record_read_latency(std::chrono::steady_clock::now() - read_start);
        if (read_bytes > 0) {
            counters.physical_bytes += static_cast<uint64_t>(read_bytes);
        }
        if (read_bytes != static_cast<std::streamsize>(page.size())) {
            ++counters.read_errors;
            fail("short payload read at page " + std::to_string(page_index));
        }
        validate_page(page, page_index);
    }

    void record_read_latency(std::chrono::steady_clock::duration elapsed) {
        const uint64_t ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        const size_t bucket = std::min<uint64_t>(ns / 1000, read_latency_histogram_us);
        ++read_latency_us[bucket];
        ++read_latency_samples;
        read_latency_max_ns = std::max(read_latency_max_ns, ns);
    }

    double read_latency_percentile(double quantile) const {
        if (read_latency_samples == 0) {
            return 0.0;
        }
        const uint64_t rank = static_cast<uint64_t>(std::ceil(quantile * read_latency_samples));
        uint64_t cumulative = 0;
        for (size_t i = 0; i < read_latency_us.size(); ++i) {
            cumulative += read_latency_us[i];
            if (cumulative >= rank) {
                if (i == read_latency_histogram_us) {
                    return static_cast<double>(read_latency_max_ns) / 1.0e6;
                }
                // Report the inclusive upper edge of each one-microsecond bin.
                return static_cast<double>(i + 1) / 1000.0;
            }
        }
        return static_cast<double>(read_latency_max_ns) / 1.0e6;
    }

    // cache_mutex must be held. The insertion order is part of H69's stable
    // bulk plan, so CLOCK state is independent of worker completion order.
    void cache_insert(uint64_t page_index, const page_data & data) {
        if (cache_capacity_pages == 0) {
            return;
        }
        const auto existing = locations.find(page_index);
        if (existing != locations.end()) {
            slots[existing->second]->referenced = true;
            return;
        }

        size_t slot_index;
        if (slots.size() < cache_capacity_pages) {
            slot_index = slots.size();
            slots.emplace_back(std::make_unique<cache_slot>(page_index, data));
        } else {
            while (slots[clock_hand]->referenced) {
                slots[clock_hand]->referenced = false;
                clock_hand = (clock_hand + 1) % cache_capacity_pages;
            }
            slot_index = clock_hand;
            locations.erase(slots[slot_index]->tag);
            slots[slot_index]->tag        = page_index;
            slots[slot_index]->referenced = true;
            slots[slot_index]->data       = data;
            clock_hand = (clock_hand + 1) % cache_capacity_pages;
        }
        locations[page_index] = slot_index;
    }

#ifdef __linux__
    void gather_rows_bulk(
            const int32_t * row_ids,
            size_t          row_count,
            float *         output,
            bool            strict_sha) {
        const auto total_started = std::chrono::steady_clock::now();
        std::vector<size_t> row_page(row_count);
        std::vector<uint8_t> row_slot(row_count);
        std::vector<bulk_page_plan> page_plan;
        page_plan.reserve(row_count);
        std::unordered_map<uint64_t, size_t> page_to_plan;
        page_to_plan.reserve(row_count);

        for (size_t i = 0; i < row_count; ++i) {
            const int32_t row_id = row_ids[i];
            if (row_id < 0 || static_cast<uint64_t>(row_id) >= qwen4exp_ple_geometry::total_rows) {
                fail("global row ID is out of range: " + std::to_string(row_id));
            }
            const uint64_t page_id = static_cast<uint64_t>(row_id) /
                    qwen4exp_ple_geometry::rows_per_page;
            const auto inserted = page_to_plan.emplace(page_id, page_plan.size());
            if (inserted.second) {
                bulk_page_plan plan;
                plan.page_id = page_id;
                page_plan.push_back(std::move(plan));
            }
            row_page[i] = inserted.first->second;
            row_slot[i] = static_cast<uint8_t>(
                    static_cast<uint64_t>(row_id) % qwen4exp_ple_geometry::rows_per_page);
        }
        const auto plan_done = std::chrono::steady_clock::now();

        size_t miss_count = 0;
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            counters.requests += row_count;
            for (auto & planned : page_plan) {
                const auto found = locations.find(planned.page_id);
                if (found == locations.end()) {
                    ++miss_count;
                    continue;
                }
                auto & slot = *slots[found->second];
                slot.referenced = true;
                planned.cache_hit = true;
                planned.data = slot.data;
            }
            // This matches the scalar semantics: the first access to a new
            // page misses, while later rows in that page are logical hits.
            counters.cache_misses += miss_count;
            counters.cache_hits += row_count - miss_count;
        }
        const auto probe_done = std::chrono::steady_clock::now();

        bulk_batch batch;
        batch.remaining = miss_count;
        std::vector<bulk_task> tasks;
        tasks.reserve(miss_count);
        for (auto & planned : page_plan) {
            if (!planned.cache_hit) {
                tasks.push_back({ &planned, &batch });
            }
        }

        if (!tasks.empty()) {
            {
                std::lock_guard<std::mutex> lock(bulk_queue_mutex);
                if (bulk_stopping) {
                    fail("P16 worker pool is stopping");
                }
                for (auto & task : tasks) {
                    bulk_queue.push_back(&task);
                }
            }
            bulk_queue_ready.notify_all();

            std::unique_lock<std::mutex> lock(batch.mutex);
            batch.ready.wait(lock, [&batch]() { return batch.remaining == 0; });
        }
        const auto io_done = std::chrono::steady_clock::now();

        uint64_t physical_bytes = 0;
        for (auto & planned : page_plan) {
            if (planned.cache_hit) {
                continue;
            }
            if (planned.read_bytes > 0) {
                physical_bytes += static_cast<uint64_t>(planned.read_bytes);
            }
            if (planned.read_bytes < 0) {
                ++counters.read_errors;
                fail("payload read failed at page " + std::to_string(planned.page_id) +
                        ": " + std::strerror(planned.read_error));
            }
            if (planned.read_bytes != static_cast<std::streamsize>(planned.data.size())) {
                ++counters.read_errors;
                fail("short payload read at page " + std::to_string(planned.page_id));
            }
            // Structural validation always remains in the candidate. Full
            // per-page SHA is available only through the explicit strict mode.
            validate_page(planned.data, planned.page_id, strict_sha);
        }
        const auto validate_done = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            counters.physical_bytes += physical_bytes;
            for (auto & planned : page_plan) {
                if (planned.cache_hit) {
                    continue;
                }
                record_read_latency(std::chrono::nanoseconds(planned.read_latency_ns));
                cache_insert(planned.page_id, planned.data);
            }
        }
        const auto commit_done = std::chrono::steady_clock::now();

        for (size_t i = 0; i < row_count; ++i) {
            const auto & planned = page_plan[row_page[i]];
            const unsigned char * const encoded = planned.data.data() +
                    row_slot[i] * qwen4exp_ple_geometry::row_bytes;
            float * const decoded = output + i * qwen4exp_ple_geometry::row_bytes;
            for (size_t j = 0; j < qwen4exp_ple_geometry::row_bytes; ++j) {
                const float value = qwen4exp_ple_pager::decode_e4m3fn(encoded[j]);
                if (std::isnan(value)) {
                    fail("FP8 E4M3FN NaN code at row " + std::to_string(row_ids[i]) +
                            ", column " + std::to_string(j));
                }
                decoded[j] = bf16_to_f32(f32_to_bf16_rne(value * scale_f32));
            }
        }
        const auto decode_done = std::chrono::steady_clock::now();

        // H69A targeted diagnostic: emit exactly one phase record for the
        // frozen PP512 gather. This is deliberately not general telemetry.
        static std::atomic<bool> pp512_reported { false };
        if (row_count == 8192 && !pp512_reported.exchange(true)) {
            uint64_t read_latency_sum_ns = 0;
            uint64_t read_latency_max_ns = 0;
            for (const auto & planned : page_plan) {
                if (!planned.cache_hit) {
                    read_latency_sum_ns += planned.read_latency_ns;
                    read_latency_max_ns = std::max(read_latency_max_ns, planned.read_latency_ns);
                }
            }
            const auto elapsed_ms = [](auto begin, auto end) {
                return std::chrono::duration<double, std::milli>(end - begin).count();
            };
            const double read_latency_mean_ms = miss_count == 0 ? 0.0 :
                    static_cast<double>(read_latency_sum_ns) / static_cast<double>(miss_count) / 1.0e6;
            const double read_latency_max_ms = static_cast<double>(read_latency_max_ns) / 1.0e6;
            std::fprintf(stderr,
                    "H69_P16_PHASE rows=%zu unique=%zu logical_hits=%zu physical_misses=%zu "
                    "plan_ms=%.6f probe_ms=%.6f io_ms=%.6f validate_ms=%.6f "
                    "commit_ms=%.6f decode_ms=%.6f total_ms=%.6f "
                    "read_mean_ms=%.6f read_max_ms=%.6f strict_sha=%d\n",
                    row_count, page_plan.size(), row_count - miss_count, miss_count,
                    elapsed_ms(total_started, plan_done), elapsed_ms(plan_done, probe_done),
                    elapsed_ms(probe_done, io_done), elapsed_ms(io_done, validate_done),
                    elapsed_ms(validate_done, commit_done), elapsed_ms(commit_done, decode_done),
                    elapsed_ms(total_started, decode_done), read_latency_mean_ms,
                    read_latency_max_ms, strict_sha ? 1 : 0);
        }
    }
#endif

    const page_data & page(uint64_t page_index) {
        const auto found = locations.find(page_index);
        if (found != locations.end()) {
            ++counters.cache_hits;
            auto & slot = *slots[found->second];
            slot.referenced = true;
            return slot.data;
        }

        ++counters.cache_misses;
        read_page(page_index, scratch);
        if (cache_capacity_pages == 0) {
            return scratch;
        }
        cache_insert(page_index, scratch);
        return slots[locations.at(page_index)]->data;
    }

#ifdef __linux__
    int payload_fd = -1;
    void * direct_buffer = nullptr;
    void * bulk_direct_buffers = nullptr;
    std::mutex bulk_queue_mutex;
    std::condition_variable bulk_queue_ready;
    std::deque<bulk_task *> bulk_queue;
    std::vector<std::thread> bulk_workers;
    bool bulk_stopping = false;
#else
    std::ifstream payload;
#endif
    std::mutex cache_mutex;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::thread worker;
    bool stopping = false;
    std::shared_ptr<qwen4exp_ple_request::state> queued_request;
    std::shared_ptr<qwen4exp_ple_request::state> active_request;
    float scale_f32 = 0.0f;
    size_t cache_capacity_pages = 0;
    size_t clock_hand = 0;
    std::vector<std::unique_ptr<cache_slot>> slots;
    std::unordered_map<uint64_t, size_t> locations;
    page_data scratch = {};
    std::array<uint64_t, read_latency_histogram_us + 1> read_latency_us = {};
    uint64_t read_latency_samples = 0;
    uint64_t read_latency_max_ns = 0;
    qwen4exp_ple_pager_stats counters;
};

qwen4exp_ple_pager::qwen4exp_ple_pager(
        const std::string & manifest_or_directory,
        uint64_t            cache_bytes) {
    if (manifest_or_directory.empty()) {
        fail("manifest path is empty");
    }

    fs::path supplied = fs::u8path(manifest_or_directory);
    std::error_code ec;
    manifest_path_ = fs::is_directory(supplied, ec) && !ec ? supplied / manifest_filename : supplied;
    if (manifest_path_.filename() != manifest_filename) {
        fail(std::string("manifest file must be named ") + manifest_filename);
    }
    manifest_path_ = fs::absolute(manifest_path_, ec);
    if (ec) {
        fail("cannot resolve manifest path: " + ec.message());
    }

    const json root = read_manifest(manifest_path_);
    validate_manifest(root);

    payload_path_ = manifest_path_.parent_path() / payload_filename;
    scale_path_   = manifest_path_.parent_path() / scale_filename;
    require_file_size(payload_path_, qwen4exp_ple_geometry::payload_bytes, "payload");
    require_file_size(scale_path_, qwen4exp_ple_geometry::scale_bytes, "scale");

    std::array<unsigned char, qwen4exp_ple_geometry::scale_bytes> scale_bytes = {};
    std::ifstream scale_input(scale_path_, std::ios::binary);
    if (!scale_input.read(reinterpret_cast<char *>(scale_bytes.data()), scale_bytes.size())) {
        fail("cannot read the exact two-byte BF16 scale: " + scale_path_.string());
    }

    const std::string expected_scale_sha = at(root, "/outputs/scale/sha256").get<std::string>();
    const std::string actual_scale_sha = hash_sha256_hex(scale_bytes.data(), scale_bytes.size());
    if (actual_scale_sha != expected_scale_sha) {
        fail("scale file SHA-256 does not match the manifest");
    }

    // Safetensors scalar payloads are little-endian. Preserve the exact bits and
    // separately decode them so graph integration cannot substitute F16/F32.
    scale_bf16_bits_ = static_cast<uint16_t>(scale_bytes[0]) |
            (static_cast<uint16_t>(scale_bytes[1]) << 8);
    if (scale_bf16_bits_ != official_scale_bf16_bits) {
        fail("scale BF16 bits do not match the official Qwen3.8 PLE scalar identity");
    }
    const uint32_t f32_bits = static_cast<uint32_t>(scale_bf16_bits_) << 16;
    std::memcpy(&scale_f32_, &f32_bits, sizeof(scale_f32_));
    if (!std::isfinite(scale_f32_) || scale_f32_ <= 0.0f) {
        fail("BF16 scale must decode to a positive finite scalar");
    }

    pimpl_ = std::make_unique<impl>(payload_path_, cache_bytes, scale_f32_);
}

qwen4exp_ple_pager::~qwen4exp_ple_pager() = default;

const fs::path & qwen4exp_ple_pager::manifest_path() const noexcept {
    return manifest_path_;
}

const fs::path & qwen4exp_ple_pager::payload_path() const noexcept {
    return payload_path_;
}

const fs::path & qwen4exp_ple_pager::scale_path() const noexcept {
    return scale_path_;
}

uint16_t qwen4exp_ple_pager::scale_bf16_bits() const noexcept {
    return scale_bf16_bits_;
}

float qwen4exp_ple_pager::scale_f32() const noexcept {
    return scale_f32_;
}

float qwen4exp_ple_pager::decode_e4m3fn(uint8_t code) noexcept {
    const float sign = code & 0x80 ? -1.0f : 1.0f;
    const uint8_t exponent = (code >> 3) & 0x0f;
    const uint8_t mantissa = code & 0x07;

    if (exponent == 0x0f && mantissa == 0x07) {
        return std::copysign(std::numeric_limits<float>::quiet_NaN(), sign);
    }

    const float magnitude = exponent == 0
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
    return std::copysign(magnitude, sign);
}

void qwen4exp_ple_pager::gather_rows(
        const int32_t * row_ids,
        size_t          row_count,
        float *         output,
        size_t          output_count) const {
    if (row_count > std::numeric_limits<size_t>::max() / qwen4exp_ple_geometry::row_bytes) {
        fail("gather row count overflows size_t");
    }
    const size_t expected_output = row_count * qwen4exp_ple_geometry::row_bytes;
    if (output_count != expected_output) {
        fail("gather output has incompatible element count");
    }
    if (row_count == 0) {
        return;
    }
    if (row_ids == nullptr || output == nullptr) {
        fail("gather row IDs and output must be non-null");
    }

#ifdef __linux__
    // Selector 0/unset preserves the original scalar path, including M=1.
    // Selector 16 admits only multi-token gathers and leaves decode behavior
    // unchanged. The environment is deliberately read per call so a matched
    // in-process harness can switch arms without hidden static state.
    const char * const workers_env = std::getenv("GGML_QWEN4EXP_PLE_WORKERS");
    if (row_count > qwen4exp_ple_geometry::lookup_heads &&
            workers_env != nullptr && std::strcmp(workers_env, "16") == 0) {
        const char * const strict_env = std::getenv("GGML_QWEN4EXP_PLE_STRICT_SHA");
        const bool strict_sha = strict_env != nullptr && std::strcmp(strict_env, "0") != 0;
        pimpl_->gather_rows_bulk(row_ids, row_count, output, strict_sha);
        return;
    }
#endif

    std::lock_guard<std::mutex> lock(pimpl_->cache_mutex);
    pimpl_->counters.requests += row_count;
    for (size_t i = 0; i < row_count; ++i) {
        const int32_t row_id = row_ids[i];
        if (row_id < 0 || static_cast<uint64_t>(row_id) >= qwen4exp_ple_geometry::total_rows) {
            fail("global row ID is out of range: " + std::to_string(row_id));
        }

        const uint64_t page_index = static_cast<uint64_t>(row_id) / qwen4exp_ple_geometry::rows_per_page;
        const uint64_t slot = static_cast<uint64_t>(row_id) % qwen4exp_ple_geometry::rows_per_page;
        const auto & page = pimpl_->page(page_index);
        const unsigned char * const encoded = page.data() + slot * qwen4exp_ple_geometry::row_bytes;
        float * const decoded = output + i * qwen4exp_ple_geometry::row_bytes;
        for (size_t j = 0; j < qwen4exp_ple_geometry::row_bytes; ++j) {
            const float value = decode_e4m3fn(encoded[j]);
            if (std::isnan(value)) {
                fail("FP8 E4M3FN NaN code at row " + std::to_string(row_id) +
                        ", column " + std::to_string(j));
            }
            // Match the locked SGLang path: FP8 is exactly representable in
            // BF16, multiplication uses the resident BF16 scalar, and the
            // product is rounded to BF16 (RNE) before graph-visible F32.
            decoded[j] = bf16_to_f32(f32_to_bf16_rne(value * scale_f32_));
        }
    }
}

std::unique_ptr<qwen4exp_ple_request> qwen4exp_ple_pager::issue_rows(
        const int32_t * row_ids,
        size_t          row_count) {
    if (row_count == 0 || row_count > qwen4exp_ple_geometry::lookup_heads) {
        fail("early issue requires 1..16 decode-M=1 rows");
    }
    if (row_ids == nullptr) {
        fail("early-issue row IDs must be non-null");
    }

    auto state = std::make_shared<qwen4exp_ple_request::state>();
    state->row_count = row_count;
    state->issued_at = std::chrono::steady_clock::now();

    for (size_t i = 0; i < row_count; ++i) {
        const int32_t row_id = row_ids[i];
        if (row_id < 0 || static_cast<uint64_t>(row_id) >= qwen4exp_ple_geometry::total_rows) {
            fail("global row ID is out of range: " + std::to_string(row_id));
        }

        const uint64_t page_id = static_cast<uint64_t>(row_id) / qwen4exp_ple_geometry::rows_per_page;
        size_t page_slot = 0;
        while (page_slot < state->unique_page_count && state->page_ids[page_slot] != page_id) {
            ++page_slot;
        }
        if (page_slot == state->unique_page_count) {
            if (state->unique_page_count == qwen4exp_ple_geometry::lookup_heads) {
                fail("early-issue request exceeds the fixed 16-page capacity");
            }
            state->page_ids[state->unique_page_count++] = page_id;
        }

        state->row_ids[i] = row_id;
        state->page_map[i] = static_cast<uint8_t>(page_slot);
        state->row_slots[i] = static_cast<uint8_t>(
                static_cast<uint64_t>(row_id) % qwen4exp_ple_geometry::rows_per_page);
    }

    {
        std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
        if (pimpl_->stopping) {
            fail("early-issue worker is stopping");
        }
        if (pimpl_->active_request != nullptr || pimpl_->queued_request != nullptr) {
            fail("previous early-issue request was not consumed before replacement");
        }
        pimpl_->active_request = state;
        pimpl_->queued_request = state;
    }
    {
        std::lock_guard<std::mutex> lock(pimpl_->cache_mutex);
        ++pimpl_->counters.issued_requests;
        pimpl_->counters.unique_pages_submitted += state->unique_page_count;
        pimpl_->counters.max_request_pages = std::max<uint64_t>(
                pimpl_->counters.max_request_pages, state->unique_page_count);
    }
    pimpl_->queue_ready.notify_one();

    return std::unique_ptr<qwen4exp_ple_request>(new qwen4exp_ple_request(std::move(state)));
}

void qwen4exp_ple_pager::join_rows(
        qwen4exp_ple_request & request,
        float *                output,
        size_t                 output_count,
        uint64_t               deadline_us) {
    if (!request.state_) {
        fail("early-issue request is empty or already consumed");
    }
    auto state = request.state_;
    const size_t expected_output = state->row_count * qwen4exp_ple_geometry::row_bytes;
    if (output == nullptr || output_count != expected_output) {
        fail("early-issue join output has incompatible element count");
    }

    bool ready_at_join = false;
    bool waited = false;
    bool deadline_missed = false;
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        ready_at_join = state->done;
        if (!state->done) {
            waited = true;
            const auto deadline = state->issued_at + std::chrono::microseconds(deadline_us);
            if (!state->ready.wait_until(lock, deadline, [&state]() { return state->done; })) {
                deadline_missed = true;
                state->ready.wait(lock, [&state]() { return state->done; });
            }
        }
        error = state->error;
    }

    {
        std::lock_guard<std::mutex> lock(pimpl_->cache_mutex);
        pimpl_->counters.ready_before_join += ready_at_join ? 1 : 0;
        pimpl_->counters.join_waits += waited ? 1 : 0;
        pimpl_->counters.deadline_misses += deadline_missed ? 1 : 0;
        pimpl_->counters.synchronous_fallbacks += deadline_missed ? 1 : 0;
    }
    {
        std::lock_guard<std::mutex> lock(pimpl_->queue_mutex);
        if (pimpl_->active_request != state) {
            fail("early-issue join does not own the pager's active request");
        }
        pimpl_->active_request.reset();
    }
    request.state_.reset();

    if (error) {
        std::rethrow_exception(error);
    }
    std::copy_n(state->output.data(), output_count, output);
}

qwen4exp_ple_pager_stats qwen4exp_ple_pager::stats() const {
    std::lock_guard<std::mutex> lock(pimpl_->cache_mutex);
    qwen4exp_ple_pager_stats result = pimpl_->counters;
    result.cache_capacity_pages = pimpl_->cache_capacity_pages;
    result.cache_resident_pages = pimpl_->slots.size();
    result.read_latency_samples  = pimpl_->read_latency_samples;
    result.read_latency_p50_ms   = pimpl_->read_latency_percentile(0.50);
    result.read_latency_p95_ms   = pimpl_->read_latency_percentile(0.95);
    result.read_latency_p99_ms   = pimpl_->read_latency_percentile(0.99);
    result.read_latency_p99_9_ms = pimpl_->read_latency_percentile(0.999);
    result.read_latency_max_ms   = static_cast<double>(pimpl_->read_latency_max_ns) / 1.0e6;
#ifdef __linux__
    result.direct_io = true;
#endif
    return result;
}
