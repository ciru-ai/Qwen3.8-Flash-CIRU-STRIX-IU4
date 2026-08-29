#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr std::uint32_t QWEN4EXP_E3_LAYERS = 48;
constexpr std::size_t QWEN4EXP_E3_LAYER_BYTES = 1363148800ull;
constexpr std::size_t QWEN4EXP_E3_BANK_BYTES =
    QWEN4EXP_E3_LAYERS * QWEN4EXP_E3_LAYER_BYTES;

// Metadata owner for fixed-offset I8 tensors. The backing buffer is borrowed:
// this object neither allocates nor frees the persistent expert bank.
struct qwen4exp_e3_bank_views {
    qwen4exp_e3_bank_views(
        ggml_backend_buffer_t bank,
        std::uint32_t n_layers = QWEN4EXP_E3_LAYERS);
    ~qwen4exp_e3_bank_views();

    qwen4exp_e3_bank_views(const qwen4exp_e3_bank_views &) = delete;
    qwen4exp_e3_bank_views & operator=(const qwen4exp_e3_bank_views &) = delete;

    ggml_tensor * layer(std::uint32_t index) const;
    ggml_backend_buffer_t bank() const;
    std::uint32_t n_layers() const;
    std::size_t allocation_bytes() const;
    std::size_t metadata_bytes() const;

private:
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t bank_ = nullptr;
    std::vector<ggml_tensor *> layers_;
};
