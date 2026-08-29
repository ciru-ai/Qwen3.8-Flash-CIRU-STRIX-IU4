#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <memory>
#include <string>

// The sole owner of one persistent E3.QR05 backend allocation. Model layer
// views borrow buffer() and must be destroyed before this owner.
class qwen4exp_e3_bank_owner {
public:
    static std::unique_ptr<qwen4exp_e3_bank_owner> load_production(
        ggml_backend_buffer_type_t buft,
        const std::string & bank_root);

    ~qwen4exp_e3_bank_owner();

    qwen4exp_e3_bank_owner(const qwen4exp_e3_bank_owner &) = delete;
    qwen4exp_e3_bank_owner & operator=(const qwen4exp_e3_bank_owner &) = delete;

    ggml_backend_buffer_t buffer() const noexcept;

private:
    explicit qwen4exp_e3_bank_owner(
        ggml_backend_buffer_t buffer) noexcept;

    ggml_backend_buffer_ptr buffer_;
};
