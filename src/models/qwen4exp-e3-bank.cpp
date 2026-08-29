#include "qwen4exp-e3-bank.h"

#include <cstdio>
#include <stdexcept>

static_assert(QWEN4EXP_E3_LAYER_BYTES == 512ull * 2662400ull);
static_assert(QWEN4EXP_E3_BANK_BYTES == 65431142400ull);
static_assert(QWEN4EXP_E3_LAYER_BYTES % 4096 == 0);

qwen4exp_e3_bank_views::qwen4exp_e3_bank_views(
        ggml_backend_buffer_t bank,
        std::uint32_t n_layers) : bank_(bank) {
    if (bank_ == nullptr || n_layers == 0 || n_layers > QWEN4EXP_E3_LAYERS) {
        throw std::runtime_error("invalid E3.QR05 bank or layer count");
    }

    const std::size_t required_bytes =
        static_cast<std::size_t>(n_layers) * QWEN4EXP_E3_LAYER_BYTES;
    if (ggml_backend_buffer_get_size(bank_) != required_bytes) {
        throw std::runtime_error("E3.QR05 bank allocation has the wrong byte size");
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(bank_);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr ||
        (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU &&
         ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU)) {
        throw std::runtime_error("E3.QR05 bank must be one GPU backend allocation");
    }

    ggml_init_params ctx_params = {
        /*.mem_size   =*/ n_layers * ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(ctx_params);
    if (ctx_ == nullptr) {
        throw std::runtime_error("cannot allocate E3.QR05 layer-view metadata");
    }

    try {
        layers_.reserve(n_layers);
        auto * base = static_cast<unsigned char *>(
            ggml_backend_buffer_get_base(bank_));
        for (std::uint32_t il = 0; il < n_layers; ++il) {
            ggml_tensor * view = ggml_new_tensor_1d(
                ctx_, GGML_TYPE_I8, QWEN4EXP_E3_LAYER_BYTES);
            char name[GGML_MAX_NAME];
            std::snprintf(name, sizeof(name), "e3_qr05.layer.%02u", il);
            ggml_set_name(view, name);

            void * address = base +
                static_cast<std::size_t>(il) * QWEN4EXP_E3_LAYER_BYTES;
            if (ggml_backend_tensor_alloc(bank_, view, address) !=
                    GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot bind an E3.QR05 layer view");
            }
            layers_.push_back(view);
        }
    } catch (...) {
        ggml_free(ctx_);
        ctx_ = nullptr;
        throw;
    }
}

qwen4exp_e3_bank_views::~qwen4exp_e3_bank_views() {
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

ggml_tensor * qwen4exp_e3_bank_views::layer(std::uint32_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range("E3.QR05 layer index is out of range");
    }
    return layers_[index];
}

ggml_backend_buffer_t qwen4exp_e3_bank_views::bank() const {
    return bank_;
}

std::uint32_t qwen4exp_e3_bank_views::n_layers() const {
    return static_cast<std::uint32_t>(layers_.size());
}

std::size_t qwen4exp_e3_bank_views::allocation_bytes() const {
    return ggml_backend_buffer_get_size(bank_);
}

std::size_t qwen4exp_e3_bank_views::metadata_bytes() const {
    return ggml_used_mem(ctx_);
}
