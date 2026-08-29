#include "qwen4exp-e3-loader.h"

#include "qwen4exp-e3-bank.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static_assert(QWEN4EXP_E3_LAYERS == 48);
static_assert(QWEN4EXP_E3_LAYER_BYTES == 1363148800ull);
static_assert(QWEN4EXP_E3_BANK_BYTES == 65431142400ull);

namespace {

fs::path numeric_layer_path(const fs::path & root, std::uint32_t layer) {
    char name[64];
    std::snprintf(name, sizeof(name), "layer-%02u.e3-qr05.bank", layer);
    return root / name;
}

fs::path numeric_receipt_path(const fs::path & root, std::uint32_t layer) {
    char name[64];
    std::snprintf(name, sizeof(name), "layer-%02u.receipt.json", layer);
    return root / name;
}

void validate_buffer_type(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || ggml_backend_buft_is_host(buft)) {
        throw std::runtime_error("E3.QR05 requires a non-host backend buffer type");
    }

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        throw std::runtime_error("E3.QR05 buffer type has no device");
    }
    const auto type = ggml_backend_dev_type(dev);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        throw std::runtime_error("E3.QR05 requires a GPU or IGPU device");
    }
    if (buft != ggml_backend_dev_buffer_type(dev)) {
        throw std::runtime_error("E3.QR05 requires the device default buffer type");
    }
}

void validate_h37_receipt(const fs::path & path, std::uint32_t layer) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("missing H37 receipt: " + path.string());
    }

    json receipt;
    stream >> receipt;
    if (receipt.value("status", std::string()) != "PASS" ||
            !receipt.contains("scope") ||
            receipt["scope"].value("layer", -1) != static_cast<int>(layer) ||
            !receipt.contains("packet") ||
            receipt["packet"].value("layer_bytes", std::uint64_t(0)) !=
                QWEN4EXP_E3_LAYER_BYTES) {
        throw std::runtime_error("invalid H37 receipt: " + path.string());
    }
}

void preflight_production_root(const fs::path & root) {
    if (!fs::is_directory(root)) {
        throw std::runtime_error("E3.QR05 bank root is not a directory: " + root.string());
    }

    // Complete admission occurs before the 60.9375-GiB allocation. Generate
    // fixed numeric names rather than enumerating a mutable directory.
    for (std::uint32_t layer = 0; layer < QWEN4EXP_E3_LAYERS; ++layer) {
        validate_h37_receipt(numeric_receipt_path(root, layer), layer);
        const fs::path bank = numeric_layer_path(root, layer);
        if (!fs::is_regular_file(bank) ||
                fs::file_size(bank) != QWEN4EXP_E3_LAYER_BYTES) {
            throw std::runtime_error("invalid E3.QR05 layer file: " + bank.string());
        }
    }
}

struct llama_model_deleter_e3 {
    void operator()(llama_model * model) const noexcept {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

} // namespace

qwen4exp_e3_bank_owner::qwen4exp_e3_bank_owner(
        ggml_backend_buffer_t buffer) noexcept : buffer_(buffer) {}

qwen4exp_e3_bank_owner::~qwen4exp_e3_bank_owner() = default;

ggml_backend_buffer_t qwen4exp_e3_bank_owner::buffer() const noexcept {
    return buffer_.get();
}

std::unique_ptr<qwen4exp_e3_bank_owner>
qwen4exp_e3_bank_owner::load_production(
        ggml_backend_buffer_type_t buft,
        const std::string & bank_root) {
    validate_buffer_type(buft);
    if (bank_root.empty()) {
        throw std::runtime_error("E3.QR05 bank root is empty");
    }
    preflight_production_root(fs::path(bank_root));

    const std::size_t max_bytes = ggml_backend_buft_get_max_size(buft);
    if (max_bytes != 0 && QWEN4EXP_E3_BANK_BYTES > max_bytes) {
        throw std::runtime_error("E3.QR05 allocation exceeds backend maximum");
    }

    ggml_backend_buffer_t raw =
        ggml_backend_buft_alloc_buffer(buft, QWEN4EXP_E3_BANK_BYTES);
    if (raw == nullptr) {
        throw std::runtime_error("cannot allocate the persistent E3.QR05 bank");
    }

    auto owner = std::unique_ptr<qwen4exp_e3_bank_owner>(
        new qwen4exp_e3_bank_owner(raw));
    if (ggml_backend_buffer_get_size(raw) != QWEN4EXP_E3_BANK_BYTES ||
            ggml_backend_buffer_get_base(raw) == nullptr) {
        throw std::runtime_error("E3.QR05 backend allocation has wrong geometry");
    }
    ggml_backend_buffer_set_usage(raw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (QWEN4EXP_E3_LAYER_BYTES >
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("E3.QR05 layer is too large for one staged read");
    }

    // Exactly one host stage is retained and overwritten for the serial loop,
    // then released before model loading begins.
    std::vector<unsigned char> stage(QWEN4EXP_E3_LAYER_BYTES);

    ggml_init_params params = {
        /*.mem_size   =*/ QWEN4EXP_E3_LAYERS * ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr upload_ctx(ggml_init(params));
    if (!upload_ctx) {
        throw std::runtime_error("cannot allocate E3.QR05 upload metadata");
    }

    auto * base = static_cast<unsigned char *>(
        ggml_backend_buffer_get_base(raw));
    for (std::uint32_t layer = 0; layer < QWEN4EXP_E3_LAYERS; ++layer) {
        const fs::path layer_path = numeric_layer_path(fs::path(bank_root), layer);
        {
            std::ifstream input(layer_path, std::ios::binary);
            if (!input) {
                throw std::runtime_error("cannot open E3.QR05 layer: " + layer_path.string());
            }
            input.read(
                reinterpret_cast<char *>(stage.data()),
                static_cast<std::streamsize>(QWEN4EXP_E3_LAYER_BYTES));
            if (input.gcount() !=
                    static_cast<std::streamsize>(QWEN4EXP_E3_LAYER_BYTES)) {
                throw std::runtime_error("short E3.QR05 layer read: " + layer_path.string());
            }
        }

        ggml_tensor * upload = ggml_new_tensor_1d(
            upload_ctx.get(), GGML_TYPE_I8, QWEN4EXP_E3_LAYER_BYTES);
        const std::size_t offset =
            static_cast<std::size_t>(layer) * QWEN4EXP_E3_LAYER_BYTES;
        if (ggml_backend_tensor_alloc(raw, upload, base + offset) !=
                GGML_STATUS_SUCCESS) {
            throw std::runtime_error("cannot bind E3.QR05 upload view");
        }
        ggml_backend_tensor_set(upload, stage.data(), 0, QWEN4EXP_E3_LAYER_BYTES);
    }

    std::vector<unsigned char>().swap(stage);
    return owner;
}

// Member order is the lifetime contract: C++ destroys model before bank.
struct llama_e3_qr05_model {
    std::array<ggml_backend_dev_t, 2> devices = { nullptr, nullptr };
    std::unique_ptr<qwen4exp_e3_bank_owner> bank;
    std::unique_ptr<llama_model, llama_model_deleter_e3> model;
};

extern "C" {

llama_e3_qr05_model * llama_e3_qr05_model_load_from_file(
        const char * path_model,
        const char * bank_root,
        ggml_backend_dev_t device,
        llama_model_params params) {
    try {
        if (path_model == nullptr || path_model[0] == '\0' ||
                bank_root == nullptr || bank_root[0] == '\0' ||
                device == nullptr) {
            throw std::runtime_error("E3.QR05 model path, bank root, and device are required");
        }
        if (params.e3_qr05_bank != nullptr) {
            throw std::runtime_error("E3.QR05 bundled load rejects a preexisting borrowed bank");
        }
        if (params.split_mode != LLAMA_SPLIT_MODE_NONE) {
            throw std::runtime_error("E3.QR05 bundled load requires split mode none");
        }
        if (params.tensor_buft_overrides != nullptr) {
            throw std::runtime_error("E3.QR05 bundled load rejects tensor buffer overrides");
        }
        if ((params.n_gpu_layers >= 0 && params.n_gpu_layers < 49) ||
                params.vocab_only || params.no_alloc) {
            throw std::runtime_error("E3.QR05 bundled load requires the complete model on one device");
        }

        auto result = std::make_unique<llama_e3_qr05_model>();
        result->devices = { device, nullptr };
        const auto buft = ggml_backend_dev_buffer_type(device);
        result->bank = qwen4exp_e3_bank_owner::load_production(buft, bank_root);

        params.devices = result->devices.data();
        params.main_gpu = 0;
        params.e3_qr05_bank = result->bank->buffer();
        result->model.reset(llama_model_load_from_file(path_model, params));
        if (!result->model) {
            throw std::runtime_error("Qwen4Exp model load failed after E3.QR05 bank admission");
        }
        return result.release();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama_e3_qr05_model_load_from_file: %s\n", error.what());
        return nullptr;
    }
}

llama_model * llama_e3_qr05_model_get(llama_e3_qr05_model * owner) {
    return owner != nullptr ? owner->model.get() : nullptr;
}

void llama_e3_qr05_model_free(llama_e3_qr05_model * owner) {
    delete owner;
}

} // extern "C"
