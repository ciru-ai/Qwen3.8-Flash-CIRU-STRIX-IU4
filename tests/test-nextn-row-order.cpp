// Reuse the repository's tiny seeded model factory.
int arch_fixture_main(int argc, char ** argv);
#define main arch_fixture_main
#include "test-llama-archs.cpp"
#undef main

#include "../src/llama-ext.h"
#include <cmath>

static std::vector<float> capture_rows(llama_model * model, int ubatch, bool masked) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 64;
    cp.n_ubatch = ubatch;
    cp.n_seq_max = 2;
    cp.kv_unified = false;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) { throw std::runtime_error("context initialization failed"); }
    llama_set_embeddings_nextn(ctx.get(), true, masked);
    const int width = llama_model_n_embd_out(model);
    std::vector<float> result;

    // Unequal sequences split into several ubatches; only the last row has logits.
    // The second call exercises continued sequences and output-buffer reuse.
    for (int round = 0; round < 2; ++round) {
        llama_batch batch = llama_batch_init(64, 0, 1);
        const int lengths[] = {19, 11};
        std::vector<int> selected;
        for (int seq = 0; seq < 2; ++seq) {
            for (int i = 0; i < lengths[seq]; ++i) {
                const int pos = round * lengths[seq] + i;
                const bool output = i == lengths[seq] - 1;
                if (!masked || output) { selected.push_back(batch.n_tokens); }
                common_batch_add(batch, (7 * pos + 31 * seq + 1) % 128, pos, {seq}, output);
            }
        }
        if (llama_decode(ctx.get(), batch)) { throw std::runtime_error("decode failed"); }
        for (int row : selected) {
            const float * h = llama_get_embeddings_nextn_ith(ctx.get(), row);
            if (!h) { throw std::runtime_error("missing hidden row"); }
            result.insert(result.end(), h, h + width);
        }
        // Repeated getters must not permute rows a second time.
        const float * h = llama_get_embeddings_nextn_ith(ctx.get(), selected.back());
        if (!std::equal(h, h + width, result.end() - width)) {
            throw std::runtime_error("non-idempotent row reorder");
        }
        llama_batch_free(batch);
    }
    return result;
}

int main() {
    common_log_set_verbosity_thold(0);
    common_init();
    llama_backend_init();
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    auto model = get_model_and_ctx(metadata.get(), nullptr, 20261004, {});
    model.second.reset();
    bool ok = true;
    for (bool masked : {false, true}) {
        const auto reference = capture_rows(model.first.get(), 64, masked);
        const auto split = capture_rows(model.first.get(), 8, masked);
        double error = nmse(split, reference);
        float max_diff = 0;
        for (size_t i = 0; i < split.size(); ++i) {
            if (!std::isfinite(split[i]) || !std::isfinite(reference[i])) {
                throw std::runtime_error("nonfinite hidden state");
            }
            max_diff = std::max(max_diff, std::fabs(split[i] - reference[i]));
        }
        printf("masked=%d values=%zu nmse=%.9g max_abs=%.9g\n", int(masked), split.size(), error, max_diff);
        ok &= error < 1e-8 && max_diff < 1e-4;
    }
    model.first.reset();
    llama_backend_free();
    puts(ok ? "PASS: hidden rows retain original batch order" : "FAIL: hidden rows changed with ubatch splitting");
    return ok ? 0 : 1;
}
