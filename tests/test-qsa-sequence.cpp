#include "llama.h"
#include "llama-model.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <vector>

static bool check_case(int n_sequences, bool fragmented, bool shared_prefix, bool tails, bool block_bias, int offset) {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    auto & hp = model->hparams;
    hp.n_layer_all = 1; hp.n_embd = 4; hp.n_ctx_train = 64;
    hp.indexer_head_size = 4; hp.n_embd_head_k_full = 4; hp.n_embd_head_v_full = 4;
    hp.n_head_arr.fill(1); hp.n_head_kv_arr.fill(1);
    const auto no_layers = [](int32_t) { return false; };
    llama_memory_hybrid_idx mem(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 64, 1, 0,
        LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 0, false, true,
        no_layers, no_layers, no_layers);
    auto & cells = const_cast<llama_kv_cells &>(mem.get_mem_idx()->get_cells(0));
    const int ratio = 4, nkv = 64, nblocks = nkv / ratio;
    int cursor = 0;
    std::vector<int> owned[2];
    auto add = [&](int pos, std::vector<int> seqs) {
        const int cell = fragmented ? (cursor * 13 + 7 + offset) % nkv : cursor + offset;
        ++cursor; cells.pos_set(cell, pos);
        for (int seq : seqs) { cells.seq_add(cell, seq); owned[seq].push_back(cell); }
    };
    if (shared_prefix) for (int p = 0; p < 4; ++p) add(p, {0, 1});
    for (int seq = 0; seq < n_sequences; ++seq) {
        for (int p = shared_prefix ? 4 : 0; p < (tails ? 11 : 12); ++p) add(p, {seq});
    }
    llama_memory_hybrid_idx_context ctx(&mem);
    ggml_init_params gp = {16384, nullptr, true};
    ggml_context * tensors = ggml_init(gp);
    auto * map = ggml_new_tensor_2d(tensors, GGML_TYPE_I32, nkv, 1);
    auto * members = ggml_new_tensor_2d(tensors, GGML_TYPE_I32, ratio*nblocks, 1);
    auto * positions = ggml_new_tensor_1d(tensors, GGML_TYPE_I32, 4*nblocks);
    auto * bias = ggml_new_tensor_3d(tensors, GGML_TYPE_F32, block_bias ? nblocks : nkv, n_sequences, 1);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(tensors, ggml_backend_cpu_buffer_type());
    llama_seq_id seqs[] = {0, 1}; llama_seq_id * ids[] = {seqs, seqs+1};
    llama_pos pos[] = {tails ? 10 : 11, tails ? 10 : 11};
    int32_t counts[] = {1, 1};
    llama_ubatch ub {}; ub.n_tokens = n_sequences; ub.n_seqs = n_sequences; ub.n_seqs_unq = n_sequences; ub.n_pos = 1;
    ub.pos = pos; ub.n_seq_id = counts; ub.seq_id = ids;
    ctx.set_input_qsa(map, members, positions, bias, &ub, ratio, block_bias);
    const auto * maps = static_cast<const int32_t *>(map->data);
    const auto * src = static_cast<const int32_t *>(members->data);
    const auto * biases = static_cast<const float *>(bias->data);
    bool ok = true;
    for (int seq = 0; seq < n_sequences; ++seq) {
        for (int cell : owned[seq]) {
            const int p = cells.pos_get(cell);
            const int b = maps[cell];
            const float value = biases[seq*(block_bias ? nblocks : nkv)+(block_bias ? b : cell)];
            if (std::isnan(value)) ok = false;
            if (tails && p >= 8) { if (!(value > 0 && std::isfinite(value))) ok = false; continue; }
            if (!std::isfinite(value)) { ok = false; continue; }
            bool found = false;
            for (int k = 0; k < ratio; ++k) {
                const int other = src[b*ratio+k];
                if (other == cell) found = true;
                if (cells.is_empty(other) || !cells.seq_has(other, seq)) ok = false;
                // A shared prefix must not pool a private suffix, or vice versa.
                for (int s = 0; s < 2; ++s) if (cells.seq_has(other,s) != cells.seq_has(cell,s)) ok = false;
            }
            if (!found) ok = false;
        }
    }
    std::printf("sequences=%d fragmented=%d shared=%d tails=%d block_bias=%d offset=%d %s\n", n_sequences, fragmented, shared_prefix, tails, block_bias, offset, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer); ggml_free(tensors);
    return ok;
}
// A raw indexer cache has no RoPE, but must retain mirrored spatial metadata
// through the sequence-state API. Nonzero x/y catch a silent loss on restore.
struct memory_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * src, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(src);
        bytes.insert(bytes.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        std::vector<uint8_t> data(size);
        ggml_backend_tensor_get(t, data.data(), offset, size);
        write(data.data(), size);
    }
    size_t n_bytes() override { return bytes.size(); }
};
struct memory_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t cursor = 0;
    explicit memory_reader(const std::vector<uint8_t> & data) : bytes(data) {}
    void read(void * dst, size_t size) override {
        if (size > bytes.size() - cursor) { throw std::runtime_error("short state"); }
        std::memcpy(dst, bytes.data() + cursor, size);
        cursor += size;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        std::vector<uint8_t> data(size);
        read(data.data(), size);
        ggml_backend_tensor_set(t, data.data(), offset, size);
    }
    size_t n_bytes() override { return cursor; }
};
static bool check_spatial_restore(bool unified) {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    auto & hp = model->hparams;
    hp.n_layer_all = 1; hp.n_embd = 4; hp.n_ctx_train = 64;
    hp.n_embd_head_k_full = 4; hp.n_embd_head_v_full = 4;
    hp.n_head_arr.fill(1); hp.n_head_kv_arr.fill(1);
    hp.rope_type = LLAMA_ROPE_TYPE_NONE; hp.ple_n_heads = 1;
    const auto no_layers = [](int32_t) { return false; };
    llama_kv_cache cache(*model, hp, GGML_TYPE_F32, GGML_TYPE_F32, false, false, unified,
        64, 2, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, no_layers, nullptr, nullptr);
    auto & source = const_cast<llama_kv_cells &>(cache.get_cells(0));
    for (int p = 0; p < 7; ++p) {
        const int cell = p * 3 + 2;
        source.pos_set(cell, p); source.seq_add(cell, 0);
        source.ext_set(cell, {11 + p, 23 + p, 101 + p});
    }
    memory_writer saved; cache.state_write(saved, 0);
    memory_reader input(saved.bytes); cache.state_read(input, 1);
    const auto & restored = cache.get_cells(1);
    bool ok = input.n_bytes() == saved.n_bytes();
    int count = 0;
    for (uint32_t i = 0; i < restored.size(); ++i) {
        if (restored.is_empty(i) || !restored.seq_has(i, 1)) { continue; }
        ++count;
        const int p = restored.pos_get(i);
        const auto & ext = restored.ext_get(i);
        ok = ok && ext.x == 11 + p && ext.y == 23 + p && ext.tok == 101 + p;
    }
    ok = ok && count == 7;
    // The source remains intact even when the destination shares its cell pool.
    for (int p = 0; p < 7; ++p) {
        const auto & ext = source.ext_get(p * 3 + 2);
        ok = ok && source.seq_has(p * 3 + 2, 0) && ext.x == 11 + p && ext.y == 23 + p;
    }
    std::printf("spatial state restore unified=%d %s\n", unified, ok ? "PASS" : "FAIL");
    return ok;
}
static bool check_direct_token() {
    llama_kv_cells cells;
    cells.resize(8);
    cells.pos_set(2, 3); cells.seq_add(2, 0); cells.ext_set(2, {7, 9, 123});
    llama_token token = LLAMA_TOKEN_NULL;
    bool ok = cells.token_at_position(2, 0, 3, token) && token == 123;
    ok &= !cells.token_at_position(2, 1, 3, token);
    ok &= !cells.token_at_position(2, 0, 4, token);
    ok &= !cells.token_at_position(8, 0, 3, token);
    cells.pos_set(4, 3); cells.seq_add(4, 0); cells.ext_set(4, {0, 0, 456});
    ok &= !cells.token_at_position(2, 0, 3, token);
    std::printf("direct token exact position/sequence/duplicate guards %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_pool_plan(bool unified) {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    auto & hp = model->hparams;
    hp.n_layer_all = 1; hp.n_embd = 4; hp.n_ctx_train = 1024;
    hp.indexer_head_size = 4; hp.n_embd_head_k_full = 4; hp.n_embd_head_v_full = 4;
    hp.n_head_arr.fill(1); hp.n_head_kv_arr.fill(1);
    const auto no_layers = [](int32_t) { return false; };
    llama_memory_hybrid_idx mem(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 1024, 256, 0,
        LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 0, false, unified,
        no_layers, no_layers, no_layers);
    auto & idx = const_cast<llama_kv_cells &>(mem.get_mem_idx()->get_cells(0));
    auto & att = const_cast<llama_kv_cells &>(mem.get_mem_attn()->get_cells(0));
    for (int p = 0; p < 264; ++p) {
        idx.pos_set(p, p); idx.seq_add(p, 0);
        att.pos_set(p, p); att.seq_add(p, 0);
    }
    llama_pos pos[] = {260, 261, 262, 263};
    llama_seq_id seq = 0; llama_seq_id * ids[] = {&seq, &seq, &seq, &seq};
    int32_t counts[] = {1, 1, 1, 1};
    llama_ubatch ub {}; ub.n_tokens = 4; ub.n_seqs = 1; ub.n_seqs_unq = 1;
    ub.n_pos = 1; ub.pos = pos; ub.n_seq_id = counts; ub.seq_id = ids;
    llama_kv_cache::slot_info info {}; info.s0 = 0; info.s1 = 0;
    info.strm = {0}; info.idxs = {{260, 261, 262, 263}};
    auto plan = mem.plan_qsa_cached_pool(ub, info, 4);
    const char * flag = std::getenv("CIRU_QSA_POOL_CACHE");
    const bool enabled = flag && std::strcmp(flag, "1") == 0;
    bool ok = plan.enabled == enabled;
    auto * cache = mem.get_mem_idx_blocks();
    if (enabled) {
        ok &= plan.first_block == 0 && plan.total_blocks == 128;
        cache->valid_base = 0; cache->valid_blocks = 128; cache->valid_seq = 0;
        plan = mem.plan_qsa_cached_pool(ub, info, 4);
        ok &= plan.enabled && plan.first_block == 64 && plan.new_blocks == 64;
        mem.seq_cp(0, 1, unified ? 0 : -1, unified ? 4 : -1);
        ok &= cache->valid_base == -1 && cache->valid_blocks == 0;
        ok &= mem.plan_qsa_cached_pool(ub, info, 4).enabled == !unified;
        mem.seq_rm(1, -1, -1);
        ok &= mem.plan_qsa_cached_pool(ub, info, 4).enabled;
        info.idxs[0][3] = 265;
        ok &= !mem.plan_qsa_cached_pool(ub, info, 4).enabled;
        if (!unified) {
            auto & idx1 = const_cast<llama_kv_cells &>(mem.get_mem_idx()->get_cells(1));
            auto & att1 = const_cast<llama_kv_cells &>(mem.get_mem_attn()->get_cells(1));
            for (int p = 0; p < 264; ++p) {
                idx1.pos_set(p, p); idx1.seq_add(p, 1);
                att1.pos_set(p, p); att1.seq_add(p, 1);
            }
            cache->valid_base = 0; cache->valid_blocks = 128; cache->valid_seq = 0;
            seq = 1; info.s0 = 1; info.s1 = 1; info.strm = {1}; info.idxs[0][3] = 263;
            plan = mem.plan_qsa_cached_pool(ub, info, 4);
            ok &= plan.enabled && plan.seq_id == 1 && plan.first_block == 0 && plan.new_blocks == 128;
        }
    }
    std::printf("pool suffix alignment/sequence invalidation/fragmentation enabled=%d unified=%d %s\n", enabled, unified, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    int failed = 0;
    if (!check_direct_token()) ++failed;
    for (bool unified : {false, true}) if (!check_pool_plan(unified)) ++failed;
    for (int sequences : {1, 2}) for (bool fragmented : {false, true}) for (bool shared : {false, true})
        for (bool tails : {false, true}) for (bool bias : {false, true}) for (int offset : {0, 3})
            if (!check_case(sequences, fragmented, shared, tails, bias, offset)) ++failed;
    for (bool unified : {false, true}) if (!check_spatial_restore(unified)) ++failed;
    std::printf("QSA map/state/guard cases: 69, failed: %d\n", failed);
    return failed ? 1 : 0;
}
