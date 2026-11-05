#include "fixture.h"
#include <fstream>

static std::vector<std::pair<ggml_tensor *, ggml_tensor *>> ciru_hc_replay_caches;
static void ciru_hc_replay_prime(ggml_backend_cuda_context & ctx) {
    GGML_ASSERT(ciru_hc_replay_caches.size() <= 1);
    for (const auto & entry : ciru_hc_replay_caches) {
        uint16_t * dst = ggml_cuda_mmb_cache_reserve(ctx, entry.first, ggml_nelements(entry.first));
        GGML_ASSERT(dst);
        CUDA_CHECK(cudaMemcpyAsync(dst, entry.second->data, ggml_nelements(entry.first) * 2, cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

static std::vector<ggml_tensor *> ciru_hc_capture_tensors;
static int ciru_hc_capture_end = -1;
static bool ciru_hc_capture_done = false;
static std::string ciru_hc_capture_root;

static void ciru_hc_save_bytes(ggml_backend_cuda_context & ctx, const std::string & name, const void * data, size_t n) {
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    std::ofstream f(ciru_hc_capture_root + "/" + name, std::ios::binary);
    GGML_ASSERT(f);
    std::vector<char> b(4 * 1024 * 1024);
    for (size_t p = 0; p < n; p += b.size()) {
        const size_t size = std::min(b.size(), n - p);
        CUDA_CHECK(cudaMemcpy(b.data(), (const char *) data + p, size, cudaMemcpyDeviceToHost));
        f.write(b.data(), size);
    }
    GGML_ASSERT(f);
}

static bool ciru_hc_capture_output(const ggml_tensor * t) {
    return strcmp(t->name, "ffn_out-1") == 0 || strcmp(t->name, "l_last-1") == 0 || strcmp(t->name, "hc_norm-2") == 0;
}

static void ciru_hc_capture_step(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int at) {
    const char * dir = getenv("CIRU_HC_FFN_CAPTURE");
    if (!dir || ciru_hc_capture_done) return;
    if (ciru_hc_capture_end >= 0 && at > ciru_hc_capture_end) {
        for (size_t i = 0; i < ciru_hc_capture_tensors.size(); ++i) {
            const ggml_tensor * t = ciru_hc_capture_tensors[i];
            if (!ciru_hc_capture_output(t)) continue;
            const uint16_t * cache = ggml_cuda_mmb_cache_lookup(t);
            const bool bf16 = ggml_cuda_mmb_is_bf16_only(t);
            const void * ptr = bf16 && cache ? cache : t->data;
            ciru_hc_save_bytes(ctx, "out-" + std::to_string(i) + ".bin", ptr, bf16 ? ggml_nelements(t) * 2 : ggml_nbytes(t));
            fprintf(stderr, "CIRU_HC_CAPTURE_OUT id=%zu name=%s bf16=%d cache=%d\n", i, t->name, bf16, cache != nullptr);
        }
        ciru_hc_capture_done = true;
        return;
    }
    const ggml_tensor * first = graph->nodes[at];
    if (ciru_hc_capture_end >= 0 || strcmp(first->name, "ffn_moe_logits-1") != 0 || first->ne[1] != 16384) return;
    GGML_ASSERT(!ciru_hc_merge_enabled());
    ciru_hc_capture_root = dir;
    int end = at;
    while (end < graph->n_nodes && strcmp(graph->nodes[end]->name, "hc_norm-2") != 0) ++end;
    GGML_ASSERT(end < graph->n_nodes && end - at < 100);
    ciru_hc_capture_end = end;
    std::unordered_map<const ggml_tensor *, int> ids;
    for (int i = at; i <= end; ++i) {
        ids[graph->nodes[i]] = ciru_hc_capture_tensors.size();
        ciru_hc_capture_tensors.push_back(graph->nodes[i]);
    }
    const int count = ciru_hc_capture_tensors.size();
    for (int i = 0; i < count; ++i) {
        ggml_tensor * t = ciru_hc_capture_tensors[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = t->src[s];
            if (src && !ids.count(src)) { ids[src] = ciru_hc_capture_tensors.size(); ciru_hc_capture_tensors.push_back(src); }
        }
        if (t->view_src && !ids.count(t->view_src)) { ids[t->view_src] = ciru_hc_capture_tensors.size(); ciru_hc_capture_tensors.push_back(t->view_src); }
    }
    std::ofstream meta(ciru_hc_capture_root + "/graph.bin", std::ios::binary);
    uint32_t header[] = {1, uint32_t(sizeof(ciru_hc_record)), uint32_t(count), uint32_t(ciru_hc_capture_tensors.size())};
    meta.write((const char *) header, sizeof(header));
    for (size_t i = 0; i < ciru_hc_capture_tensors.size(); ++i) {
        const ggml_tensor * t = ciru_hc_capture_tensors[i];
        const ggml_tensor * root = t->view_src ? t->view_src : t;
        const uint16_t * cache = ggml_cuda_mmb_cache_lookup(t);
        ciru_hc_record r = {};
        r.type = t->type; r.op = t->op; r.flags = t->flags; r.external = i >= size_t(count);
        r.bf16 = ggml_cuda_mmb_is_bf16_only(root); r.cache = cache != nullptr; r.output = ciru_hc_capture_output(t);
        for (int j = 0; j < 4; ++j) { r.ne[j] = t->ne[j]; r.nb[j] = t->nb[j]; }
        memcpy(r.params, t->op_params, sizeof(r.params));
        for (int j = 0; j < GGML_MAX_SRC; ++j) r.src[j] = !r.external && t->src[j] ? ids.at(t->src[j]) : -1;
        r.view = !r.external && t->view_src ? ids.at(t->view_src) : -1;
        r.view_offs = r.external ? 0 : t->view_offs;
        r.bytes = ggml_nbytes(t); r.data_address = (uintptr_t) t->data; r.cache_address = (uintptr_t) cache;
        memcpy(r.name, t->name, sizeof(r.name));
        meta.write((const char *) &r, sizeof(r));
        if (r.external) {
            const void * ptr = r.bf16 && cache ? cache : t->data;
            ciru_hc_save_bytes(ctx, "in-" + std::to_string(i) + ".bin", ptr, r.bf16 ? ggml_nelements(t) * 2 : r.bytes);
            if (cache) ciru_hc_save_bytes(ctx, "cache-" + std::to_string(i) + ".bin", cache, ggml_nelements(t) * 2);
        }
    }
    GGML_ASSERT(meta);
    fprintf(stderr, "CIRU_HC_CAPTURE_BEGIN nodes=%d tensors=%zu graph_end=%d\n", count, ciru_hc_capture_tensors.size(), end);
}
