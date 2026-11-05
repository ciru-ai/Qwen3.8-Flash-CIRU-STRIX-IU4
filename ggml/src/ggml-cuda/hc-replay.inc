// Only the standalone component driver calls these diagnostic exports.
extern "C" __attribute__((visibility("default"))) void ciru_hc_replay_cache(ggml_tensor * t, ggml_tensor * cache) {
    ciru_hc_replay_caches.emplace_back(t, cache);
}
extern "C" __attribute__((visibility("default"))) void ciru_hc_replay_mark(ggml_tensor * t) {
    ggml_cuda_mmb_mark_bf16_only(t);
}
extern "C" __attribute__((visibility("default"))) void ciru_hc_replay_get(
        ggml_backend_t backend, const ggml_tensor * t, void * host, size_t offset, size_t size) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    const void * data = ggml_cuda_mmb_cache_lookup(t);
    if (!data) data = t->data;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaMemcpy(host, (const char *) data + offset, size, cudaMemcpyDeviceToHost));
}
