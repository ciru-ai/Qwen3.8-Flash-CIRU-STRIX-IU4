static bool ciru_hc_merge_enabled() {
    static const bool enabled = getenv("CIRU_HC_SHARED_FIRST") && atoi(getenv("CIRU_HC_SHARED_FIRST")) == 1;
    return enabled;
}

static bool ciru_hc_exact_merge(const ggml_tensor * t) {
    return ciru_hc_merge_enabled() && t->op == GGML_OP_ADD && strncmp(t->name, "ffn_out-", 8) == 0;
}

static void ciru_hc_schedule_shared(ggml_cgraph * graph) {
    if (!ciru_hc_merge_enabled()) return;
    std::unordered_map<const ggml_tensor *, int> pos;
    for (int i = 0; i < graph->n_nodes; ++i) pos[graph->nodes[i]] = i;
    auto index = [&](const ggml_tensor * t) { auto it = pos.find(t); return it == pos.end() ? -1 : it->second; };
    unsigned moved = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_cuda_moe_weighted_reduction_match mm;
        if (graph->nodes[i]->op != GGML_OP_MUL || !ggml_cuda_match_moe_weighted_reduction(graph, i, mm)) continue;
        const ggml_tensor * ex = mm.experts;
        if (ex->op != GGML_OP_MUL_MAT_ID || !ex->src[3] || !ex->src[4] || ex->ne[2] < 512 || ex->ne[0] != 2560) continue;
        const ggml_tensor * glu = ex->src[1];
        if (!glu || glu->op != GGML_OP_GLU || !glu->src[0] || !glu->src[1] ||
                glu->src[0]->op != GGML_OP_MUL_MAT_ID || glu->src[1]->op != GGML_OP_MUL_MAT_ID) continue;
        const int at = std::min(index(glu->src[0]), index(glu->src[1]));
        const int begin = i + mm.node_count;
        if (at < 0 || at >= i || begin >= graph->n_nodes) continue;
        int end = -1;
        for (int j = begin; j < graph->n_nodes && j < begin + 16; ++j) {
            const ggml_tensor * t = graph->nodes[j];
            if (ciru_hc_exact_merge(t) && (t->src[0] == mm.dst || t->src[1] == mm.dst)) { end = j; break; }
        }
        if (end <= begin) continue;
        const ggml_tensor * merge = graph->nodes[end]->src[0] == mm.dst ? graph->nodes[end]->src[1] : graph->nodes[end]->src[0];
        if (index(merge) != end - 1 || merge->op != GGML_OP_MUL ||
                strncmp(merge->name, "ffn_shexp_gated-", 16) != 0) continue;
        bool closed = true;
        for (int j = begin; j < end && closed; ++j) {
            const ggml_tensor * t = graph->nodes[j];
            if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_GLU && t->op != GGML_OP_UNARY && t->op != GGML_OP_MUL) { closed = false; break; }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (!t->src[s]) continue;
                const int p = index(t->src[s]);
                if (p >= at && !(p >= begin && p < j)) { closed = false; break; }
            }
            const int v = index(t->view_src);
            if (v >= at && !(v >= begin && v < j)) closed = false;
        }
        if (!closed) continue;
        // All moved inputs are ready at `at`; allocation follows this order.
        std::rotate(graph->nodes + at, graph->nodes + begin, graph->nodes + end);
        for (int j = at; j < end; ++j) pos[graph->nodes[j]] = j;
        ++moved;
        i = end;
    }
    if (moved && getenv("CIRU_HC_MERGE_LOG")) fprintf(stderr, "CIRU_HC_SHARED_FIRST moved=%u nodes=%d\n", moved, graph->n_nodes);
}
