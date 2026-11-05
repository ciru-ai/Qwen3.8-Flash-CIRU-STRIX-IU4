// Indexer projections decide a discrete top-k selection. Short-batch F16 GEMM
// and GEMV can use different arithmetic, changing the selected blocks during
// speculative verification. Keep these small projections on the single-token
// path, including LoRA supplied by the caller; leave large prefills batched.
template <typename Project>
static ggml_tensor * qwen4exp_indexer_projection(ggml_context * ctx, ggml_tensor * input, Project project) {
    const int64_t tokens = input->ne[1];
    if (tokens <= 1 || tokens > 32 || input->ne[2] != 1 || input->ne[3] != 1) {
        return project(input);
    }
    ggml_tensor * result = nullptr;
    for (int64_t token = 0; token < tokens; ++token) {
        auto * column = ggml_view_2d(ctx, input, input->ne[0], 1, input->nb[1], token*input->nb[1]);
        auto * output = project(column);
        result = result ? ggml_concat(ctx, result, output, 1) : output;
    }
    return result;
}

// Keep short indexer dot products independent of the number of active streams
// and verification queries. Batched GEMM can round a near-tied block score
// differently from a single-query call before the discrete selection.
static ggml_tensor * qwen4exp_indexer_scores(
        ggml_context * ctx, ggml_tensor * keys, ggml_tensor * queries, int64_t heads) {
    GGML_ASSERT(queries->ne[1] % heads == 0 && keys->ne[2] == queries->ne[2]);
    const int64_t count = queries->ne[1]/heads;
    if (count > 32 || queries->ne[3] != 1 || keys->ne[3] != 1) {
        return ggml_mul_mat(ctx, keys, queries);
    }
    ggml_tensor * result = nullptr;
    for (int64_t stream = 0; stream < queries->ne[2]; ++stream) {
        auto * k = ggml_view_2d(ctx, keys, keys->ne[0], keys->ne[1], keys->nb[1], stream*keys->nb[2]);
        ggml_tensor * columns = nullptr;
        for (int64_t token = 0; token < count; ++token) {
            auto * q = ggml_view_2d(ctx, queries, queries->ne[0], heads, queries->nb[1],
                    stream*queries->nb[2] + token*heads*queries->nb[1]);
            auto * score = ggml_mul_mat(ctx, k, q);
            columns = columns ? ggml_concat(ctx, columns, score, 1) : score;
        }
        result = result ? ggml_concat(ctx, result, columns, 2) : columns;
    }
    return result;
}

// TOP_K returns an unordered set. Direct indexed attention consumes that order,
// so canonicalize the selected cell IDs before its floating-point reduction.
// This only permutes the selected cells; their scores and visibility are unchanged.
static ggml_tensor * qwen4exp_order_selected_cells(ggml_context * ctx, ggml_tensor * cells) {
    GGML_ASSERT(cells->type == GGML_TYPE_I32 && ggml_is_contiguous(cells) && cells->ne[3] == 1);
    const int64_t width = cells->ne[0], queries = cells->ne[1], streams = cells->ne[2];
    auto * order = ggml_argsort(ctx, ggml_cast(ctx, cells, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
    auto * rows = ggml_reshape_4d(ctx, cells, 1, width, queries, streams);
    return ggml_reshape_3d(ctx, ggml_get_rows(ctx, rows, order), width, queries, streams);
}

static ggml_tensor * qwen4exp_apply_compact_visibility(ggml_context *ctx,ggml_tensor *score,
        ggml_tensor *limits,int64_t blocks,int64_t first,int64_t queries) {
    GGML_ASSERT(limits->type==GGML_TYPE_I32 && score->ne[0]<=blocks && score->ne[1]==queries && score->ne[2]==1);
    auto *starts=ggml_cast(ctx,ggml_view_1d(ctx,limits,score->ne[0],0),GGML_TYPE_F32);
    auto *tails=ggml_cast(ctx,ggml_view_1d(ctx,limits,queries,(blocks+first)*sizeof(int32_t)),GGML_TYPE_F32);
    tails=ggml_reshape_2d(ctx,tails,1,queries);
    auto *visible=ggml_step(ctx,ggml_sub(ctx,ggml_repeat(ctx,tails,score),starts));
    return ggml_add(ctx,score,ggml_log(ctx,visible));
}

static ggml_tensor * qwen4exp_select_complete_blocks(ggml_context *ctx0, ggml_tensor *score,
        ggml_tensor *block_cells, ggml_tensor *tail, int64_t block_budget, int64_t r) {
    const int64_t n_blocks=score->ne[0], n_query=score->ne[1], n_stream=score->ne[2];
    GGML_ASSERT(score->type==GGML_TYPE_F32 && ggml_is_contiguous(score));
    GGML_ASSERT(block_cells->type==GGML_TYPE_I32 && block_cells->ne[0]==r*n_blocks && block_cells->ne[1]==n_stream);
    GGML_ASSERT(tail->type==GGML_TYPE_I32 && tail->ne[0]==r-1 && tail->ne[1]==n_query && tail->ne[3]==n_stream);
            GGML_ASSERT(n_blocks>=block_budget);
            ggml_tensor * blocks=ggml_cont(ctx0,ggml_top_k(ctx0,score,block_budget));
            ggml_tensor * order=ggml_argsort(ctx0,ggml_cast(ctx0,blocks,GGML_TYPE_F32),GGML_SORT_ORDER_ASC);
            ggml_tensor * blocks_view=ggml_view_4d(ctx0,blocks,1,block_budget,n_query,n_stream,
                    blocks->nb[0],blocks->nb[1],blocks->nb[2],0);
            blocks=ggml_reshape_3d(ctx0,ggml_get_rows(ctx0,blocks_view,order),block_budget,n_query,n_stream);
            ggml_tensor * score_view=ggml_view_4d(ctx0,score,1,n_blocks,n_query,n_stream,
                    score->nb[0],score->nb[1],score->nb[2],0);
            ggml_tensor * picked_score=ggml_get_rows(ctx0,score_view,blocks);
            ggml_tensor * valid=ggml_step(ctx0,ggml_scale_bias(ctx0,picked_score,1.0f,1.0f));
            ggml_tensor * cells_view=ggml_reshape_3d(ctx0,block_cells,r,n_blocks,n_stream);
            ggml_tensor * flat_blocks=ggml_reshape_2d(ctx0,blocks,block_budget*n_query,n_stream);
            ggml_tensor * cells=ggml_get_rows(ctx0,cells_view,flat_blocks);
            cells=ggml_reshape_4d(ctx0,cells,r,block_budget,n_query,n_stream);
            cells=ggml_scale_bias(ctx0,ggml_cast(ctx0,cells,GGML_TYPE_F32),1.0f,1.0f);
            cells=ggml_mul(ctx0,cells,valid);
            cells=ggml_cast(ctx0,ggml_scale_bias(ctx0,cells,1.0f,-1.0f),GGML_TYPE_I32);
            cells=ggml_reshape_4d(ctx0,cells,block_budget*r,n_query,1,n_stream);
            return ggml_concat(ctx0,cells,tail,0);
}
