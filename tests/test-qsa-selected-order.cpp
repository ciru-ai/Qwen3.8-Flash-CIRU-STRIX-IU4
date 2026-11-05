#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../src/models/block-graph.inc"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include <limits>

static bool test_complete_blocks(ggml_backend_t backend) {
    constexpr int streams=4, queries=3, blocks=521, ratio=4, budget=512, width=budget*ratio+ratio-1;
    auto * ctx=ggml_init({32*1024*1024,nullptr,true});
    auto * scores=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,blocks,queries,streams);
    auto * members=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,blocks*ratio,streams);
    auto * tails=ggml_new_tensor_4d(ctx,GGML_TYPE_I32,ratio-1,queries,1,streams);
    auto * selected=qwen4exp_select_complete_blocks(ctx,scores,members,tails,budget,ratio);
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,selected);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if (!buffer) return false;
    std::vector<float> values(blocks*queries*streams);
    std::vector<int32_t> cells(blocks*ratio*streams), tail((ratio-1)*queries*streams);
    for (int s=0;s<streams;++s) {
        for (int i=0;i<blocks*ratio;++i) cells[s*blocks*ratio+i]=(i+s*37)%(blocks*ratio);
        for (int q=0;q<queries;++q) {
            for (int b=0;b<blocks;++b) values[(s*queries+q)*blocks+b]=
                    q==0 && b%20==0 ? -std::numeric_limits<float>::infinity() : float((b*17+q*7+s*11)%blocks)*0.125f;
            for (int j=0;j<ratio-1;++j) tail[(s*queries+q)*(ratio-1)+j]=j<q ? 3000+s*100+q*10+j : -1;
        }
    }
    ggml_backend_tensor_set(scores,values.data(),0,values.size()*sizeof(float));
    ggml_backend_tensor_set(members,cells.data(),0,cells.size()*sizeof(int32_t));
    ggml_backend_tensor_set(tails,tail.data(),0,tail.size()*sizeof(int32_t));
    if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return false;
    std::vector<int32_t> actual(width*queries*streams);
    ggml_backend_tensor_get(selected,actual.data(),0,actual.size()*sizeof(int32_t));
    for (int s=0;s<streams;++s) for (int q=0;q<queries;++q) {
        std::vector<int> order(blocks);std::iota(order.begin(),order.end(),0);
        const int offset=(s*queries+q)*blocks;
        std::sort(order.begin(),order.end(),[&](int a,int b) {return values[offset+a]>values[offset+b];});
        std::vector<int32_t> expected, observed;
        for (int i=0;i<budget;++i) for (int j=0;j<ratio;++j) expected.push_back(
                std::isfinite(values[offset+order[i]]) ? cells[s*blocks*ratio+order[i]*ratio+j] : -1);
        const auto begin=actual.begin()+(s*queries+q)*width;
        observed.assign(begin,begin+budget*ratio);
        std::sort(expected.begin(),expected.end());std::sort(observed.begin(),observed.end());
        if (expected!=observed) return false;
        for (int j=0;j<ratio-1;++j) if (begin[budget*ratio+j]!=tail[(s*queries+q)*(ratio-1)+j]) return false;
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
    std::printf("complete_block_tail_stream_rows=%d pass=1\n",streams*queries);
    return true;
}

// Selection membership must survive canonicalization, including masked/tied
// scores. Repeated GPU TOP_K calls may reorder the original selected set.
int main(int argc, char ** argv) {
    if (argc > 1) ggml_backend_load_all_from_path(argv[1]);
    else ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_type(argc > 1 ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev && argc > 1) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev) return 2;
    auto backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 2;
    int checks = 0, unordered_changes = 0;
    int sort_rows = 0;
    for (int cols : {7, 1023, 1024, 1025, 2051, 4096, 4097}) {
        for (auto direction : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
            auto * ctx = ggml_init({32*1024*1024, nullptr, true});
            auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 3);
            auto * output = ggml_argsort(ctx, input, direction);
            auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) return 2;
            std::vector<float> values(cols*3);
            for (int i=0; i<cols*3; ++i) values[i] = float((i*31)%71)-35;
            ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
            std::vector<int32_t> indices(cols*3);
            ggml_backend_tensor_get(output, indices.data(), 0, indices.size()*sizeof(int32_t));
            for (int row=0; row<3; ++row) {
                std::vector<bool> seen(cols);
                for (int col=0; col<cols; ++col) {
                    int id=indices[row*cols+col];
                    if (id < 0 || id >= cols || seen[id]) return 1;
                    seen[id]=true;
                    if (col) {
                        float previous=values[row*cols+indices[row*cols+col-1]], current=values[row*cols+id];
                        if (direction==GGML_SORT_ORDER_ASC ? previous>current : previous<current) return 1;
                    }
                }
                ++sort_rows;
            }
            ggml_backend_buffer_free(buffer); ggml_free(ctx);
        }
    }
    for (bool tied : {false, true}) {
        const int cols = 9216, rows = argc > 1 ? 720 : 3, width = 2051;
        auto * ctx = ggml_init({32*1024*1024, nullptr, true});
        auto * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        auto * selected = ggml_cont(ctx, ggml_top_k(ctx, scores, width));
        ggml_set_output(selected);
        auto * ordered = qwen4exp_order_selected_cells(ctx, selected);
        ggml_set_output(ordered);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ordered);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) return 2;
        std::vector<float> input(cols*rows);
        for (int r=0; r<rows; ++r) for (int c=0; c<cols; ++c) {
            input[r*cols+c] = tied ? float((c/4*17+r*31)%503) : float((c*17+r*31)%cols);
            if (c >= cols-1024) input[r*cols+c] = -std::numeric_limits<float>::infinity();
        }
        ggml_backend_tensor_set(scores, input.data(), 0, input.size()*sizeof(float));
        std::vector<int32_t> previous, first_ordered, actual(width*rows), raw(width*rows);
        for (int pass=0; pass<2; ++pass) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
            ggml_backend_tensor_get(ordered, actual.data(), 0, actual.size()*sizeof(int32_t));
            ggml_backend_tensor_get(selected, raw.data(), 0, raw.size()*sizeof(int32_t));
            for (int r=0; r<rows; ++r) {
                auto sorted = std::vector<int32_t>(raw.begin()+r*width, raw.begin()+(r+1)*width);
                std::sort(sorted.begin(), sorted.end());
                if (!std::equal(sorted.begin(), sorted.end(), actual.begin()+r*width)) {
                    std::fprintf(stderr,"canonical order changed membership: tied=%d row=%d pass=%d\n",int(tied),r,pass); return 1;
                }
                if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return 1;
                std::vector<int32_t> expected(cols); std::iota(expected.begin(), expected.end(), 0);
                std::partial_sort(expected.begin(), expected.begin()+width, expected.end(), [&](int a, int b) {
                    float x=input[r*cols+a], y=input[r*cols+b]; return x == y ? a < b : x > y;
                });
                const float threshold = input[r*cols+expected[width-1]];
                for (int id : sorted) {
                    if (id < 0 || id >= cols || input[r*cols+id] < threshold) return 1;
                }
                for (int id=0; id<cols; ++id) {
                    if (input[r*cols+id] > threshold && !std::binary_search(sorted.begin(),sorted.end(),id)) return 1;
                }
                ++checks;
            }
            if (pass == 0) { previous = raw; first_ordered = actual; }
            else { unordered_changes += previous != raw; if (actual != first_ordered) return 1; }
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    std::printf("backend=%s sort_rows=%d rows_checked=%d exact_membership=1 repeat_order_exact=1 unordered_changes=%d\n",
                ggml_backend_name(backend), sort_rows, checks, unordered_changes);
    if (!test_complete_blocks(backend)) return 1;
    ggml_backend_free(backend);
    return 0;
}
