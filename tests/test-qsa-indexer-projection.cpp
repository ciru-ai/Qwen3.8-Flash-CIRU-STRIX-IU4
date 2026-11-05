#include "ggml.h"
#include "ggml-backend.h"
#include "../src/models/block-graph.inc"

#include <cmath>
#include <cstdio>
#include <vector>

static bool test_scores(ggml_backend_t backend) {
    int checked = 0;
    float raw_max_error = 0.0f;
    for (int streams : {1, 2, 4}) for (int tokens : {2, 4, 8}) {
        constexpr int dim = 128, blocks = 2496, heads = 4;
        auto * ctx = ggml_init({16*1024*1024, nullptr, true});
        auto * keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, blocks, streams);
        auto * queries = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads*tokens, streams);
        auto * fixed = qwen4exp_indexer_scores(ctx, keys, queries, heads);
        auto * raw = ggml_mul_mat(ctx, keys, queries);
        auto * graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_set_output(fixed); ggml_set_output(raw);
        ggml_build_forward_expand(graph, fixed); ggml_build_forward_expand(graph, raw);
        auto * single_keys = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, blocks);
        auto * single_query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, heads);
        auto * single = ggml_mul_mat(ctx, single_keys, single_query);
        auto * single_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(single_graph, single);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) return false;
        std::vector<float> k(dim*blocks*streams), q(dim*heads*tokens*streams);
        for (size_t i=0; i<k.size(); ++i) k[i]=std::sin(float(i*17%997))*0.8f;
        for (size_t i=0; i<q.size(); ++i) q[i]=std::cos(float(i*31%1013))*0.7f;
        ggml_backend_tensor_set(keys,k.data(),0,k.size()*sizeof(float));
        ggml_backend_tensor_set(queries,q.data(),0,q.size()*sizeof(float));
        if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return false;
        std::vector<float> actual(blocks*heads*tokens*streams), original(actual.size()), expected(blocks*heads);
        ggml_backend_tensor_get(fixed,actual.data(),0,actual.size()*sizeof(float));
        ggml_backend_tensor_get(raw,original.data(),0,original.size()*sizeof(float));
        for (int s=0; s<streams; ++s) for (int t=0; t<tokens; ++t) {
            ggml_backend_tensor_set(single_keys,k.data()+s*dim*blocks,0,dim*blocks*sizeof(float));
            ggml_backend_tensor_set(single_query,q.data()+(s*tokens+t)*dim*heads,0,dim*heads*sizeof(float));
            if (ggml_backend_graph_compute(backend,single_graph)!=GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(single,expected.data(),0,expected.size()*sizeof(float));
            for (int i=0; i<blocks*heads; ++i) {
                const int index=(s*tokens+t)*blocks*heads+i;
                if (actual[index]!=expected[i] || !std::isfinite(actual[index])) {
                    std::fprintf(stderr,"score differs: streams=%d tokens=%d stream=%d token=%d index=%d\n",streams,tokens,s,t,i);
                    return false;
                }
                raw_max_error=std::fmax(raw_max_error,std::fabs(original[index]-expected[i]));
                ++checked;
            }
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    std::printf("score_values_checked=%d fixed_exact=1 original_max_abs=%g\n",checked,raw_max_error);
    return true;
}

// Compare every token with an independently evaluated single-token projection.
// The unfixed F16 batch operation is retained as a diagnostic negative control.
int main(int argc, char ** argv) {
    if (argc > 1) ggml_backend_load_all_from_path(argv[1]);
    else ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(argc > 1 ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!device && argc > 1) device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!device) return 2;
    auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 2;
    int checked = 0;
    float raw_max_error = 0.0f;
    for (int width : {128, 512}) {
        for (int tokens : {2, 4, 8, 16, 32}) {
            constexpr int inner = 2560;
            auto * ctx = ggml_init({16*1024*1024, nullptr, true});
            auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, inner, width);
            auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, tokens);
            auto * fixed = qwen4exp_indexer_projection(ctx, input, [&](ggml_tensor * column) {
                return ggml_mul_mat(ctx, weights, column);
            });
            auto * raw = ggml_mul_mat(ctx, weights, input);
            auto * graph = ggml_new_graph_custom(ctx, 4096, false);
            ggml_set_output(fixed); ggml_set_output(raw);
            ggml_build_forward_expand(graph, fixed); ggml_build_forward_expand(graph, raw);
            auto * single_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, 1);
            auto * single = ggml_mul_mat(ctx, weights, single_input);
            auto * single_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(single_graph, single);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) return 2;
            std::vector<ggml_fp16_t> w(inner*width);
            std::vector<float> x(inner*tokens);
            for (int i=0; i<inner*width; ++i) w[i]=ggml_fp32_to_fp16(std::sin(float(i*17%997))*0.08f);
            for (int i=0; i<inner*tokens; ++i) x[i]=std::cos(float(i*31%1013))*0.7f;
            ggml_backend_tensor_set(weights,w.data(),0,w.size()*sizeof(w[0]));
            ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 2;
            std::vector<float> actual(width*tokens), original(width*tokens), expected(width);
            ggml_backend_tensor_get(fixed,actual.data(),0,actual.size()*sizeof(float));
            ggml_backend_tensor_get(raw,original.data(),0,original.size()*sizeof(float));
            for (int t=0; t<tokens; ++t) {
                ggml_backend_tensor_set(single_input,x.data()+t*inner,0,inner*sizeof(float));
                if (ggml_backend_graph_compute(backend,single_graph)!=GGML_STATUS_SUCCESS) return 2;
                ggml_backend_tensor_get(single,expected.data(),0,width*sizeof(float));
                for (int i=0; i<width; ++i) {
                    if (actual[t*width+i]!=expected[i] || !std::isfinite(actual[t*width+i])) {
                        std::fprintf(stderr,"projection differs: width=%d tokens=%d token=%d index=%d\n",width,tokens,t,i);
                        return 1;
                    }
                    raw_max_error=std::fmax(raw_max_error,std::fabs(original[t*width+i]-expected[i]));
                    ++checked;
                }
            }
            ggml_backend_buffer_free(buffer); ggml_free(ctx);
        }
    }
    std::printf("backend=%s values_checked=%d fixed_exact=1 original_max_abs=%g\n",ggml_backend_name(backend),checked,raw_max_error);
    if (!test_scores(backend)) return 1;
    ggml_backend_free(backend);
    return 0;
}
