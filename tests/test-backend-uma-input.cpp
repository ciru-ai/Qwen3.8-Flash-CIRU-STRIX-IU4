#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Model an asynchronous UMA device: graph execution reads the host pointers
// only at synchronize(), so overwriting an in-flight input is deterministic.
struct queued_add { float * x; float * y; float * out; };
static std::vector<queued_add> queue;
static const char * name(ggml_backend_t) { return "Deferred-UMA"; }
static const char * dev_name(ggml_backend_dev_t) { return "Deferred-UMA"; }
static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_IGPU; }
static ggml_backend_buffer_type_t buft(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }
static bool supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return ggml_backend_buft_is_host(b); }
static bool supports_op(ggml_backend_dev_t, const ggml_tensor * t) { return t->op==GGML_OP_ADD || t->op==GGML_OP_NONE; }
static void props(ggml_backend_dev_t, ggml_backend_dev_props * p) {
    *p={};p->name="Deferred-UMA";p->description="regression fixture";p->type=GGML_BACKEND_DEVICE_TYPE_IGPU;p->caps.async=true;
}
static void synchronize(ggml_backend_t) {
    for (auto & p:queue) *p.out=*p.x+*p.y;
    queue.clear();
}
static ggml_status compute(ggml_backend_t, ggml_cgraph * g) {
    for (int i=0;i<ggml_graph_n_nodes(g);++i) {
        auto * t=ggml_graph_node(g,i);
        if (t->op==GGML_OP_ADD) queue.push_back({(float*)t->src[0]->data,(float*)t->src[1]->data,(float*)t->data});
    }
    return GGML_STATUS_SUCCESS;
}

int main() {
    setenv("GGML_SCHED_UMA_RING","1",1);
    ggml_backend_device dev{};dev.iface.get_name=dev_name;dev.iface.get_description=dev_name;dev.iface.get_type=dev_type;
    dev.iface.get_props=props;dev.iface.get_buffer_type=buft;dev.iface.supports_buft=supports_buft;dev.iface.supports_op=supports_op;
    ggml_backend gpu{};gpu.device=&dev;gpu.iface.get_name=name;gpu.iface.synchronize=synchronize;gpu.iface.graph_compute=compute;
    auto cpu=ggml_backend_cpu_init();ggml_backend_t backends[]={&gpu,cpu};
    bool passed=true;
    for (bool reuse:{true,false}) {
        auto * ctx=ggml_init({1024*1024,nullptr,true});
        auto * x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);ggml_set_input(x);
        auto * y=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);ggml_set_input(y);
        auto * sum=ggml_add(ctx,x,y);ggml_set_output(sum);
        auto * graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,sum);
        auto sched=ggml_backend_sched_new(backends,nullptr,2,128,false,true);
        ggml_backend_sched_set_tensor_backend(sched,x,cpu);ggml_backend_sched_set_tensor_backend(sched,y,cpu);
        ggml_backend_sched_set_tensor_backend(sched,sum,&gpu);
        if (!ggml_backend_sched_alloc_graph(sched,graph)) return 2;
        if (ggml_backend_sched_get_n_copies(sched)!=1) return 2;
        *(float*)x->data=3;*(float*)y->data=4;*(float*)sum->data=-1;
        if (ggml_backend_sched_graph_compute_async(sched,graph)!=GGML_STATUS_SUCCESS || queue.empty()) return 2;
        if (reuse) ggml_backend_sched_prepare_inputs(sched);
        else {
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_set_tensor_backend(sched,x,cpu);ggml_backend_sched_set_tensor_backend(sched,y,cpu);
            ggml_backend_sched_set_tensor_backend(sched,sum,&gpu);
            if (!ggml_backend_sched_alloc_graph(sched,graph)) return 2;
        }
        const bool ownership_returned=queue.empty();
        *(float*)x->data=30;*(float*)y->data=40;
        ggml_backend_sched_synchronize(sched);
        const bool value_ok=*(float*)sum->data==7;
        printf("%s: input_ownership=%d original_result=%g\n",reuse?"reused graph":"reallocated graph",int(ownership_returned),*(float*)sum->data);
        passed &= ownership_returned && value_ok;
        ggml_backend_sched_free(sched);ggml_free(ctx);
    }
    ggml_backend_free(cpu);
    return passed?0:1;
}
