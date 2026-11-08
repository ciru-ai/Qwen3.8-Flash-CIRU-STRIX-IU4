from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
cu = root / 'ggml/src/ggml-cuda/ggml-cuda.cu'
header = root / 'ggml/src/ggml-cuda/common.cuh'
text = cu.read_text()
pool = text[text.index('struct ggml_cuda_pool_leg :'):text.index('// pool with virtual memory')]
start = text.index('std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(')
factory = text[start:text.index('// destroying a cuBLAS handle', start)]
declaration = re.search(r'static std::unique_ptr<ggml_cuda_pool> new_pool_for_device\([^;]+;', header.read_text()).group()
has_context = 'ggml_backend_cuda_context &' in declaration
prefix = r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>
#define GGML_CUDA_NAME "CPU-BOUNDARY"
#define GGML_ASSERT assert
#define GGML_LOG_DEBUG(...) std::fprintf(stderr,__VA_ARGS__)
#define CUDA_CHECK(call) do { if ((call)!=0) { std::fprintf(stderr,"unexpected CUDA boundary error\n"); std::abort(); } } while(0)
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2;
static bool fail_next=false;
static int physically_freed=0, synchronizations=0;
static int cudaFree(void*p){std::free(p);++physically_freed;return 0;}
static int cudaGetLastError(){return 0;}
static int cudaDeviceSynchronize(){++synchronizations;return 0;}
static void ggml_cuda_set_device(int){}
static int ggml_cuda_device_malloc(void**p,size_t n,int){
 if(fail_next){fail_next=false;*p=nullptr;return 2;}
 *p=std::malloc(n);return *p?0:2;
}
struct ggml_cuda_pool {virtual ~ggml_cuda_pool()=default;virtual void*alloc(size_t,size_t*)=0;virtual void free(void*,size_t)=0;};
struct Graph {void * instance=reinterpret_cast<void*>(1);};
struct ggml_backend_cuda_context {
 std::unordered_map<const void*,std::unique_ptr<Graph>> cuda_graphs;
 DECLARATION
};
static void seed(ggml_backend_cuda_context&ctx,int n){for(int i=1;i<=n;++i)ctx.cuda_graphs.emplace(reinterpret_cast<void*>(i),std::make_unique<Graph>());}
'''.replace('DECLARATION', declaration)
tests = r'''
static std::unique_ptr<ggml_cuda_pool> make_pool(ggml_backend_cuda_context&ctx){return ggml_backend_cuda_context::new_pool_for_device(ARGS0,0);}
static bool expected_after_free(const ggml_backend_cuda_context&ctx){
#ifdef USE_CUDA_GRAPH
 return ctx.cuda_graphs.empty();
#else
 return ctx.cuda_graphs.size()==2;
#endif
}
static bool test_reuse_preserves_graphs(){
 ggml_backend_cuda_context ctx,other;seed(ctx,2);seed(other,1);auto pool=make_pool(ctx);
 int before=physically_freed;size_t n=0;void*p=pool->alloc(64,&n);pool->free(p,n);
 size_t n2=0;void*p2=pool->alloc(64,&n2);pool->free(p2,n2);
 bool ok=p==p2&&physically_freed==before&&ctx.cuda_graphs.size()==2&&other.cuda_graphs.size()==1;
 std::printf("cached_reuse_keeps_graphs=%s\n",ok?"PASS":"FAIL");return ok;
}
static bool test_alloc_failure_invalidates_owner(){
 ggml_backend_cuda_context ctx,other;auto pool=make_pool(ctx);size_t n=0;void*p=pool->alloc(64,&n);pool->free(p,n);
 seed(ctx,2);seed(other,1);int before=physically_freed,sync_before=synchronizations;fail_next=true;
 size_t n2=0;void*p2=pool->alloc(n+256,&n2);
 bool ok=!fail_next&&physically_freed==before+1&&synchronizations==sync_before+1&&expected_after_free(ctx)&&other.cuda_graphs.size()==1;
 std::printf("allocation_failure_flush_invalidates_only_owner=%s owner_graphs=%zu other_graphs=%zu\n",ok?"PASS":"FAIL",ctx.cuda_graphs.size(),other.cuda_graphs.size());
 pool->free(p2,n2);return ok;
}
static bool test_full_pool_free_invalidates_owner(){
 ggml_backend_cuda_context ctx,other;auto pool=make_pool(ctx);std::vector<std::pair<void*,size_t>> allocations;
 for(int i=0;i<257;++i){size_t n=0;void*p=pool->alloc(64,&n);allocations.emplace_back(p,n);}
 for(int i=0;i<256;++i)pool->free(allocations[i].first,allocations[i].second);
 seed(ctx,2);seed(other,1);int before=physically_freed;
 pool->free(allocations[256].first,allocations[256].second);
 bool ok=physically_freed==before+1&&expected_after_free(ctx)&&other.cuda_graphs.size()==1;
 std::printf("full_pool_physical_free_invalidates_only_owner=%s owner_graphs=%zu other_graphs=%zu\n",ok?"PASS":"FAIL",ctx.cuda_graphs.size(),other.cuda_graphs.size());return ok;
}
int main(){bool a=test_reuse_preserves_graphs(),b=test_alloc_failure_invalidates_owner(),c=test_full_pool_free_invalidates_owner();return a&&b&&c?0:1;}
'''.replace('ARGS', 'ctx,' if has_context else '')
generated = out / 'actual-pool-test.cpp'
generated.write_text(prefix + pool + factory + tests)
receipt = {'source_root': str(root), 'source_sha256': hashlib.sha256(cu.read_bytes()).hexdigest(),
           'header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
           'scope': 'Actual extracted legacy pool class and actual factory; fake device allocation boundary only. No GPU calls.', 'variants': []}
for graphs in (True, False):
    name = 'graphs' if graphs else 'no-graphs'
    command = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter']
    if graphs:
        command += ['-DUSE_CUDA_GRAPH']
    command += [str(generated), '-o', str(out/name)]
    build = subprocess.run(command, text=True, capture_output=True)
    row = {'name': name, 'command': command, 'build_exit': build.returncode, 'build_stderr': build.stderr}
    if build.returncode == 0:
        run = subprocess.run([str(out/name)], text=True, capture_output=True)
        row.update(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
    receipt['variants'].append(row)
    print(json.dumps(row, indent=2))
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
sys.exit(0 if all(x.get('exit_code') == 0 for x in receipt['variants']) else 1)
