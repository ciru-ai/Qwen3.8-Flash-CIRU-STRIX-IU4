"""CPU structural execution of actual host dispatch predicates; no GPU math mocked as validated.

Catches the demonstrated T9-16 grouped-path selection, mismatched bypass/admission,
and accidental loss of existing geometry/layout boundaries. Extracted production
expressions execute in C++; the fixture supplies tensor metadata, not GPU results.
"""
from pathlib import Path
import os,re,shlex,subprocess
import pytest
R=Path(__file__).resolve().parents[1]
ORIGINAL=R/'ggml/src/ggml-cuda'

def body(text,start,opening='{',closing='}'):
    a=text.index(opening,start);depth=1;i=a+1
    while depth:
        if text[i]==opening:depth+=1
        elif text[i]==closing:depth-=1
        i+=1
    return text[a+1:i-1],i

def function(text,name):
    start=text.index('static bool '+name+'(');_,end=body(text,start)
    return text[start:end]

@pytest.fixture(scope='session')
def executable(tmp_path_factory):
    s=Path(os.environ.get('C1_DISPATCH_SOURCE',str(ORIGINAL/'c1.cu'))).read_text();g=(ORIGINAL/'c1-grouped.cuh').read_text()
    group=g[g.index('    const ggml_tensor * w',g.index('static bool run(')):g.index('    const int all_routes',g.index('static bool run('))]
    dispatch=s[s.index('void ggml_cuda_mul_mat_id_c1('):]
    group_start=dispatch.index('if (',dispatch.index('#if defined(__HIP_PLATFORM_AMD__)'))
    group_expr,_=body(dispatch,group_start,'(',')')
    vector_start=dispatch.index('if (',dispatch.index('#if defined(__HIP_PLATFORM_AMD__)',group_start))
    vector_expr,_=body(dispatch,vector_start,'(',')')
    prefix=dispatch[dispatch.index('    const ggml_tensor * w'):dispatch.index('    static int dispatch_hits')]
    helpers=''
    if 'static bool c1_small_batch_vector_eligible(' in s:
        helpers=function(s,'c1_small_batch_vector_eligible')
    declaration=''
    m=re.search(r'    const bool small_batch_vector = [^;]+;',dispatch)
    if m:declaration=m.group(0)
    r4=(ORIGINAL/'c1-vector-r4.cuh').read_text()
    code='''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
struct ggml_tensor {int64_t ne[4]={}; size_t nb[4]={}; void *data=(void*)16; int type=0; bool contiguous=true; const ggml_tensor *src[5]={};};
struct ggml_backend_cuda_context {};
bool ggml_is_contiguous(const ggml_tensor*t){return t->contiguous;}
#define GGML_ASSERT(x) ((void)0)
#define GGML_TYPE_C1_G16_K2560 1
#define GGML_TYPE_C1_G16_K640 2
#define GGML_TYPE_F32 3
#define GGML_TYPE_I32 4
#define GGML_TYPE_F16 5
bool c1_use_same_value_iu8(){return false;}
bool c1_use_centered_activation(){return true;}
'''+function(r4,'c1_use_vector_r4')+'\n'+function(r4,'c1_use_vector_r4_lut')+'\n'+helpers+'\nnamespace c1_grouped {bool run(ggml_backend_cuda_context &,ggml_tensor*dst,bool,bool){'+group+'return true;}}\n'
    code+='int select(ggml_tensor*dst){ggml_backend_cuda_context ctx;'+prefix+declaration+'if('+group_expr+')return 1;if('+vector_expr+')return 2;return 0;}\n'
    code+='''int main(int argc,char**argv){
 ggml_tensor w,x,ids,rows,tables,d; int t=atoi(argv[1]); int bad=atoi(argv[2]);
 w.ne[0]=2560;w.ne[1]=640;w.ne[2]=512;w.nb[1]=1440;w.nb[2]=921600;
 x.ne[0]=2560;x.ne[1]=1;x.ne[2]=t;x.nb[0]=4;
 ids.ne[0]=10;ids.ne[1]=t;ids.nb[0]=4;ids.nb[1]=2048;
 d.ne[0]=640;d.ne[1]=10;d.ne[2]=t;d.nb[0]=4;
 rows.ne[0]=640;rows.ne[1]=512;tables.ne[0]=2;tables.ne[1]=256;tables.ne[2]=512;
 d.src[0]=&w;d.src[1]=&x;d.src[2]=&ids;d.src[3]=&rows;d.src[4]=&tables;
 if(argc>3 && atoi(argv[3])){w.ne[0]=640;w.ne[1]=2560;w.nb[1]=360;w.nb[2]=921600;x.ne[0]=640;d.ne[0]=2560;rows.ne[0]=2560;}
 if(bad==1)w.contiguous=false;if(bad==2)x.contiguous=false;if(bad==3)ids.nb[0]=8;
 if(bad==4)ids.nb[1]=2049;if(bad==5)d.contiguous=false;if(bad==6)rows.contiguous=false;
 if(bad==7)tables.contiguous=false;if(bad==8)w.ne[2]=511;if(bad==9)x.ne[1]=2;
 if(bad==10)w.ne[0]=2559;if(bad==11)w.data=(void*)17;if(bad==12)w.nb[1]=1441;
 if(bad==13)w.nb[2]=921601;if(bad==14)d.ne[1]=0;
 printf("%d\\n",select(&d));}
'''
    p=tmp_path_factory.mktemp('dispatch');(p/'test.cpp').write_text(code)
    subprocess.run(shlex.split(os.environ.get('CXX', 'c++'))+['-std=c++17','-Wall','-Wextra','-Wno-unused-function','-Wno-unused-variable','-Wno-unused-parameter','-Wno-misleading-indentation',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    return p/'test'

def route(executable,t,bad=0,enabled='1',r4='1',lut='0',reverse=False):
    env=dict(os.environ,LLAMA_C1_VECTOR_R4_SMALL_BATCH=enabled,LLAMA_C1_VECTOR_R4=r4,LLAMA_C1_VECTOR_R4_LUT=lut)
    return int(subprocess.check_output([str(executable),str(t),str(bad),str(int(reverse))],env=env,text=True))

@pytest.mark.parametrize('tokens',[9,12,16])
def test_opt_in_joint_verification_uses_existing_vector(executable,tokens):
    assert route(executable,tokens)==2

@pytest.mark.parametrize('tokens,want',[(1,2),(4,2),(8,2),(9,1),(16,1),(17,1),(512,1)])
def test_default_route_unchanged(executable,tokens,want):
    assert route(executable,tokens,enabled='0')==want

@pytest.mark.parametrize('tokens,want',[(1,2),(4,2),(8,2),(17,1),(512,1)])
def test_opt_in_is_bounded_without_changing_original_small_path(executable,tokens,want):
    assert route(executable,tokens)==want

@pytest.mark.parametrize('bad',range(1,15))
def test_new_small_path_preserves_grouped_eligibility_guards(executable,bad):
    assert route(executable,16,bad=bad)==0

def test_requires_existing_r4_variant(executable):
    assert route(executable,16,r4='0')==1

def test_preserves_existing_lut_variant_when_selected(executable):
    assert route(executable,16,r4='0',lut='1')==2

@pytest.mark.parametrize('tokens,enabled,want',[(8,'0',2),(8,'1',2),(9,'0',1),(9,'1',2),(16,'0',1),(16,'1',2),(17,'0',1),(17,'1',1)])
def test_down_projection_same_boundary(executable,tokens,enabled,want):
    assert route(executable,tokens,enabled=enabled,reverse=True)==want


def profile_environment(**overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "CIRU_"))}
    env.update(overrides)
    script = 'set -a; source "$1/profiles/strix-halo-production.env"; source "$1/profiles/strix-halo-v5.env"; env -0'
    raw = subprocess.check_output(['bash', '-c', script, 'profile-test', str(R)], env=env)
    return dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)


def test_production_profile_selects_small_batch_vector(executable):
    env = profile_environment()
    assert int(subprocess.check_output([str(executable), '16', '0', '0'], env=env)) == 2


def test_production_profile_opt_out_restores_grouped_path(executable):
    env = profile_environment(LLAMA_C1_VECTOR_R4_SMALL_BATCH='0')
    assert int(subprocess.check_output([str(executable), '16', '0', '0'], env=env)) == 1
