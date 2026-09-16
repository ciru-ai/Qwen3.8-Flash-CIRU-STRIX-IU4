#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){
    if(argc!=8){std::cerr<<"usage: capture_logits BACKEND_DIR MODEL PLE_OR_builtin PANEL OUT LABEL WINDOWS\n";return 2;}
    try{
        const fs::path modelpath=argv[2], panel=argv[4], out=argv[5];
        const std::string ple=argv[3],label=argv[6];const int windows=std::stoi(argv[7]);
        if(windows<1||windows>16||fs::exists(out))throw std::runtime_error("invalid windows or existing output");
        fs::create_directories(out);
        if(fs::file_size(panel)!=16*2049*sizeof(int32_t))throw std::runtime_error("panel size mismatch");
        std::vector<llama_token> tokens(16*2049);
        std::ifstream input(panel,std::ios::binary);input.read(reinterpret_cast<char*>(tokens.data()),tokens.size()*sizeof(llama_token));
        if(!input)throw std::runtime_error("short token read");
        ggml_backend_load_all_from_path(argv[1]);llama_backend_init();
        bool gpu=false;json devices=json::array();
        for(size_t i=0;i<ggml_backend_dev_count();++i){auto d=ggml_backend_dev_get(i);auto type=ggml_backend_dev_type(d);if(type==GGML_BACKEND_DEVICE_TYPE_GPU||type==GGML_BACKEND_DEVICE_TYPE_IGPU)gpu=true;devices.push_back({{"name",ggml_backend_dev_name(d)},{"description",ggml_backend_dev_description(d)},{"device_type",int(type)}});}
        if(!gpu)throw std::runtime_error("GPU unavailable");
        llama_model_params mp=llama_model_default_params();mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.main_gpu=0;mp.load_mtp=false;
        mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_DIRECT;
        llama_model_tensor_buft_override overrides[]={{"per_layer_token_embd.weight",ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))},{nullptr,nullptr}};
        mp.tensor_buft_overrides=overrides;
        if(ple!="builtin"){mp.ple_sidecar=ple.c_str();mp.ple_cache_bytes=512ULL*1024*1024;}
        const auto began=Clock::now();
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(modelpath.c_str(),mp),llama_model_free);
        if(!model)throw std::runtime_error("model load failed");
        auto vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
        if(nv!=248320)throw std::runtime_error("vocabulary size mismatch");
        std::ofstream raw(out/"logits.f32le",std::ios::binary);json records=json::array();size_t stored=0;
        for(int w=0;w<windows;++w){
            auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;cp.n_rs_seq=0;
            cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.n_threads=8;cp.n_threads_batch=8;cp.offload_kqv=true;cp.op_offload=true;cp.kv_unified=false;cp.embeddings=false;
            std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
            if(!ctx)throw std::runtime_error("context creation failed");
            std::vector<char> text(65536);
            int n=llama_detokenize(vocab,tokens.data()+w*2049,2049,text.data(),text.size(),false,true);
            if(n<0)throw std::runtime_error("detokenization failed");
            std::ofstream(out/("window-"+std::to_string(w)+".txt"),std::ios::binary).write(text.data(),n);
            auto batch=llama_batch_init(512,0,1);const auto start=Clock::now();
            if(!batch.token||!batch.pos||!batch.logits)throw std::runtime_error("batch allocation failed");
            for(int offset=0;offset<2048;offset+=512){
                batch.n_tokens=512;
                for(int j=0;j<512;++j){batch.token[j]=tokens[w*2049+offset+j];batch.pos[j]=offset+j;batch.n_seq_id[j]=1;batch.seq_id[j][0]=0;batch.logits[j]=offset>=1024;}
                const int status=llama_decode(ctx.get(),batch);llama_synchronize(ctx.get());
                if(status){llama_batch_free(batch);throw std::runtime_error("decode status "+std::to_string(status));}
                if(offset>=1024){
                    for(int j=0;j<512;++j){
                        const float* logits=llama_get_logits_ith(ctx.get(),j);
                        if(!logits)throw std::runtime_error("missing logits");
                        for(int k=0;k<nv;++k)if(!std::isfinite(logits[k]))throw std::runtime_error("nonfinite logits");
                        raw.write(reinterpret_cast<const char*>(logits),nv*sizeof(float));++stored;
                    }
                    raw.flush();if(!raw)throw std::runtime_error("logit output failure");
                }
            }
            llama_batch_free(batch);
            const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
            json row={{"window",w},{"input_tokens",2048},{"scored_positions",1024},{"seconds",seconds}};records.push_back(row);std::cout<<row.dump()<<std::endl;
        }
        raw.close();const uint64_t expected=uint64_t(windows)*1024*nv*sizeof(float);
        if(stored!=size_t(windows)*1024||fs::file_size(out/"logits.f32le")!=expected)throw std::runtime_error("output shape mismatch");
        json result={{"status","PASS"},{"label",label},{"model",modelpath.string()},{"model_bytes",fs::file_size(modelpath)},
          {"ple",ple},{"output_shape",{windows,1024,nv}},{"dtype","F32"},{"output_bytes",expected},{"windows",records},{"devices",devices},
          {"settings",{{"context",2048},{"batch",512},{"ubatch",512},{"kv_k","F16"},{"kv_v","F16"},{"threads",8},{"flash_attention",true},{"mtp",false},{"new_context_per_window",true},{"gpu_layers","all"},{"load_mode","none"},{"lazy_mode","on-direct"},{"ple_placement","CPU or exact sidecar"},{"sampling","none: teacher-forced logits"}}},
          {"seconds",std::chrono::duration<double>(Clock::now()-began).count()}};
        std::ofstream(out/"manifest.json")<<result.dump(2)<<'\n';std::cout<<json({{"status","PASS"},{"vectors",stored}}).dump()<<std::endl;
        model.reset();llama_backend_free();return 0;
    }catch(const std::exception&e){std::cerr<<"capture failed: "<<e.what()<<std::endl;return 1;}
}
