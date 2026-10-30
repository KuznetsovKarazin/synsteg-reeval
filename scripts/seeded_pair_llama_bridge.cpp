// Pinned llama.cpp bridge. No sampling, networking or JSON parser here.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string unhex(const std::string &hex) {
    if (hex == "-") return "";
    if (hex.size() % 2) throw std::runtime_error("odd hex length");
    std::string out;
    for (size_t i = 0; i < hex.size(); i += 2) {
        auto digit = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            throw std::runtime_error("noncanonical hex");
        };
        out += char(digit(hex[i]) * 16 + digit(hex[i+1]));
    }
    return out;
}
static std::string hexbytes(const std::string &s) {
    if (s.empty()) return "-";
    const char *h="0123456789abcdef"; std::string out;
    for (unsigned char c : s) { out += h[c >> 4]; out += h[c & 15]; }
    return out;
}
static std::vector<llama_token> read_ids(std::istringstream &in) {
    int n; if (!(in >> n) || n < 0 || n > 2048) throw std::runtime_error("invalid token count");
    std::vector<llama_token> ids(n);
    for (auto &id: ids) if (!(in >> id)) throw std::runtime_error("missing token ID");
    return ids;
}
static std::string piece(const llama_vocab *vocab, llama_token id) {
    std::vector<char> buf(64);
    int n=llama_token_to_piece(vocab,id,buf.data(),buf.size(),0,false);
    if(n<0){buf.resize(-n);n=llama_token_to_piece(vocab,id,buf.data(),buf.size(),0,false);}
    if(n<0)throw std::runtime_error("token piece failed");
    return std::string(buf.data(),n);
}
int main(int argc,char **argv) {
    if(argc!=3){std::cerr<<"usage: bridge MODEL_GGUF CPU_BACKEND_SO\n";return 2;}
    llama_model *model=nullptr;llama_context *ctx=nullptr;
    try{
        if(!ggml_backend_load(argv[2]))throw std::runtime_error("CPU backend load failed");
        llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        model=llama_model_load_from_file(argv[1],mp);
        if(!model)throw std::runtime_error("model load failed");
        auto cp=llama_context_default_params();
        cp.n_ctx=2048;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;
        cp.n_threads=4;cp.n_threads_batch=4;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.offload_kqv=false;
        ctx=llama_init_from_model(model,cp);
        if(!ctx)throw std::runtime_error("context init failed");
        const llama_vocab *vocab=llama_model_get_vocab(model);
        const int nv=llama_vocab_n_tokens(vocab);
        std::vector<bool> allowed(nv);
        int masked=0;
        for(int id=0;id<nv;id++){
            allowed[id]=!llama_vocab_is_eog(vocab,id)&&!llama_vocab_is_control(vocab,id)&&!piece(vocab,id).empty();
            masked+=!allowed[id];
        }
        int past=0;
        auto eval=[&](const std::vector<llama_token>&ids){
            if(ids.empty()||past+(int)ids.size()>2048)throw std::runtime_error("invalid context token count");
            for(size_t off=0;off<ids.size();off+=512){
                int n=std::min<size_t>(512,ids.size()-off);
                auto batch=llama_batch_init(n,0,1);batch.n_tokens=n;
                for(int j=0;j<n;j++){
                    if(ids[off+j]<0||ids[off+j]>=nv)throw std::runtime_error("invalid token ID");
                    batch.token[j]=ids[off+j];batch.pos[j]=past+j;
                    batch.n_seq_id[j]=1;batch.seq_id[j][0]=0;batch.logits[j]=(j==n-1);
                }
                int rc=llama_decode(ctx,batch);llama_batch_free(batch);
                if(rc)throw std::runtime_error("llama_decode failed "+std::to_string(rc));
                past+=n;
            }
        };
        std::cout<<"READY "<<nv<<" "<<masked<<" "<<llama_n_ctx(ctx)<<"\n"<<std::flush;
        std::string line;
        while(std::getline(std::cin,line)){
            std::istringstream in(line);std::string cmd;in>>cmd;
            if(cmd=="QUIT"){std::cout<<"BYE\n"<<std::flush;break;}
            if(cmd=="TOKENIZE"){
                int add,parse;std::string raw;
                if(!(in>>add>>parse>>raw))throw std::runtime_error("bad TOKENIZE");
                auto text=unhex(raw);std::vector<llama_token>ids(text.size()+16);
                int n=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),add,parse);
                if(n<0){ids.resize(-n);n=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),add,parse);}
                if(n<0)throw std::runtime_error("tokenize failed");
                std::cout<<"TOK "<<n;for(int i=0;i<n;i++)std::cout<<" "<<ids[i];std::cout<<"\n"<<std::flush;
            }else if(cmd=="TEXT"){
                auto ids=read_ids(in);std::string raw;
                for(auto id:ids){if(id<0||id>=nv)throw std::runtime_error("bad token");raw+=piece(vocab,id);}
                std::cout<<"TEXT "<<hexbytes(raw)<<"\n"<<std::flush;
            }else if(cmd=="PROMPT"){
                auto ids=read_ids(in);llama_memory_clear(llama_get_memory(ctx),true);past=0;eval(ids);
                std::cout<<"OK "<<past<<"\n"<<std::flush;
            }else if(cmd=="STEP"){
                int id;if(!(in>>id))throw std::runtime_error("bad STEP");eval({id});
                std::cout<<"OK "<<past<<"\n"<<std::flush;
            }else if(cmd=="DIST"){
                if(!past)throw std::runtime_error("no evaluated prefix");
                const float *logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("missing logits");
                std::vector<int> ids;ids.reserve(nv-masked);
                for(int id=0;id<nv;id++)if(allowed[id]){
                    if(!std::isfinite(logits[id]))throw std::runtime_error("nonfinite allowed logit");
                    ids.push_back(id);
                }
                auto cmp=[&](int a,int b){return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);};
                int k=std::min<int>(50,ids.size());std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),cmp);ids.resize(k);
                std::cout<<"DIST "<<nv<<" "<<masked<<" "<<k;
                for(auto id:ids){uint32_t bits;std::memcpy(&bits,&logits[id],4);
                    std::cout<<" "<<id<<" "<<std::hex<<std::setw(8)<<std::setfill('0')<<bits<<std::dec;
                }
                std::cout<<"\n"<<std::flush;
            }else throw std::runtime_error("unknown command");
        }
        llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    }catch(const std::exception &e){
        std::cerr<<"BRIDGE_ERROR "<<e.what()<<"\n";
        if(ctx)llama_free(ctx);
        if(model)llama_model_free(model);
        return 1;
    }
}
