#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "global_state/forward_context.hpp"
#include "layers/attention/backends/attention_layer.hpp"
#include "layers/attention/backends/flash_attn.hpp"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <cstddef>
#include <algorithm>
#include <set>
#include <thread>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
// Declarations only: these bodies must resolve to actual CPU1075 c78 DSO.
namespace kimi_split_v128 {
infinilm::runtime::Tensor allocate(infinilm::runtime::Runtime&);
infinilm::runtime::Tensor forward(infinilm::runtime::Runtime&,const infinilm::runtime::Tensor&,const infinilm::runtime::Tensor&,const infinilm::runtime::Tensor&,const infinilm::runtime::Tensor&,const infinilm::global_state::AttentionMetadata&,float);
}
extern "C" int kimi_split_store(void*,const void*,const void*,const void*,int,std::size_t,const std::int64_t*,const std::int64_t*,void*);
namespace rt=infinilm::runtime;namespace att=infinilm::layers::attention;
namespace {
using Clock=std::chrono::steady_clock;thread_local std::string error;
struct Context{
 std::shared_ptr<rt::Runtime>r;std::thread::id owner;long tid=0;bool poison=false;int mode=-1;
 std::vector<rt::Tensor>cache;std::vector<int32_t>table;
 std::unique_ptr<att::AttentionLayer>layer;std::unique_ptr<att::backends::FlashAttentionImpl>flash;
 uint64_t queries=0,seed_chunks=0,resets=0;size_t valid=0;float scale=1.f/std::sqrt(192.f);
};
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){if(!c||c->poison||c->owner!=std::this_thread::get_id())throw std::invalid_argument("splitprobe null/poison/wrong owner");c->r->assertOwnerThread();int device=-1;check(cudaGetDevice(&device));if(device!=0)throw std::invalid_argument("splitprobe visible local device must0");}
void memory(uint64_t*out){size_t f=0,t=0;check(cudaMemGetInfo(&f,&t));out[0]=f;out[1]=t;}
void floor(){uint64_t m[2];memory(m);if(m[0]<uint64_t(4)*1024*1024*1024)throw std::runtime_error("original4GiB floor");}
void table_check(const int32_t*p,size_t n){if(!p||n!=128)throw std::invalid_argument("128 logical page IDs");std::set<int32_t>s(p,p+n);if(s.size()!=128||*s.begin()!=1||*s.rbegin()!=128)throw std::invalid_argument("permutation active1..128; reserved0/129..131");}
void clear(Context*c){drain(c);c->cache.clear();c->valid=0;c->mode=-1;drain(c);}
struct Guard{rt::Tensor backing;std::vector<std::byte>expected;rt::DataType dtype;};
struct Work{Context*c;std::vector<Guard>guards;std::vector<rt::Tensor>refs;std::vector<std::vector<int32_t>>i32;std::vector<std::vector<std::byte>>hostbytes;std::vector<int64_t>slots;~Work(){drain(c);} };
rt::Tensor gpu(Context*c,const void*p,rt::Shape shape,rt::DataType dt,Work&w){
 size_t width=dt==rt::DataType::kBFloat16?2:dt==rt::DataType::kInt32?4:dt==rt::DataType::kInt64?8:0;if(!width)throw std::invalid_argument("host staging dtype");size_t elements=1;for(auto n:shape)elements*=n;
 w.hostbytes.emplace_back((elements+32)*width,std::byte{0xa5});std::memcpy(w.hostbytes.back().data()+16*width,p,elements*width);
 auto host=rt::Tensor::from_blob(w.hostbytes.back().data(),{elements+32},dt,{rt::Device::Type::kCpu,0});auto backing=rt::Tensor::empty(*c->r,{elements+32},dt);w.refs.push_back(host);w.refs.push_back(backing);backing->copy_from(*c->r,host);w.guards.push_back({backing,w.hostbytes.back(),dt});return backing->narrow({{0,16,elements}})->view(shape);
}
void audit_inputs(Context*c,Work&w){
 for(auto&g:w.guards){std::vector<std::byte>actual(g.expected.size());auto host=rt::Tensor::from_blob(actual.data(),g.backing->shape(),g.dtype,{rt::Device::Type::kCpu,0});host->copy_from(*c->r,g.backing);c->r->syncStream();if(actual!=g.expected)throw std::runtime_error("actual GPU input/metadata bits or guards changed");}
}
infinilm::global_state::AttentionMetadata metadata(Context*c,size_t past,size_t count,Work&w){
 w.i32={{int32_t(past)},{int32_t(past+count)},{0,int32_t(count)},{0,int32_t(past+count)}};w.slots.resize(count);
 for(size_t i=0;i<count;++i){size_t pos=past+i;w.slots[i]=int64_t(c->table[pos/256])*256+pos%256;}
 auto hp=gpu(c,w.i32[0].data(),{1},rt::DataType::kInt32,w),ht=gpu(c,w.i32[1].data(),{1},rt::DataType::kInt32,w),off=gpu(c,w.i32[2].data(),{2},rt::DataType::kInt32,w),cu=gpu(c,w.i32[3].data(),{2},rt::DataType::kInt32,w),tab=gpu(c,c->table.data(),{1,128},rt::DataType::kInt32,w),slot=gpu(c,w.slots.data(),{count},rt::DataType::kInt64,w);
 return {hp,ht,off,cu,tab,slot,count,past+count,past+count};
}
rt::Tensor value192(Context*c,const rt::Tensor&v,size_t count,Work&w){
 auto pad=rt::Tensor::empty(*c->r,{count,3,192},rt::DataType::kBFloat16);w.refs.push_back(pad);c->r->setDeviceMemoryAsync(pad->data(),0,pad->nbytes(),c->r->stream());pad->narrow({{2,0,128}})->copy_from(*c->r,v);return pad;
}
void seed_piece(Context*c,size_t start,size_t count,const uint16_t*k,const uint16_t*v){
 Work w{c};auto md=metadata(c,start,count,w);auto key=gpu(c,k,{count,3,192},rt::DataType::kBFloat16,w),val=gpu(c,v,{count,3,128},rt::DataType::kBFloat16,w);
 if(c->mode==0){auto pad=value192(c,val,count,w);c->flash->do_kv_cache_update(*c->layer,key,pad,c->cache[0],*md.slot_mapping,*c->r);}
 else{const int64_t ks[]={int64_t(key->stride(0)),int64_t(key->stride(1))},vs[]={int64_t(val->stride(0)),int64_t(val->stride(1))};if(kimi_split_store(c->cache[0]->data(),key->data(),val->data(),(*md.slot_mapping)->data(),64,count,ks,vs,c->r->stream()))throw std::runtime_error("actual c78 prefix cache writer");}
 c->r->syncStream();++c->seed_chunks;
}
}
extern "C" const char*kimi_split_probe_error(){return error.c_str();}
extern "C" const char*kimi_split_probe_symbol_path(int which){
 static thread_local std::string path;Dl_info info{};void*p=which==0?reinterpret_cast<void*>(&kimi_split_v128::allocate):which==1?reinterpret_cast<void*>(&kimi_split_v128::forward):reinterpret_cast<void*>(&kimi_split_store);
 if(which<0||which>2||!dladdr(p,&info)||!info.dli_fname)return nullptr;path=info.dli_fname;return path.c_str();
}
extern "C" void*kimi_split_probe_create(int device){
 std::unique_ptr<Context>c;try{if(device!=0)throw std::invalid_argument("one visible local device0");check(cudaSetDevice(device));c=std::make_unique<Context>();c->owner=std::this_thread::get_id();c->tid=syscall(SYS_gettid);c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
 c->layer=std::make_unique<att::AttentionLayer>(3,192,c->scale,3,0,rt::Tensor{},rt::Tensor{},infinilm::backends::AttentionBackend::FLASH_ATTN,infinilm::quantization::KVQuantAlgo::NONE);
 c->flash=std::make_unique<att::backends::FlashAttentionImpl>(3,192,c->scale,3,0);floor();return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}
}
extern "C" int kimi_split_probe_begin(void*raw,int mode,const int32_t*table,size_t n){
 auto*c=static_cast<Context*>(raw);try{owner(c);if(mode!=0&&mode!=1)throw std::invalid_argument("OFF0/ON1");table_check(table,n);clear(c);c->r->trimMemory();c->table.assign(table,table+n);c->mode=mode;
 for(size_t i=0;i<24;++i){
  rt::Tensor t;if(mode)t=kimi_split_v128::allocate(*c->r);
  else{t=rt::Tensor::empty_parameter(*c->r,{2,132,256,3,192},rt::DataType::kBFloat16);c->r->setDeviceMemoryAsync(t->data(),0,t->nbytes(),c->r->stream());}
  c->cache.push_back(t);
 }c->r->syncStream();floor();return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_split_probe_reset(void*raw,const int32_t*table,size_t n){
 auto*c=static_cast<Context*>(raw);try{owner(c);table_check(table,n);if(c->cache.size()!=24)throw std::invalid_argument("reset before begin");drain(c);for(auto&t:c->cache)c->r->setDeviceMemoryAsync(t->data(),0,t->nbytes(),c->r->stream());c->table.assign(table,table+n);c->r->syncStream();c->valid=0;++c->resets;floor();return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_split_probe_seed(void*raw,size_t past,const void*k,const void*v){
 auto*c=static_cast<Context*>(raw);try{owner(c);if(!k||!v||past>32767||c->cache.size()!=24||c->valid)throw std::invalid_argument("seed exact clean prefix");for(size_t start=0;start<past;start+=512){size_t count=std::min(size_t(512),past-start);seed_piece(c,start,count,static_cast<const uint16_t*>(k)+start*3*192,static_cast<const uint16_t*>(v)+start*3*128);}c->valid=past;floor();return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_split_probe_query(void*raw,size_t past,size_t count,const void*q,const void*k,const void*v,void*out,size_t words,double*seconds,size_t ns){
 auto*c=static_cast<Context*>(raw);auto start=Clock::now();try{owner(c);if(!q||!k||!v||!out||!seconds||ns!=3||!count||count>512||past+count>32768||past!=c->valid||words!=count*3*128)throw std::invalid_argument("query prefix/shape/output contract");
 Work w{c};auto md=metadata(c,past,count,w);auto query=gpu(c,q,{count,3,192},rt::DataType::kBFloat16,w),key=gpu(c,k,{count,3,192},rt::DataType::kBFloat16,w),val=gpu(c,v,{count,3,128},rt::DataType::kBFloat16,w);c->r->syncStream();auto dispatch=Clock::now();rt::Tensor y;
 if(c->mode==0){auto pad=value192(c,val,count,w);auto padded=c->flash->forward(*c->layer,query,key,pad,c->cache[0],md,*c->r);w.refs.push_back(padded);y=padded->view({count,3,192})->narrow({{2,0,128}})->contiguous(*c->r);}
 else y=kimi_split_v128::forward(*c->r,query,key,val,c->cache[0],md,c->scale);
 w.refs.push_back(y);c->r->syncStream();seconds[0]=std::chrono::duration<double>(Clock::now()-dispatch).count();auto copy=Clock::now();auto host=rt::Tensor::from_blob(out,{count,3,128},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});w.refs.push_back(host);host->copy_from(*c->r,y);c->r->syncStream();audit_inputs(c,w);seconds[1]=std::chrono::duration<double>(Clock::now()-copy).count();seconds[2]=std::chrono::duration<double>(Clock::now()-start).count();c->valid=past+count;++c->queries;floor();return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_split_probe_audit_pages(void*raw,uint64_t*out,size_t n){
 auto*c=static_cast<Context*>(raw);try{owner(c);if(!out||n!=2||c->cache.size()!=24)throw std::invalid_argument("guardpage audit schema");out[0]=0;out[1]=0;
 for(size_t page:{size_t(0),size_t(129),size_t(130),size_t(131)}){
  auto view=c->mode?c->cache[0]->narrow({{0,page,1}}):c->cache[0]->narrow({{1,page,1}});auto contiguous=view->contiguous(*c->r);std::vector<uint16_t>data(contiguous->numel());auto host=rt::Tensor::from_blob(data.data(),contiguous->shape(),rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});host->copy_from(*c->r,contiguous);c->r->syncStream();if(std::any_of(data.begin(),data.end(),[](uint16_t v){return v!=0;}))throw std::runtime_error("reserved cache page changed");++out[0];out[1]+=data.size()*2;
 }return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_split_probe_observe(void*raw,uint64_t*out,size_t n){
 auto*c=static_cast<Context*>(raw);try{owner(c);if(!out||n!=16+24*8)throw std::invalid_argument("census schema");std::memset(out,0,n*sizeof(uint64_t));memory(out);out[2]=c->tid;out[3]=syscall(SYS_gettid);out[4]=c->mode;out[5]=c->cache.size();out[6]=c->queries;out[7]=c->seed_chunks;out[8]=c->resets;out[9]=c->valid;out[10]=c->r->device().index();
 for(size_t i=0;i<c->cache.size();++i){auto&t=c->cache[i];size_t off=16+i*8;out[off]=reinterpret_cast<uintptr_t>(t->data());out[off+1]=t->nbytes();out[off+2]=t->ndim();for(size_t j=0;j<t->ndim();++j)out[off+3+j]=t->size(j);}return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int kimi_split_probe_destroy(void*raw,uint64_t*out,size_t n){
 auto*c=static_cast<Context*>(raw);if(!c||c->owner!=std::this_thread::get_id()||!out||n!=5){error="wrong owner/destroy schema";return 1;}
 drain(c);memory(out);c->cache.clear();c->flash.reset();c->layer.reset();std::weak_ptr<rt::Runtime>weak=c->r;c->r->trimMemory();c->r.reset();out[2]=weak.expired();delete c;memory(out+3);return out[2]?0:1;
}
