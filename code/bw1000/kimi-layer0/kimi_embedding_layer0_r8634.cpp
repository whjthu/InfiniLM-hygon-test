// Thin real embedding + actual layer0 forward, CPU-build candidate only.
// All constructors/forward/operators resolve to CPU183 Model and its actual Common.
// No diagnostic class, random residual, inverse norm, hidden hook or new math.
#include "models/kimi_k3/kimi_k3_decoder_layer.hpp"
#include "layers/embedding/embedding.hpp"
#include "global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <string>
#include <vector>
namespace rt=infinilm::runtime;
namespace kk=infinilm::models::kimi_k3;
namespace infinilm::models::kimi_k3 {
std::shared_ptr<infinilm::config::ModelConfig> create_kimi_k3_model_config(std::shared_ptr<infinilm::config::ModelConfig>);
}
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
// Actual TextModel object2985e6b allocates0x150 for shared block, decoder at+0x10.
static_assert(sizeof(kk::KimiK3DecoderLayer)==0x140,"Actual selected decoder allocation ABI differs");
// Same actual TextModel shared Embedding block0xf0, object at+0x10.
static_assert(sizeof(infinilm::layers::Embedding)==0xe0,"Actual selected Embedding allocation ABI differs");
namespace {
thread_local std::string error;
struct Context {
 std::shared_ptr<rt::Runtime> runtime;
 std::unique_ptr<infinilm::weights::ModelWeights> embed_owner,layer_owner;
 infinilm::layers::Embedding*embed=nullptr;kk::KimiK3DecoderLayer*layer=nullptr;
 infinicclComm_t comm=nullptr;bool ready=false;std::set<std::string>loaded;
 rt::Tensor conv,state;std::thread::id owner=std::this_thread::get_id();
};
void owner(Context*c){if(!c||c->owner!=std::this_thread::get_id())throw std::logic_error("8629 owner thread mismatch");c->runtime->assertOwnerThread();}
void drain(Context*c)noexcept{if(c&&c->runtime)try{c->runtime->syncStream();}catch(...){std::terminate();}}
void cleanup(Context*c)noexcept{
 if(!c)return;drain(c);c->layer=nullptr;c->embed=nullptr;c->layer_owner.reset();c->embed_owner.reset();c->conv={};c->state={};drain(c);
 if(c->comm){if(infinicclCommDestroy(c->comm)!=infinicclSuccess)std::terminate();c->comm=nullptr;}c->runtime.reset();
}
void postload(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(const auto&child:m.children())postload(r,*child.second);m.process_weights_after_loading(r,false);}
rt::Tensor cpufrom(const void*p,rt::Shape s,rt::DataType d){return rt::Tensor::from_blob(const_cast<void*>(p),s,d,{rt::Device::Type::kCpu,0});}
rt::Tensor upload(Context*c,const void*p,rt::Shape s,rt::DataType d){auto out=rt::Tensor::empty(*c->runtime,s,d);out->copy_from(*c->runtime,cpufrom(p,s,d));return out;}
struct Flags {Flags(){modern_kimi_set_grouped_prefill(1);modern_kimi_set_tiled_w13_r1861(1);}~Flags(){modern_kimi_set_tiled_w13_r1861(0);modern_kimi_set_grouped_prefill(0);}};
}
extern "C" const char* kimi_layer0_error_r8634(){return error.c_str();}
extern "C" size_t kimi_layer0_id_size_r8634(){return sizeof(infinicclUniqueId);}
extern "C" int kimi_layer0_id_r8634(void*out){try{if(!out)throw std::invalid_argument("nullID");infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)throw std::runtime_error("getID");std::memcpy(out,&id,sizeof(id));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" void*kimi_layer0_create_r8634(const char*raw,int rank,const void*idbytes){
 std::unique_ptr<Context>c;
 try{
  int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||rank!=device||rank<0||rank>=8||!raw||!idbytes)throw std::invalid_argument("TP8 exactrank/device");
  c=std::make_unique<Context>();c->runtime=rt::Runtime::create({rt::Device::Type::kHygon,device});c->runtime->enableOwnerDrainFailStop();
  infini::rt::DeviceGuard guard{c->runtime->device().type(),device,true};infinicclUniqueId id{};std::memcpy(&id,idbytes,sizeof(id));
  if(infinicclCommInitRank(&c->comm,8,id,rank)!=infinicclSuccess)throw std::runtime_error("TP8comm");
  auto original=std::make_shared<infinilm::config::ModelConfig>(nlohmann::json::parse(raw));auto cfg=kk::create_kimi_k3_model_config(original);
  if(cfg->get<size_t>("hidden_size")!=7168||cfg->get<size_t>("first_k_dense_replace")!=1||cfg->get<size_t>("attn_res_block_size")!=12||cfg->get_dtype()!=rt::DataType::kBFloat16)throw std::invalid_argument("actual layer0 config");
  infinilm::engine::distributed::RankInfo ri(c->runtime->device());ri.tp_size=8;ri.tp_rank=rank;ri.world_size=8;ri.world_rank=rank;ri.comm=c->comm;
  // Delete through actual virtual ModelWeights, not a locally emitted decoder vtable.
  c->embed=new infinilm::layers::Embedding(*c->runtime,163840,7168,std::nullopt,rt::DataType::kBFloat16);c->embed_owner.reset(c->embed);
  c->layer=new kk::KimiK3DecoderLayer(*c->runtime,cfg,0,ri,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});c->layer_owner.reset(c->layer);
  c->conv=rt::Tensor::empty(*c->runtime,{33,4608,3},rt::DataType::kBFloat16);c->state=rt::Tensor::empty(*c->runtime,{33,12,128,128},rt::DataType::kBFloat16);drain(c.get());return c.release();
 }catch(const std::exception&e){error=e.what();cleanup(c.get());return nullptr;}
}
extern "C" int kimi_layer0_load_r8634(void*raw,const char*name,const void*p,size_t ndim,const size_t*shape,int dtype){auto*c=static_cast<Context*>(raw);try{
 owner(c);if(c->ready||!name||!p||!shape||!ndim||ndim>4||dtype<0||dtype>4)throw std::invalid_argument("load args");std::string key(name);if(!c->loaded.insert(key).second)throw std::invalid_argument("duplicate weight");
 const auto type=dtype==0?rt::DataType::kFloat32:dtype==1?rt::DataType::kBFloat16:dtype==2?rt::DataType::kUInt8:dtype==3?rt::DataType::kInt64:rt::DataType::kInt8;
 auto source=cpufrom(p,rt::Shape(shape,shape+ndim),type);if(key=="embed_tokens.weight")c->embed->load_parameter(*c->runtime,"weight",source);else if(key.rfind("layers.0.",0)==0)c->layer->load_parameter(*c->runtime,key.substr(9),source);else throw std::invalid_argument("not layer0/embedding");
 drain(c);return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int kimi_layer0_finish_r8634(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);if(c->ready)throw std::logic_error("alreadyready");if(!c->loaded.count("embed_tokens.weight"))throw std::runtime_error("missingembedding");for(const auto&name:c->layer->state_dict_keys())if(!c->loaded.count("layers.0."+name))throw std::runtime_error("missinglayer0:"+name);postload(*c->runtime,*c->embed);postload(*c->runtime,*c->layer);drain(c);c->ready=true;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int kimi_layer0_prefill_r8634(void*raw,size_t count,const int32_t*rows,const int64_t*ids,void*host_output){auto*c=static_cast<Context*>(raw);try{
 owner(c);if(!c->ready||(count!=1&&count!=4)||!rows||!ids||!host_output)throw std::invalid_argument("actual O512/N2048 only");const size_t m=512*count;
 std::vector<int32_t>offsets(count+1),pasts(count,0),totals(count,512),reads(count),cu(count+1),table(count*4);std::vector<int64_t>positions(m),slots(m);
 for(size_t i=0;i<count;++i){if(rows[i]!=int32_t(i+1))throw std::invalid_argument("immutable rows1..count");reads[i]=rows[i];offsets[i]=512*i;cu[i]=512*i;int32_t page=1+(rows[i]-1)*4;for(size_t j=0;j<4;++j)table[i*4+j]=page+j;for(size_t j=0;j<512;++j){if(ids[i*512+j]<0||ids[i*512+j]>=163840)throw std::invalid_argument("actual token IDs");positions[i*512+j]=j;slots[i*512+j]=(page+j/256)*256+j%256;}}
 offsets[count]=m;cu[count]=m;auto&r=*c->runtime;
 // Exact original reset state, retained33-row pools; reset outside any timing.
 if(cudaMemset(c->conv->data(),0,c->conv->nbytes())!=cudaSuccess||cudaMemset(c->state->data(),0,c->state->nbytes())!=cudaSuccess)throw std::runtime_error("resetstate");
 if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("resetstate explicit cross-stream completion");
 auto input=upload(c,ids,{1,m},rt::DataType::kInt64);infinilm::global_state::ForwardContext fc;fc.runtime=&r;fc.conv_state_vec={c->conv};fc.ssm_state_vec={c->state};
 fc.attn_metadata.past_sequence_lengths=upload(c,pasts.data(),{count},rt::DataType::kInt32);fc.attn_metadata.total_sequence_lengths=upload(c,totals.data(),{count},rt::DataType::kInt32);fc.attn_metadata.input_offsets=upload(c,offsets.data(),{count+1},rt::DataType::kInt32);fc.attn_metadata.cu_seqlens=upload(c,cu.data(),{count+1},rt::DataType::kInt32);fc.attn_metadata.block_tables=upload(c,table.data(),{count,4},rt::DataType::kInt32);fc.attn_metadata.slot_mapping=upload(c,slots.data(),{m},rt::DataType::kInt64);fc.attn_metadata.max_query_length=512;fc.attn_metadata.max_sequence_length=512;fc.attn_metadata.logical_kv_upper_bound=512;
 fc.mamba_metadata.input_offsets=fc.attn_metadata.input_offsets;fc.mamba_metadata.init_state_indices=upload(c,reads.data(),{count},rt::DataType::kInt32);fc.mamba_metadata.final_state_indices=fc.mamba_metadata.init_state_indices;
 auto residual=rt::Tensor::empty(r,{m,3,7168},rt::DataType::kBFloat16);auto scratch=residual->narrow({{1,0,1}})->view({1,m,7168});auto embedded=c->embed->forward(r,input);scratch->copy_from(r,embedded);size_t residual_count=0;
 Flags flags;auto result=c->layer->forward(scratch,residual,residual_count,fc);if(residual_count!=1)throw std::runtime_error("actual layer0 residual count");auto flat=result->contiguous(r);r.syncStream();if(cudaMemcpy(host_output,flat->data(),m*7168*2,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("outputcopy");return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int kimi_layer0_destroy_r8634(void*raw){auto*c=static_cast<Context*>(raw);try{owner(c);cleanup(c);delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
