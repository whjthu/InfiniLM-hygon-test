#include "kimi_k3_kda_trace.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <set>
#include <stdexcept>
namespace infinilm::models::kimi_k3 {
namespace rt=infinilm::runtime;
struct KdaTraceImpl{
 struct Snapshot{rt::Tensor value;std::string point;nlohmann::json source_layout;};
 rt::Runtime&r;const global_state::ForwardContext&fc;std::string directory;size_t layer,call;int stage,rank;bool finished=false;
 std::vector<Snapshot>snapshots;
 KdaTraceImpl(rt::Runtime&owner,std::string d,size_t l,size_t c,const global_state::ForwardContext&context):r(owner),fc(context),directory(std::move(d)),layer(l),call(c),stage(std::getenv("SLURM_PROCID")?std::atoi(std::getenv("SLURM_PROCID")):0),rank(owner.device().index()){}
 void capture_direct(const rt::Tensor&x,const char*point){
  r.assertOwnerThread();if(!x||x->device()!=r.device()||r.isGraphRecording()||snapshots.size()>=64)throw std::logic_error("KDA diagnostic snapshot owner/state differs");
  if(x->dtype()!=rt::DataType::kBFloat16&&x->dtype()!=rt::DataType::kFloat32&&x->dtype()!=rt::DataType::kInt32&&x->dtype()!=rt::DataType::kInt64)throw std::invalid_argument("KDA diagnostic supports BF16, F32, I32 and I64 without conversion");
  auto copy=rt::Tensor::empty(r,x->shape(),x->dtype());copy->copy_from(r,x);snapshots.push_back({copy,point,{{"shape",x->shape()},{"strides",x->strides()}}});
  const char*sync=std::getenv("INFINILM_KIMI_KDA_TRACE_SYNC_EACH");if(sync&&std::string(sync)=="1")r.syncStream();
 }
 void capture(const rt::Tensor&x,const char*point){
  const std::string tag(point);
  const bool pool=tag=="conv_before"||tag=="conv_after"||tag=="ssm_before"||tag=="ssm_after";
  if(!pool){capture_direct(x,point);return;}
  r.assertOwnerThread();
  auto indices=[&](const auto&field){
   if(!field||!*field)throw std::logic_error("KDA trace missing state rowmetadata");
   auto tensor=(*field)->contiguous(r);if(tensor->numel()<1||tensor->numel()>4)throw std::logic_error("KDA trace active request bound");
   std::vector<int64_t> values(tensor->numel());
   if(tensor->dtype()==rt::DataType::kInt32){std::vector<int32_t>v(values.size());r.memcpyD2H(v.data(),tensor->data(),v.size()*4);for(size_t i=0;i<v.size();++i)values[i]=v[i];}
   else if(tensor->dtype()==rt::DataType::kInt64)r.memcpyD2H(values.data(),tensor->data(),values.size()*8);
   else throw std::logic_error("KDA trace row dtype must be I32/I64");return values;
  };
  auto reads=indices(fc.mamba_metadata.init_state_indices),writes=indices(fc.mamba_metadata.final_state_indices);
  if(reads.size()!=writes.size())throw std::logic_error("KDA trace state row count differs");
  const char*mode=std::getenv("INFINILM_KIMI_TRACE_STATE_ROWS_MODE_R8516");bool all=mode&&std::string(mode)=="all";
  const char*target=std::getenv("INFINILM_KIMI_TRACE_PACKED_REQUEST_R8508");size_t packed=target&&*target?std::stoull(target):0;
  if(packed>=reads.size())throw std::logic_error("KDA trace packedtarget outofrange");
  std::set<size_t> active;
  for(size_t i=0;i<reads.size();++i){if(!all&&i!=packed)continue;for(auto row:{reads[i],writes[i]}){if(row<0||size_t(row)>=x->size(0))throw std::logic_error("KDA trace actual state row outofrange");active.insert(size_t(row));}}
  if(active.size()>8)throw std::logic_error("KDA trace stateunion bound");
  for(size_t row:active){auto name=tag+"_row_"+std::to_string(row);capture_direct(x->narrow({{0,row,1}}),name.c_str());auto&layout=snapshots.back().source_layout;layout["pool_shape"]=x->shape();layout["pool_strides"]=x->strides();layout["pool_row"]=row;layout["pool_row_offset_elements"]=row*x->strides()[0];layout["selection"]=all?"all_companion_read_write_union":"target_read_write_union";}
 }
 void finish(){
  r.syncStream();
  std::string stem="stage"+std::to_string(stage)+"-rank"+std::to_string(rank)+"-layer"+std::to_string(layer)+"-kda-call"+std::to_string(call);
  std::ofstream journal(directory+"/"+stem+".jsonl",std::ios::app);
  if(!journal)throw std::runtime_error("Cannot open KDA snapshot journal");
  for(auto&snapshot:snapshots){
   auto&x=snapshot.value;bool bf16=x->dtype()==rt::DataType::kBFloat16,f32=x->dtype()==rt::DataType::kFloat32;bool i64=x->dtype()==rt::DataType::kInt64;size_t itemsize=bf16?2:(i64?8:4);
   std::vector<unsigned char>raw(x->numel()*itemsize);r.memcpyD2H(raw.data(),x->data(),raw.size());uint64_t hash=14695981039346656037ULL;for(auto b:raw)hash=(hash^b)*1099511628211ULL;
   size_t nonfinite=0;for(size_t i=0;i<x->numel()&&(bf16||f32);++i){if(bf16){uint16_t bits;std::memcpy(&bits,raw.data()+i*2,2);if((bits&0x7f80)==0x7f80)++nonfinite;}else{uint32_t bits;std::memcpy(&bits,raw.data()+i*4,4);if((bits&0x7f800000)==0x7f800000)++nonfinite;}}
   auto name=stem+"-"+snapshot.point+"-raw.bin";std::ofstream binary(directory+"/"+name,std::ios::binary);binary.write(reinterpret_cast<const char*>(raw.data()),raw.size());if(!binary)throw std::runtime_error("Cannot write KDA snapshot");
   nlohmann::json row={{"stage",stage},{"rank",rank},{"layer",layer},{"call",call},{"point",snapshot.point},{"shape",x->shape()},{"dtype",bf16?"bf16":f32?"f32":i64?"i64":"i32"},{"bytes",raw.size()},{"fnv1a64",std::to_string(hash)},{"nonfinite",nonfinite},{"file",name},{"source_layout",snapshot.source_layout}};journal<<row.dump()<<'\n';
  }
  if(!journal)throw std::runtime_error("KDA snapshot journal write failed");finished=true;
 }
};
namespace {thread_local KdaTraceImpl*active=nullptr;thread_local std::map<std::string,size_t>calls;}
KdaTraceBatch::KdaTraceBatch(rt::Runtime&r,const global_state::ForwardContext&fc,size_t layer):previous_(active){
 const char*d=std::getenv("INFINILM_KIMI_TRACE_DIR");const char*s=std::getenv("INFINILM_KIMI_KDA_TRACE_LAYER");size_t selected=s?std::stoull(s):46;
 if(d&&*d&&layer==selected){if(r.isGraphRecording())throw std::logic_error("KDA raw trace requires eager Runtime");std::string key=std::string(d)+"/"+std::to_string(layer);impl_=std::make_unique<KdaTraceImpl>(r,d,layer,calls[key]++,fc);active=impl_.get();}
}
KdaTraceBatch::~KdaTraceBatch(){if(impl_&&!impl_->finished)try{impl_->r.syncStream();}catch(...){std::terminate();}active=previous_;}
void KdaTraceBatch::capture(const rt::Tensor&x,const char*p){if(impl_)impl_->capture(x,p);}
void KdaTraceBatch::finish(){if(impl_)impl_->finish();}
void trace_kda_projection(rt::Runtime&r,const rt::Tensor&x,const char*p){if(active){if(&active->r!=&r)throw std::logic_error("KDA projection trace crossed Runtime owners");active->capture(x,p);}}
}
