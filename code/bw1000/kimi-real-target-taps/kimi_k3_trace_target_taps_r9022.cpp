#include "kimi_k3_trace.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include "kimi_target_taps_r9022.hpp"
namespace infinilm::models::kimi_k3 {
namespace {
size_t option(const char*name,size_t value){auto s=std::getenv(name);return s?std::stoull(s):value;}
bool selected_boundary_r8506(size_t layer,const char*point){
 auto selected=[](const char*env,const std::string&key){
  const char*p=std::getenv(env);if(!p||!*p)return true;
  std::string value(p);size_t begin=0;
  while(begin<=value.size()){auto end=value.find(',',begin);if(value.substr(begin,end==std::string::npos?end:end-begin)==key)return true;if(end==std::string::npos)break;begin=end+1;}
  return false;
 };
 return selected("INFINILM_KIMI_TRACE_BOUNDARY_LAYERS_R8506",std::to_string(layer))&&selected("INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506",point);
}

}
bool trace_detail_layer(size_t layer){
 if(kimi_target_taps_r9022::enabled())return true;
 // Only the explicitly traced epoch changes detail selection. Preserve the
 // parent's disabled-trace path and its existing range behavior otherwise.
 const char*directory=std::getenv("INFINILM_KIMI_TRACE_DIR");
 const char*cursor=directory&&*directory?std::getenv("INFINILM_KIMI_TRACE_LAYERS"):nullptr;
 if(cursor&&*cursor){
  while(*cursor){char*end=nullptr;auto value=std::strtoull(cursor,&end,10);
   if(end==cursor||(*end&&*end!=','))throw std::invalid_argument("Invalid Kimi trace layer list");
   if(value==layer)return true;cursor=*end?end+1:end;
  }return false;
 }
 return layer>=option("INFINILM_KIMI_TRACE_FIRST",45)&&layer<=option("INFINILM_KIMI_TRACE_LAST",52);
}
void trace_hidden(runtime::Runtime&r,const runtime::Tensor&x,const global_state::ForwardContext&fc,size_t layer,const char*point){
 kimi_target_taps_r9022::collect(r,x,fc,layer,point);
 const char*directory=std::getenv("INFINILM_KIMI_TRACE_DIR");if(!directory||!*directory)return;
 if(!selected_boundary_r8506(layer,point))return; // Gate BEFORE GPU copy, no math or state mutation.
 r.assertOwnerThread();if(r.isGraphRecording()||!x||x->dtype()!=runtime::DataType::kBFloat16||x->ndim()<2||!fc.attn_metadata.input_offsets)throw std::logic_error("Kimi trace requires eager BF16 hidden with request offsets");
 const auto original_shape=x->shape();const auto original_strides=x->strides();
 size_t hidden=x->size(x->ndim()-1),tokens=x->numel()/hidden;
 auto offsets=fc.attn_metadata.input_offsets.value()->contiguous(r);if(offsets->dtype()!=runtime::DataType::kInt32)throw std::invalid_argument("Kimi trace offset type differs");std::vector<int32_t>host_offsets(offsets->numel());r.memcpyD2H(host_offsets.data(),offsets->data(),host_offsets.size()*4);
 if(host_offsets.size()<2||host_offsets.front()!=0||host_offsets.back()!=int64_t(tokens))throw std::invalid_argument("Kimi trace packed token count differs");
 const auto original_offsets=host_offsets;auto selected_view=x;size_t selected_request=0;bool target_only=false;
 const char*target=std::getenv("INFINILM_KIMI_TRACE_PACKED_REQUEST_R8508");
 if(target&&*target){
  target_only=true;selected_request=std::stoull(target);
  if(selected_request+1>=host_offsets.size()||!((x->ndim()==2)||(x->ndim()==3&&x->size(0)==1)))throw std::invalid_argument("Kimi target trace packed row/shape invalid");
  int32_t begin=host_offsets[selected_request],end=host_offsets[selected_request+1];
  if(begin<0||begin>=end||size_t(end)>tokens)throw std::invalid_argument("Kimi target trace offsets invalid");
  selected_view=x->narrow({{x->ndim()-2,size_t(begin),size_t(end-begin)}});
  tokens=size_t(end-begin);host_offsets={0,end-begin};
 }
 auto contiguous=selected_view->contiguous(r);std::vector<uint16_t>raw(selected_view->numel());r.memcpyD2H(raw.data(),contiguous->data(),raw.size()*2);
 auto stage_env=std::getenv("SLURM_PROCID");int stage=stage_env?std::atoi(stage_env):0,rank=r.device().index();
 std::string prefix=std::string(directory)+"/stage"+std::to_string(stage)+"-rank"+std::to_string(rank)+"-layer"+std::to_string(layer)+"-"+point;
 std::vector<uint16_t>last;std::vector<nlohmann::json>requests;
 for(size_t request=0;request+1<host_offsets.size();++request){int32_t begin=host_offsets[request],end=host_offsets[request+1];if(begin<0||begin>=end||size_t(end)>tokens)throw std::invalid_argument("Kimi trace offsets invalid");uint64_t hash=14695981039346656037ULL;size_t nonfinite=0;double sum=0,squares=0;
  for(size_t i=size_t(begin)*hidden;i<size_t(end)*hidden;++i){uint16_t b=raw[i];hash=(hash^uint8_t(b))*1099511628211ULL;hash=(hash^uint8_t(b>>8))*1099511628211ULL;uint32_t bits=uint32_t(b)<<16;float value;std::memcpy(&value,&bits,4);if(!std::isfinite(value))++nonfinite;else{sum+=value;squares+=double(value)*value;}}
  requests.push_back({{"request",target_only?selected_request:request},{"tokens",end-begin},{"fnv1a64",std::to_string(hash)},{"sum",sum},{"sum_squares",squares},{"nonfinite",nonfinite}});last.insert(last.end(),raw.begin()+size_t(end-1)*hidden,raw.begin()+size_t(end)*hidden);
 }
 auto write=[&](const std::string&path,const uint16_t*p,size_t n){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(p),n*2);if(!f)throw std::runtime_error("Kimi trace write failed: "+path);};
 const char*hash_env=std::getenv("INFINILM_KIMI_TRACE_HASH_ONLY_R8506");
 if(!(hash_env&&std::string(hash_env)=="1")){
 write(prefix+"-last-bf16.bin",last.data(),last.size());
 // Full snapshots of the first two layers help inspect an earliest mismatch.
 if((rank==0||rank==6)&&((std::getenv("INFINILM_KIMI_TRACE_MLA_FULL")&&std::string(point).rfind("mla_",0)==0)||(layer>=option("INFINILM_KIMI_TRACE_FULL_FIRST",45)&&layer<=option("INFINILM_KIMI_TRACE_FULL_LAST",48)))&&std::string(point).find("residual_slot_")!=0)write(prefix+"-full-bf16.bin",raw.data(),raw.size());
 }
 nlohmann::json record={{"stage",stage},{"rank",rank},{"layer",layer},{"point",point},{"tokens",tokens},{"hidden",hidden},{"requests",requests},{"original_shape",original_shape},{"original_strides",original_strides},{"original_offsets",original_offsets},{"selected_shape",selected_view->shape()},{"selected_strides",selected_view->strides()},{"packed_target_only",target_only}};
 std::ofstream f(std::string(directory)+"/stage"+std::to_string(stage)+"-rank"+std::to_string(rank)+".jsonl",std::ios::app);f<<record.dump()<<'\n';if(!f)throw std::runtime_error("Kimi trace journal write failed");
}
}
