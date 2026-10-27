// Included once, by original kimi_k3_trace.cpp. Diagnostic only; default off.
#include <atomic>
#include <mutex>
#include <filesystem>
#include <limits>
namespace kimi_target_taps_r9022 {
using J=nlohmann::json;
std::atomic<bool> armed{false};std::mutex lock;
std::string label,root;J records=J::array();uint64_t epoch=0,sequence=0;
bool enabled(){return armed.load(std::memory_order_acquire);}
int node(){auto*p=std::getenv("SLURM_PROCID");return p?std::stoi(p):0;}
bool representative(infinilm::runtime::Runtime&r){return r.device().index()==0&&node()%2==0;}
J ints(infinilm::runtime::Runtime&r,const std::optional<infinilm::runtime::Tensor>&x){
 if(!x)return nullptr;auto t=(*x)->contiguous(r);if(t->dtype()!=infinilm::runtime::DataType::kInt32||t->numel()>16)throw std::logic_error("9022 int32 metadata bound");
 std::vector<int32_t>v(t->numel());r.memcpyD2H(v.data(),t->data(),v.size()*4);return v;
}
J state_row(infinilm::runtime::Runtime&r,const infinilm::runtime::Tensor&t,int32_t row){
 if(!t||row<0||size_t(row)>=t->size(0))throw std::logic_error("9022 recurrent row bound");
 auto x=t->narrow({{0,size_t(row),1}})->contiguous(r);if(x->nbytes()>8*1024*1024)throw std::logic_error("9022 bounded recurrent row");
 std::vector<uint8_t>v(x->nbytes());r.memcpyD2H(v.data(),x->data(),v.size());uint64_t h=14695981039346656037ULL;for(auto b:v)h=(h^b)*1099511628211ULL;
 return {{"row",row},{"shape",x->shape()},{"bytes",v.size()},{"fnv1a64",std::to_string(h)}};
}
void collect(infinilm::runtime::Runtime&r,const infinilm::runtime::Tensor&x,const infinilm::global_state::ForwardContext&fc,size_t layer,const char*point){
 if(!enabled()||!representative(r))return;
 const std::string p(point);const bool tap=(layer==19||layer==37||layer==66||layer==78||layer==90)&&p=="decoder";
 const bool boundary=p=="attn_norm"||p=="attn_output"||p=="exit"||p=="final_norm";
 if(!tap&&!boundary)return;
 r.assertOwnerThread();if(r.isGraphRecording()||!x||x->dtype()!=infinilm::runtime::DataType::kBFloat16||x->size(x->ndim()-1)!=7168)throw std::logic_error("9022 eager BF16 H7168 required");
 auto past=ints(r,fc.attn_metadata.past_sequence_lengths);auto total=ints(r,fc.attn_metadata.total_sequence_lengths);
 auto offsets=ints(r,fc.attn_metadata.input_offsets);
 if(past.size()!=1||total.size()!=1||offsets.size()!=2||offsets[0]!=0)throw std::logic_error("9022 B1 only");
 int64_t begin=past[0],end=total[0];size_t tokens=x->numel()/7168;
 if(end-begin!=int64_t(tokens)||offsets[1]!=tokens||tokens>512)throw std::logic_error("9022 position bounds");
 std::vector<size_t> selected;if(tap){for(size_t i=0;i<tokens;++i)selected.push_back(i);}else for(int64_t pos:{536,537})if(pos>=begin&&pos<end)selected.push_back(size_t(pos-begin));
 if(selected.empty())return;
 auto t=x->contiguous(r);std::vector<uint16_t>raw(tokens*7168);r.memcpyD2H(raw.data(),t->data(),raw.size()*2);
 // memcpyD2H follows the existing Runtime contract; no arithmetic/CCL change.
 std::lock_guard<std::mutex>guard(lock);if(!enabled())throw std::logic_error("9022 arm changed during forward");
 J positions=J::array();std::vector<uint16_t>owned;owned.reserve(selected.size()*7168);size_t nonfinite=0;uint64_t hash=14695981039346656037ULL;
 for(auto i:selected){positions.push_back(begin+int64_t(i));for(size_t j=0;j<7168;++j){auto b=raw[i*7168+j];owned.push_back(b);hash=(hash^uint8_t(b))*1099511628211ULL;hash=(hash^uint8_t(b>>8))*1099511628211ULL;if((b&0x7f80)==0x7f80)++nonfinite;}}
 auto path=root+"/e"+std::to_string(epoch)+"-n"+std::to_string(node())+"-"+std::to_string(sequence++)+".bf16";
 std::ofstream f(path,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(owned.data()),owned.size()*2);f.close();if(!f)throw std::runtime_error("9022 BF16 owned buffer write");
 auto read=ints(r,fc.mamba_metadata.init_state_indices),write=ints(r,fc.mamba_metadata.final_state_indices);J recurrent=nullptr;
 if((p=="attn_norm"||p=="attn_output")&&layer<fc.ssm_state_vec.size()&&fc.ssm_state_vec.at(layer)){
  if(read.size()!=1||write.size()!=1)throw std::logic_error("9022 recurrent B1 metadata");
  recurrent={{"ssm_read",state_row(r,fc.ssm_state_vec.at(layer),read[0])},{"ssm_write",state_row(r,fc.ssm_state_vec.at(layer),write[0])},{"conv_read",state_row(r,fc.conv_state_vec.at(layer),read[0])},{"conv_write",state_row(r,fc.conv_state_vec.at(layer),write[0])}};
 }
 records.push_back({{"layer",layer},{"point",p},{"node",node()},{"device",0},{"positions",positions},{"shape",{1,selected.size(),7168}},{"dtype","BF16"},{"remote_owned_buffer",path},{"bytes",owned.size()*2},{"fnv1a64",std::to_string(hash)},{"nonfinite",nonfinite},{"past",past},{"total",total},{"offsets",offsets},{"actual_init_state_indices",read},{"actual_final_state_indices",write},{"recurrent_rows_at_boundary",recurrent}});
 if(records.size()>1500)throw std::logic_error("9022 record bound");
}
}
extern "C" const char*modern_kimi_diag_capabilities_r9008(){return R"({"abi":"kimi-crossover-r9008-v1","positions":[536,537],"actual_final_hidden":true,"actual_layer_boundaries":true,"actual_recurrent_state_metadata":true,"actual_gpu_calls":true,"gpu_calls_meaning":"actual callback invocation count, not hardware profiler","source_only_unqualified":true,"state_checksum":true,"representative_only":"TP16PP2 node0/2 device0","target_tap_training_contract_verified":false})";}
extern "C" int modern_kimi_diag_arm_r9008(void*ctx,const char*arm,int enabled){using namespace kimi_target_taps_r9022;try{if(!ctx||!arm||(enabled!=0&&enabled!=1))return 1;std::lock_guard<std::mutex>guard(lock);armed.store(false,std::memory_order_release);if(enabled){auto*d=std::getenv("INFINILM_KIMI_TARGET_TAPS_DIR_R9022");if(!d||!*d||std::string(arm).size()>64)return 1;root=d;std::filesystem::create_directories(root);label=arm;++epoch;sequence=0;records=J::array();armed.store(true,std::memory_order_release);}return 0;}catch(...){return 1;}}
extern "C" const char*modern_kimi_diag_read_r9008(void*ctx){using namespace kimi_target_taps_r9022;static thread_local std::string result;std::lock_guard<std::mutex>guard(lock);bool finite=true;for(auto&r:records)finite=finite&&r.at("nonfinite")==0;result=J({{"epoch",epoch},{"arm",label},{"finite",finite},{"enabled",enabled()},{"actual_callback_invocations",sequence},{"records",records},{"raw_data_stays_remote",true},{"training_contract_verified",false},{"state_checksum_implemented",true}}).dump();return result.c_str();}
