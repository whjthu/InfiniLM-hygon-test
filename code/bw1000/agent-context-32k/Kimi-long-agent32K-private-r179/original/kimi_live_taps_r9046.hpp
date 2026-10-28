// Default-OFF bounded live feature export. Host buffers own their bytes until next begin.
#include <atomic>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdint>
namespace kimi_live_taps_r9046 {
struct Record{int64_t layer,begin,tokens;std::vector<uint16_t>bytes;};
std::atomic<bool>armed{false};int tap_count=5;std::mutex mutex;std::vector<Record>records;void*context=nullptr;uint64_t epoch=0;
bool representative(infinilm::runtime::Runtime&r){auto*s=std::getenv("SLURM_PROCID");int n=s?std::atoi(s):0;return r.device().index()==0&&(n==0||n==2);}
void collect(infinilm::runtime::Runtime&r,const infinilm::runtime::Tensor&x,const infinilm::global_state::ForwardContext&fc,size_t layer,const char*point){
 if(!armed.load(std::memory_order_acquire)||!representative(r)||std::strcmp(point,"decoder")!=0||!(layer==19||layer==37||(layer==54&&tap_count==6)||layer==66||layer==78||layer==90))return;
 r.assertOwnerThread();if(r.isGraphRecording()||!x||x->dtype()!=infinilm::runtime::DataType::kBFloat16||x->ndim()!=3||x->size(0)!=1||x->size(2)!=7168||x->size(1)<1||x->size(1)>512)throw std::logic_error("9046 eagerB1H7168 rows1..512 required");
 auto past=fc.attn_metadata.past_sequence_lengths.value()->contiguous(r),total=fc.attn_metadata.total_sequence_lengths.value()->contiguous(r);
 if(past->dtype()!=infinilm::runtime::DataType::kInt32||total->dtype()!=infinilm::runtime::DataType::kInt32||past->numel()!=1||total->numel()!=1)throw std::logic_error("9046 B1I32 metadata required");
 int32_t begin=0,end=0;r.memcpyD2H(&begin,past->data(),4);r.memcpyD2H(&end,total->data(),4);if(begin<0||end-begin!=int64_t(x->size(1)))throw std::logic_error("9046 position mismatch");
 auto t=x->contiguous(r);Record rec{int64_t(layer),begin,int64_t(x->size(1)),std::vector<uint16_t>(x->numel())};r.memcpyD2H(rec.bytes.data(),t->data(),rec.bytes.size()*2);
 std::lock_guard<std::mutex>g(mutex);if(!armed.load()||records.size()>=size_t(tap_count))throw std::logic_error("9046 arm/buffer bound");for(auto&v:records)if(v.layer==rec.layer)throw std::logic_error("9046 duplicate layer");records.push_back(std::move(rec));
}
}
namespace kimi_live_taps_r9046 {
int begin_mode(void*ctx,uint64_t new_epoch,int enabled,int count){try{if(!ctx||(enabled!=0&&enabled!=1)||(count!=5&&count!=6))return 1;std::lock_guard<std::mutex>g(mutex);armed.store(false,std::memory_order_release);records.clear();context=ctx;epoch=new_epoch;tap_count=count;armed.store(enabled,std::memory_order_release);return 0;}catch(...){return 1;}}
}
// The old entry point always reinstates exactly the original five-tap contract.
extern "C" int modern_kimi_live_taps_begin_r9046(void*ctx,uint64_t epoch,int enabled){return kimi_live_taps_r9046::begin_mode(ctx,epoch,enabled,5);}
// Opt in independently per native call. No default, compute, or SDK change.
extern "C" int modern_kimi_live_taps_begin_r9112(void*ctx,uint64_t epoch,int enabled){return kimi_live_taps_r9046::begin_mode(ctx,epoch,enabled,6);}
extern "C" int modern_kimi_live_taps_capabilities_r9112(uint64_t*out){if(!out)return 1;*out=0x06050200ULL;return 0;}

extern "C" int modern_kimi_live_taps_count_r9046(void*ctx,size_t*out){using namespace kimi_live_taps_r9046;if(!ctx||!out)return 1;std::lock_guard<std::mutex>g(mutex);if(ctx!=context)return 2;*out=records.size();return 0;}
extern "C" int modern_kimi_live_taps_info_r9046(void*ctx,size_t index,int64_t*out,size_t count){using namespace kimi_live_taps_r9046;if(!ctx||!out||count!=6)return 1;std::lock_guard<std::mutex>g(mutex);if(ctx!=context||index>=records.size())return 2;auto&v=records[index];out[0]=epoch;out[1]=v.layer;out[2]=v.begin;out[3]=v.tokens;out[4]=7168;out[5]=v.bytes.size()*2;return 0;}
extern "C" int modern_kimi_live_taps_copy_r9046(void*ctx,size_t index,void*out,size_t bytes){using namespace kimi_live_taps_r9046;if(!ctx||!out)return 1;std::lock_guard<std::mutex>g(mutex);if(ctx!=context||index>=records.size()||bytes!=records[index].bytes.size()*2)return 2;std::memcpy(out,records[index].bytes.data(),bytes);return 0;}

extern "C" int modern_kimi_live_taps_count_r9112(void*ctx,size_t*out){return modern_kimi_live_taps_count_r9046(ctx,out);}
extern "C" int modern_kimi_live_taps_info_r9112(void*ctx,size_t index,int64_t*out,size_t count){return modern_kimi_live_taps_info_r9046(ctx,index,out,count);}
extern "C" int modern_kimi_live_taps_copy_r9112(void*ctx,size_t index,void*out,size_t bytes){return modern_kimi_live_taps_copy_r9046(ctx,index,out,bytes);}
