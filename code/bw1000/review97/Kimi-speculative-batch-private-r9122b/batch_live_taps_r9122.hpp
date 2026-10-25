#pragma once
#include <atomic>
#include <mutex>
#include <vector>
#include <set>
#include <array>
#include <cstring>
#include <cstdint>
namespace kimi_batch_live_r9122 {
struct Metadata{uint64_t epoch=0;std::vector<int32_t>logical,past,width;size_t total=0;};
struct Record{int64_t layer;Metadata metadata;std::vector<uint16_t>bytes;};
static std::atomic<bool>armed{false};static std::mutex mutex;static void*context=nullptr;static Metadata metadata;static std::vector<Record>records;
bool representative(infinilm::runtime::Runtime&r){auto*s=std::getenv("SLURM_PROCID"),*t=std::getenv("KIMI_TP_NODES");int node=s?std::atoi(s):0,tp=t?std::atoi(t):2;return r.device().index()==0&&tp>0&&node%tp==0;}
void collect(infinilm::runtime::Runtime&r,const infinilm::runtime::Tensor&x,const infinilm::global_state::ForwardContext&fc,size_t layer,const char*point){
 if(!armed.load(std::memory_order_acquire)||!representative(r)||std::strcmp(point,"decoder")!=0||!(layer==19||layer==37||layer==66||layer==78||layer==90))return;r.assertOwnerThread();Metadata expected;void*owner=nullptr;{std::lock_guard<std::mutex>lock(mutex);if(!armed.load())return;expected=metadata;owner=context;}
 if(r.isGraphRecording()||!x||x->dtype()!=infinilm::runtime::DataType::kBFloat16||x->shape()!=infinilm::runtime::Shape{1,expected.total,7168}||!fc.attn_metadata.past_sequence_lengths||!fc.attn_metadata.total_sequence_lengths||!fc.attn_metadata.input_offsets||!fc.attn_metadata.block_tables)throw std::logic_error("r9122 packed feature shape/metadata");
 auto past=fc.attn_metadata.past_sequence_lengths.value()->contiguous(r),total=fc.attn_metadata.total_sequence_lengths.value()->contiguous(r),offset=fc.attn_metadata.input_offsets.value()->contiguous(r),blocks=fc.attn_metadata.block_tables.value()->contiguous(r);size_t C=expected.logical.size();
 if(past->dtype()!=infinilm::runtime::DataType::kInt32||total->dtype()!=infinilm::runtime::DataType::kInt32||offset->dtype()!=infinilm::runtime::DataType::kInt32||blocks->dtype()!=infinilm::runtime::DataType::kInt32||past->numel()!=C||total->numel()!=C||offset->numel()!=C+1||blocks->shape()!=infinilm::runtime::Shape{C,4})throw std::logic_error("r9122 feature request metadata shape");
 std::vector<int32_t>pp(C),tt(C),oo(C+1),bb(C*4);r.memcpyD2H(pp.data(),past->data(),C*4);r.memcpyD2H(tt.data(),total->data(),C*4);r.memcpyD2H(oo.data(),offset->data(),(C+1)*4);r.memcpyD2H(bb.data(),blocks->data(),C*16);size_t cursor=0;
 for(size_t i=0;i<C;++i){if(pp[i]!=expected.past[i]||tt[i]-pp[i]!=expected.width[i]||oo[i]!=int32_t(cursor))throw std::logic_error("r9122 feature request past/width/offset mismatch");cursor+=size_t(expected.width[i]);for(size_t k=0;k<4;++k)if(bb[i*4+k]!=1+(expected.logical[i]-1)*4+int32_t(k))throw std::logic_error("r9122 feature logical KV row mismatch");}if(oo.back()!=int32_t(cursor)||cursor!=expected.total)throw std::logic_error("r9122 feature packed endpoint mismatch");
 auto contiguous=x->contiguous(r);Record rec{int64_t(layer),expected,std::vector<uint16_t>(x->numel())};r.memcpyD2H(rec.bytes.data(),contiguous->data(),rec.bytes.size()*2);std::lock_guard<std::mutex>lock(mutex);if(!armed.load()||context!=owner||metadata.epoch!=expected.epoch||records.size()>=5)throw std::logic_error("r9122 stale feature capture/bound");for(auto&v:records)if(v.layer==rec.layer)throw std::logic_error("r9122 duplicate feature layer");records.push_back(std::move(rec));
}
}
extern "C" int modern_kimi_batch_taps_begin_r9122(void*ctx,uint64_t epoch,int enabled,size_t count,const int32_t*logical,const int32_t*past,const int32_t*width){using namespace kimi_batch_live_r9122;try{if(!ctx||(enabled!=0&&enabled!=1)||epoch>uint64_t(INT64_MAX))return 1;Metadata next;next.epoch=epoch;if(enabled){if(!epoch||!count||count>8||!logical||!past||!width)return 1;std::set<int32_t>seen;for(size_t i=0;i<count;++i){if(logical[i]<1||logical[i]>8||!seen.insert(logical[i]).second||past[i]<0||width[i]<1||width[i]>1024||int64_t(past[i])+width[i]>1024)return 1;next.logical.push_back(logical[i]);next.past.push_back(past[i]);next.width.push_back(width[i]);next.total+=size_t(width[i]);}if(next.total>8192)return 1;}else if(count)return 1;
 std::lock_guard<std::mutex>lock(mutex);armed.store(false,std::memory_order_release);records.clear();context=ctx;metadata=std::move(next);armed.store(enabled,std::memory_order_release);return 0;}catch(...){return 1;}}
extern "C" int modern_kimi_batch_taps_count_r9122(void*ctx,size_t*out){using namespace kimi_batch_live_r9122;if(!ctx||!out)return 1;std::lock_guard<std::mutex>lock(mutex);if(ctx!=context)return 2;*out=records.size();return 0;}
extern "C" int modern_kimi_batch_taps_info_r9122(void*ctx,size_t index,int64_t*out,size_t n){using namespace kimi_batch_live_r9122;if(!ctx||!out)return 1;std::lock_guard<std::mutex>lock(mutex);if(ctx!=context||index>=records.size())return 2;auto&r=records[index];auto&m=r.metadata;if(n!=5+3*m.logical.size())return 3;out[0]=int64_t(m.epoch);out[1]=r.layer;out[2]=int64_t(m.total);out[3]=7168;out[4]=int64_t(m.logical.size());for(size_t i=0;i<m.logical.size();++i){out[5+3*i]=m.logical[i];out[6+3*i]=m.past[i];out[7+3*i]=m.width[i];}return 0;}
extern "C" int modern_kimi_batch_taps_copy_r9122(void*ctx,size_t index,void*out,size_t n){using namespace kimi_batch_live_r9122;if(!ctx||!out)return 1;std::lock_guard<std::mutex>lock(mutex);if(ctx!=context||index>=records.size())return 2;auto&r=records[index];if(n!=r.bytes.size()*2)return 3;std::memcpy(out,r.bytes.data(),n);return 0;}
extern "C" uint64_t modern_kimi_batch_taps_capabilities_r9122(){return 0x01080500;}
