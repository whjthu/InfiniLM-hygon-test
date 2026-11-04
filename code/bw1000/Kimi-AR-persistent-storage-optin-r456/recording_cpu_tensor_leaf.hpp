#pragma once
// Recording CPU Tensor leaf only. Production freeze_input/CPU-copy/template
// code is extracted byte-for-byte from the SHA-bound production header.
// No SDK/Native/Runtime is linked or executed by this recording consumer.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
namespace infinilm::runtime {
using Shape=std::vector<size_t>;
enum class DataType{kInt32,kInt64};
struct Device{enum class Type{kCpu,kHygon};Type t;int index;Type type()const{return t;}};
struct RecordingStorage{
 void*p;Shape dims;DataType dt;Device dev;std::shared_ptr<void>lease;
 size_t numel()const{size_t n=1;for(size_t x:dims)n*=x;return n;}
 size_t nbytes()const{return numel()*(dt==DataType::kInt64?8:4);}
 DataType dtype()const{return dt;}Device device()const{return dev;}
 size_t ndim()const{return dims.size();}const Shape&shape()const{return dims;}
 bool is_contiguous()const{return true;}void*data()const{return p;}
};
class Tensor{
 std::shared_ptr<RecordingStorage>p_;
public:
 Tensor()=default;explicit Tensor(std::shared_ptr<RecordingStorage>p):p_(std::move(p)){}
 RecordingStorage*operator->()const{return p_.get();}explicit operator bool()const{return bool(p_);}
 static Tensor from_blob(void*p,Shape s,DataType d,Device v,std::shared_ptr<void>lease={}){return Tensor(std::make_shared<RecordingStorage>(RecordingStorage{p,std::move(s),d,v,std::move(lease)}));}
};
}
namespace infinilm::models::kimi_native {
namespace rt=infinilm::runtime;
struct Request{size_t row,past,tokens;};
struct RecordingSpecProof{const void*owner=nullptr;size_t capacity=0;std::vector<int32_t>kv,read,write;};
inline const RecordingSpecProof*active_spec_proof_r3632=nullptr;
struct RecordingInput{
 std::optional<rt::Tensor>input_ids,position_ids,past_sequence_lengths,total_sequence_lengths,input_offsets,cu_seqlens,mamba_init_state_indices,mamba_final_state_indices,block_tables,slot_mapping;
 bool sample_all_positions=false;
 bool nsa_owner_snapshot=false,pixel_values=false,image_bound=false,tgt_sizes=false,image_grid_thw=false,image_req_ids=false,visual_token_ranges=false,target_hidden_states=false,random_values=false,sampling_rows=false;
 int top_k=1;float top_p=1.f,temperature=1.f;
};
}
