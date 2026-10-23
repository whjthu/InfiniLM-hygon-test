#pragma once
// Validation image: retain the replicated head and a shard for same-load ABBA.
// This image does not demonstrate a Model-level memory saving.
#include "kimi_vocab_parallel_r8296.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
namespace infinilm::layers::linear::head_policy_r8341 {
inline std::mutex mutex;
inline std::once_flag initialized;
inline std::atomic<int> policy{0};
inline size_t in_flight=0;
struct Counts {std::atomic<uint64_t> baseline{0},sharded{0},rows{0};};
inline std::array<Counts,8> counts;
inline void initialize(){std::call_once(initialized,[]{const char*p=std::getenv("INFINILM_KIMI_VOCAB_HEAD_R8341");if(!p)return;if(std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("8341 head flag must exactly0or1");policy.store(p[0]-'0');});}
class Flight {
public:
 Flight(){initialize();std::lock_guard<std::mutex>lock(mutex);++in_flight;}
 ~Flight(){std::lock_guard<std::mutex>lock(mutex);--in_flight;}
 Flight(const Flight&)=delete;Flight&operator=(const Flight&)=delete;
};
}
namespace infinilm::layers::linear {
class DualVocabHeadR8341 final:public ReplicatedLinear {
    infinilm::runtime::Runtime*const owner_;
    std::shared_ptr<VocabParallelR8296> candidate_;
    bool ready_=false;
public:
    DualVocabHeadR8341(infinilm::runtime::Runtime&r,size_t hidden,size_t vocab,bool bias,
       infinilm::runtime::DataType dtype,std::optional<infinilm::runtime::DataType> output,
       const engine::distributed::RankInfo&rank)
      :ReplicatedLinear(r,hidden,vocab,bias,dtype,output),owner_(&r){
        head_policy_r8341::initialize();
        if(bias||dtype!=infinilm::runtime::DataType::kBFloat16||output.value_or(dtype)!=dtype
           ||hidden!=7168||vocab!=163840||rank.tp_size!=8||rank.device!=r.device())
            throw std::invalid_argument("8341 validated BF16 Kimi TP8 head geometry required");
        try{candidate_=std::make_shared<VocabParallelR8296>(r,hidden,vocab,dtype,output,rank);}
        catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
    }
    void process_weights_after_loading(infinilm::runtime::Runtime&r,bool transpose)override{
        r.assertOwnerThread();
        if(&r!=owner_||ready_||transpose)throw std::logic_error("8341 one-time native-layout postload required");
        auto full=weight();
        try{
            ReplicatedLinear::process_weights_after_loading(r,false);
            // The ordinary checkpoint key is unchanged. The private shard is
            // copied once from the loaded full GPU tensor; no disk reload.
            candidate_->load_parameters_no_sync(r,{{"weight",full}},true);
            candidate_->process_weights_after_loading(r,false);r.syncStream();
            if(full->nbytes()!=2348810240ULL||candidate_->weight()->nbytes()!=293601280ULL)
                throw std::logic_error("8341 full/shard head byte geometry");
            ready_=true;
        }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
    }
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime&r,infinilm::runtime::Tensor&input)const{
        r.assertOwnerThread();if(&r!=owner_||!ready_||r.device().index()<0||r.device().index()>=8)
            throw std::logic_error("8341 head Runtime/ready mismatch");
        auto&c=head_policy_r8341::counts[r.device().index()];
        if(head_policy_r8341::policy.load()==1){auto result=candidate_->forward(r,input);++c.sharded;c.rows.fetch_add(input->numel()/7168);return result;}
        auto result=ReplicatedLinear::forward(r,input);++c.baseline;return result;
    }
};
}
