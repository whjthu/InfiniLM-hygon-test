#include "embedding_index_r461.hpp"
#include "actual_Weight_Embedding_recording.hpp"
#include "long_geometry_r453.hpp"
#include <cassert>
#include <iostream>
#include <set>

int main(){
    using namespace infinilm;using runtime::DataType;using runtime::Tensor;
    runtime::Runtime r;std::size_t boundary_checks=0,loader_checks=0,rejections=0;
    assert(kimi_long_geometry_r453::head_geometry(32).rows==kimi_vocab_r461::local_rows);
    const auto original_allocations=runtime::allocations.size();
    {layers::Embedding legacy(r,163840,7168,std::nullopt,DataType::kBFloat16);
     assert(legacy.weight()->nbytes()==2348810240ULL);}
    assert(runtime::allocations.size()==original_allocations+1);
    std::set<std::int64_t> ids{0,1,163839};
    for(std::size_t rank=0;rank<32;++rank){
        ids.insert(std::int64_t(rank*5120));ids.insert(std::int64_t((rank+1)*5120-1));
        if(rank)ids.insert(std::int64_t(rank*5120-1));
    }
    for(auto id:ids){
        kimi_vocab_r461::validate_global_ids(&id,1);std::size_t contributors=0;
        for(std::size_t rank=0;rank<32;++rank){
            if(!kimi_vocab_r461::owns(id,rank))continue;
            ++contributors;assert(rank==std::size_t(id)/5120);
            for(std::size_t col:{0U,1U,13U,7167U}){
                const auto local=kimi_vocab_r461::local_index(id,rank,col);
                assert(local<5120*7168&&local/7168+rank*5120==std::size_t(id));++boundary_checks;
            }
        }
        assert(contributors==1);
    }
    RecordingContextR461 selected{true},old_default{false};
    for(std::int64_t id:std::array<std::int64_t,3>{{-1,163840,INT64_MAX}}){
        std::size_t submits=0;bool rejected=false;
        try{recording_native_input_boundary(&selected,&id,1,submits);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected&&submits==0);++rejections;
        recording_native_input_boundary(&old_default,&id,1,submits);assert(submits==1);
    }
    for(std::size_t rank=0;rank<32;++rank){
        const auto allocation_start=runtime::allocations.size();
        layers::RecordingParallel embedding(r,rank);
        assert(runtime::allocations.size()==allocation_start+1&&runtime::allocations.back()==runtime::Shape({5120,7168}));
        assert(embedding.weight()->nbytes()==73400320&&embedding.num_embeddings()==5120);
        auto binding=embedding.registered();assert(binding.tensor()->data()==embedding.weight()->data());
        auto source=Tensor::source({163840,7168},DataType::kBFloat16,64);
        const auto source_start=reinterpret_cast<std::uintptr_t>(source->data());
        std::weak_ptr<runtime::Storage> weak=source->backing;
        const auto copy_count=r.copies.size();weights::LoadBatch batch(r);
        binding.recording_load_batch(r,source,batch);
        assert(r.copies.size()==copy_count+1);const auto copy=r.copies.back();
        assert(copy.async&&copy.bytes==73400320&&copy.src==source_start+rank*73400320ULL);
        assert(copy.src+copy.bytes<=source_start+2348810240ULL);
        if(rank==31)assert(copy.src+copy.bytes==source_start+2348810240ULL);
        assert(copy.dst==reinterpret_cast<std::uintptr_t>(embedding.weight()->data()));
        source.reset();assert(!weak.expired());const auto syncs=r.syncs;batch.drain();
        assert(r.syncs==syncs+1&&weak.expired());++loader_checks;
    }
    {
        layers::RecordingParallel embedding(r,31);auto binding=embedding.registered();
        auto source=Tensor::source({163840,7168},DataType::kBFloat16);
        std::weak_ptr<runtime::Storage> weak=source->backing;weights::LoadBatch batch(r);r.throw_copy=true;
        bool rejected=false;try{binding.recording_load_batch(r,source,batch);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);source.reset();assert(!weak.expired());r.throw_copy=false;
        batch.drain();assert(weak.expired());++loader_checks;
        for(const auto &bad:std::vector<Tensor>{Tensor::source({163840,7168},DataType::kFloat32),
                Tensor::source({163839,7168},DataType::kBFloat16),Tensor::source({1,163840,7168},DataType::kBFloat16)}){
            bool bad_rejected=false;try{binding.load_no_sync(r,bad);}catch(const std::runtime_error&){bad_rejected=true;}
            assert(bad_rejected);++rejections;
        }
    }
    // One-hot finite BF16 payload algebra; not an SDK/device numerical test.
    for(auto id:ids)for(std::size_t col:{0U,1U,7167U}){
        const std::uint16_t expected=std::uint16_t(0x3f00U+((std::size_t(id)+col)%127));
        std::uint16_t result=0;std::size_t calls=0,contributors=0;
        for(std::size_t rank=0;rank<32;++rank){
            const auto local=kimi_vocab_r461::owns(id,rank)?expected:std::uint16_t(0);
            if(local){result=local;++contributors;}
            ++calls; // All32 participate, including ranks owning no selected rows.
        }
        assert(calls==32&&contributors==1&&result==expected);++boundary_checks;
    }
    std::cout<<"{\"passed\":true,\"all32_loader_slices\":"<<loader_checks
             <<",\"boundary_checks\":"<<boundary_checks<<",\"rejections\":"<<rejections
             <<",\"recording_Runtime_Tensor_only\":true,\"actual_SDK\":false,\"GPU\":0,\"Model\":0}\n";
}
