#pragma once
#include "probe_protocol.hpp"
#include "vocab_parallel_embedding_r461.hpp"
#include "models/infinilm_model.hpp"
#include <cstring>
namespace kimi_embedding_probe_r461 {
inline void execute(const std::shared_ptr<Request>&request,
                    infinilm::runtime::Runtime &r,
                    infinilm::InfinilmModel &model,
                    const infinilm::engine::distributed::RankInfo &rank) {
    using namespace infinilm;
    r.assertOwnerThread();
    if(request->device!=std::size_t(r.device().index())||rank.device!=r.device()
        ||rank.tp_size!=32||rank.pp_size!=1||rank.tp_rank!=rank.world_rank
        ||rank.world_size!=32||rank.tp_rank<0||rank.tp_rank>=32)
        throw std::invalid_argument("r461 actual owner/TP32PP1");
    kimi_vocab_r461::validate_global_ids(request->ids.data(),request->ids.size());
    // The public named tree is the SAME tree used by ModelWeights::resolve.
    // No private offset/cast, model copy, new Embedding, Runtime or communicator.
    auto language=model.children().at("language_model");
    auto text=language->children().at("model");
    auto leaf=text->children().at("embed_tokens");
    auto embedding=std::dynamic_pointer_cast<layers::VocabParallelEmbeddingR461>(leaf);
    if(!embedding)throw std::invalid_argument("r461 actual opt-in embedding object required");
    const auto weights=embedding->state_dict();
    const auto backing=weights.at("weight").tensor();
    const auto &census=layers::vocab_r461::census.at(request->device);
    if(weights.size()!=1||backing->shape()!=runtime::Shape{5120,7168}
        ||backing->dtype()!=runtime::DataType::kBFloat16||backing->device()!=r.device()
        ||!backing->is_contiguous()||backing->nbytes()!=73400320
        ||reinterpret_cast<std::uintptr_t>(backing->data())!=census.backing
        ||census.rank!=std::size_t(rank.tp_rank))
        throw std::logic_error("r461 actual registry/backing census identity");
    // No device work until all eight local RUN calls have been accepted.
    metadata_ready(request);
    await_admission(request);
    runtime::Tensor input,output,host;
    try {
        embedding->process_weights_after_loading(r,false);
        auto cpu=runtime::Tensor::from_blob(const_cast<std::int64_t*>(request->ids.data()),
            {1,request->ids.size()},runtime::DataType::kInt64,{runtime::Device::Type::kCpu,0});
        input=cpu->to(r,r.device());
        output=embedding->forward_sharded(r,input); // actual production path, including its one AllReduce/fence
        host=output->to(r,{runtime::Device::Type::kCpu,0});
        r.syncStream();
        if(host->shape()!=runtime::Shape{1,request->ids.size(),7168}
            ||host->dtype()!=runtime::DataType::kBFloat16||!host->is_contiguous())
            throw std::logic_error("r461 completed output layout");
        request->output.resize(request->ids.size()*7168);
        std::memcpy(request->output.data(),host->data(),host->nbytes());
        host.reset();output.reset();input.reset();
        request->completed=true;
    }catch(...) {
        auto primary=std::current_exception();
        try{r.syncStream();}catch(...){std::terminate();}
        std::rethrow_exception(primary);
    }
}
}
