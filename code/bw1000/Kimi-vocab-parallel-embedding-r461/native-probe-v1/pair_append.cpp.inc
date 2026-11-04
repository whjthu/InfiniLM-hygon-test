// New private CAPI body; this file is appended only to the new pair TU.
void ProbeEngine::embedding_probe_r461(const void*context,std::uint64_t epoch,
        const std::int64_t*ids,std::size_t tokens,std::uint16_t*out) {
    namespace probe=kimi_embedding_probe_r461;
    std::lock_guard<std::mutex> submission(submission_mutex_);
    if(workers_.size()!=8)throw std::invalid_argument("r461 eight original workers");
    auto gate=std::make_shared<probe::Gate>();
    std::array<std::shared_ptr<probe::Request>,8> requests;
    for(std::size_t d=0;d<8;++d) {
        auto request=std::make_shared<probe::Request>();
        request->worker=workers_[d].get();request->context=context;request->epoch=epoch;
        request->device=d;request->ids.assign(ids,ids+tokens);request->gate=gate;
        requests[d]=std::move(request);
    }
    probe::install(requests);
    std::vector<en::RankWorker*>submitted;
    std::exception_ptr primary;
    for(auto&w:workers_) {
        try{w->run(en::RankWorker::Input{});submitted.push_back(w.get());}
        catch(...){primary=std::current_exception();break;}
    }
    if(primary)probe::release(gate,true);
    else if(!probe::await_all_metadata(gate)) {
        primary=std::make_exception_ptr(std::runtime_error("r461 owner metadata preflight failed"));
        probe::release(gate,true);
    }else probe::release(gate,false);
    // Clear is AFTER all submitted owner wait/drain calls, including exceptions.
    // A device collective/drain that cannot return is bounded by the external
    // owned-step keeper fail-stop, not by unsafe local Tensor retirement.
    for(auto*w:submitted)try{w->wait();}catch(...){if(!primary)primary=std::current_exception();}
    if(!primary) {
        try {
            if(submitted.size()!=8)throw std::logic_error("r461 incomplete local submission");
            for(std::size_t d=0;d<8;++d) {
                const auto &request=requests[d];
                if(!request->completed||request->output.size()!=tokens*7168)
                    throw std::logic_error("r461 completed original-owner output required");
                std::memcpy(out+d*tokens*7168,request->output.data(),tokens*7168*2);
            }
        }catch(...){primary=std::current_exception();}
    }
    probe::clear(requests);
    if(primary)std::rethrow_exception(primary);
}

extern "C" int modern_kimi_embedding_probe_load_r461(void*raw,const char*descriptors) {
    Context*c=nullptr;bool started=false;
    try {
        const char*enabled=std::getenv("INFINILM_KIMI_VOCAB_PROBE_R461");
        c=static_cast<Context*>(raw);
        if(!enabled||std::string(enabled)!="1"||!c||!c->vocab_embedding_r461||!descriptors)
            throw std::invalid_argument("r461 explicit new opt-in probe context");
        auto values=nlohmann::json::parse(descriptors);
        if(!values.is_array()||values.size()!=1)throw std::invalid_argument("r461 embedding key only");
        const auto &value=values.at(0);
        const auto pointer=value.at("pointer").get<std::uintptr_t>();
        if(value.at("name")!="language_model.model.embed_tokens.weight"||value.at("dtype")!=1
            ||value.at("shape").get<rt::Shape>()!=rt::Shape{163840,7168}||!pointer)
            throw std::invalid_argument("r461 original complete CPU BF16 embedding source");
        std::lock_guard<std::mutex> lock(c->mutex);
        if(c->embedding_probe_failed_r461||c->embedding_probe_loaded_r461||c->embedding_probe_epoch_r461)
            throw std::invalid_argument("r461 single load before probe");
        std::unordered_map<std::string,rt::Tensor> source;
        source.emplace("language_model.model.embed_tokens.weight",
            cpu(reinterpret_cast<void*>(pointer),{163840,7168},rt::DataType::kBFloat16));
        started=true;c->engine->load_params(source,true); // actual Weight.select_source/LoadBatch
        c->embedding_probe_loaded_r461=true;
        return 0;
    }catch(const std::exception&failure) {
        if(c&&started)c->embedding_probe_failed_r461=true;
        error=failure.what();return 1;
    }
}
extern "C" int modern_kimi_embedding_probe_r461(void*raw,std::uint64_t epoch,
        const std::int64_t*ids,std::size_t tokens,std::uint16_t*out,std::size_t output_elements) {
    Context*c=nullptr;bool started=false;
    try {
        const char*enabled=std::getenv("INFINILM_KIMI_VOCAB_PROBE_R461");
        c=static_cast<Context*>(raw);
        if(!enabled||std::string(enabled)!="1"||!c||!c->vocab_embedding_r461
            ||!out||!epoch||tokens>512||output_elements!=8*tokens*7168)
            throw std::invalid_argument("r461 fixed original-owner probe boundary");
        kimi_vocab_r461::validate_global_ids(ids,tokens); // before any RUN/epoch mutation
        std::lock_guard<std::mutex> lock(c->mutex);
        if(!c->embedding_probe_loaded_r461||c->embedding_probe_failed_r461
            ||epoch!=c->embedding_probe_epoch_r461+1||c->engine->append_ready_r247())
            throw std::invalid_argument("r461 partial embedding-only load/probe sequence");
        started=true;c->embedding_probe_epoch_r461=epoch;
        c->engine->embedding_probe_r461(c,epoch,ids,tokens,out);
        return 0;
    }catch(const std::exception&failure) {
        if(c&&started)c->embedding_probe_failed_r461=true;
        error=failure.what();return 1;
    }
}
