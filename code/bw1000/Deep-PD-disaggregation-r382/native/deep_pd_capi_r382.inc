// Included at the end of the private original-CAPI TU; shares its opaque Context.
namespace {
using namespace infinilm::pd_r382;
Context &pd_context(void *raw){if(!raw)throw std::invalid_argument("PD null context");return *static_cast<Context*>(raw);}
PDSession &pd_session(Context &c,uint64_t id){if(c.pd_failed||!c.session||!id||c.session->id!=id)throw std::invalid_argument("PD stale/missing session");return *c.session;}
Block &pd_block(PDSession &s,uint32_t rank,uint32_t block){if(rank>=8||block>=s.op->shards[rank].blocks.size())throw std::invalid_argument("PD block index");return s.op->shards[rank].blocks[block];}
void pd_new(Context &c,uint64_t epoch,uint64_t *sid){
    if(!sid||c.pd_failed||c.session||!epoch||epoch<=c.last_epoch||c.next_sid==UINT64_MAX)
        throw std::invalid_argument("PD active/failed/stale epoch");
}
}
extern "C" const char *deep_pd_state_abi_r382(){return infinilm::pd_r382::ABI;}
extern "C" uint64_t deep_pd_state_generation_r382(void *raw){try{auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);if(c.pd_failed)throw std::invalid_argument("PD failed context");return c.generation;}catch(const std::exception&e){error=e.what();return 0;}}
extern "C" int deep_pd_state_begin_export_r382(void *raw,uint64_t past,uint64_t epoch,uint64_t *sid){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);pd_new(c,epoch,sid);
    auto session=std::make_unique<PDSession>();session->id=c.next_sid++;session->epoch=epoch;
    session->op=std::make_shared<Operation>(Action::Export,past);
    try{c.engine->pd_control(session->op);}catch(...){c.pd_failed=true;throw;}
    session->descriptor=describe(*session->op,epoch,c.generation).dump();
    if(session->descriptor.size()>1024*1024)throw std::runtime_error("PD descriptor exceeds bounded wire header");
    c.last_epoch=epoch;*sid=session->id;c.session=std::move(session);return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char *deep_pd_state_descriptor_r382(void *raw,uint64_t sid){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);return pd_session(c,sid).descriptor.c_str();
}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int deep_pd_state_read_r382(void *raw,uint64_t sid,uint32_t rank,uint32_t block,uint64_t offset,void *dst,uint64_t n){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);auto &s=pd_session(c,sid);
    if(s.importing)throw std::invalid_argument("PD cannot read import session");auto &b=pd_block(s,rank,block);
    if((n&&!dst)||offset>b.bytes.size()||n>b.bytes.size()-offset)throw std::invalid_argument("PD read bounds");
    if(n)std::memcpy(dst,b.bytes.data()+offset,n);return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_pd_state_begin_import_r382(void *raw,const char *text,uint64_t epoch,uint64_t generation,uint64_t *sid){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);pd_new(c,epoch,sid);
    if(generation!=c.generation||!text||std::strlen(text)>1024*1024)throw std::invalid_argument("PD target generation/manifest bound");
    auto manifest=Json::parse(text);
    if(manifest.at("epoch")!=epoch)throw std::invalid_argument("PD source epoch differs");
    uint64_t past=manifest.at("past");if(!past||past>=32768)throw std::invalid_argument("PD past bound");
    auto session=std::make_unique<PDSession>();session->id=c.next_sid++;session->epoch=epoch;session->importing=true;
    session->op=std::make_shared<Operation>(Action::Describe,past);
    try{c.engine->pd_control(session->op);}catch(...){c.pd_failed=true;throw;}
    check_manifest(manifest,*session->op);
    for(unsigned r=0;r<8;++r)for(unsigned b=0;b<273;++b)session->op->shards[r].blocks[b].layout=manifest.at("shards").at(r).at("blocks").at(b);
    session->descriptor=manifest.dump();session->op->action=Action::Import;
    c.last_epoch=epoch;*sid=session->id;c.session=std::move(session);return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_pd_state_write_r382(void *raw,uint64_t sid,uint32_t rank,uint32_t block,uint64_t offset,const void *src,uint64_t n){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);auto &s=pd_session(c,sid);
    if(!s.importing||s.validated)throw std::invalid_argument("PD session is not writable");auto &b=pd_block(s,rank,block);
    // A block is a monotonically increasing byte stream: no holes, overwrite or replay.
    if((n&&!src)||offset!=b.received||offset>b.bytes.size()||n>b.bytes.size()-offset)throw std::invalid_argument("PD write bounds/order");
    if(n)std::memcpy(b.bytes.data()+offset,src,n);b.received+=n;return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_pd_state_validate_r382(void *raw,uint64_t sid){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);auto &s=pd_session(c,sid);
    if(!s.importing)throw std::invalid_argument("PD validation needs import session");validate_payload(*s.op);s.validated=true;return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_pd_state_commit_r382(void *raw,uint64_t sid){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);auto &s=pd_session(c,sid);
    if(!s.importing||!s.validated)throw std::invalid_argument("PD unvalidated commit");
    try{c.engine->pd_control(s.op);}catch(...){c.pd_failed=true;throw;}
    c.session.reset();return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_pd_state_abort_r382(void *raw,uint64_t sid){try{
    auto &c=pd_context(raw);std::lock_guard<std::mutex>lock(c.mutex);pd_session(c,sid);c.session.reset();return 0;
}catch(const std::exception&e){error=e.what();return 1;}}
