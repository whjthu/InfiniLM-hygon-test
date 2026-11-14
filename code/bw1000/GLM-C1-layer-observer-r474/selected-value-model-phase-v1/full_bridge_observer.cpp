#include "selected_value_phase.hpp"
// Same-TU typed Context family. Never reinterpret a component or oldpublicctx.
#include "geometry.hpp"
#include "policy_r403.hpp"
#include "private_bridge_base.cpp"
namespace {
struct GlobalModel470{static constexpr uint64_t signature=0x47545031364d3437ULL;uint64_t magic=signature;std::unique_ptr<Context>inner;nlohmann::json cfg,geo;std::weak_ptr<rt::Runtime>weak;bool poisoned=false;};
thread_local std::string global_error470,global_serialized470;
GlobalModel470&global470(void*raw,bool allow_poison=false){auto*g=static_cast<GlobalModel470*>(raw);if(!g||g->magic!=GlobalModel470::signature||!g->inner||(!allow_poison&&g->poisoned))throw std::invalid_argument("r470 typedglobalModel context");g->inner->owner->assertOwnerThread();return*g;}
void report470(const std::exception&e){global_error470=e.what();}
bool flag470(const char*name){const char*p=std::getenv(name);if(!p||std::string(p)=="0")return false;if(std::string(p)=="1")return true;throw std::invalid_argument("r470 flag0or1");}
int result470(GlobalModel470&g,int rc){if(rc){g.poisoned=true;global_error470=r470_internal_modern_glm_model_error();}return rc;}
}
extern "C" const char*modern_glm_global_error_r470(){return global_error470.c_str();}
extern "C" size_t modern_glm_global_id_size_r470(){return r470_internal_modern_glm_model_id_size();}
extern "C" int modern_glm_global_id_r470(void*out,size_t size){if(!out||size!=sizeof(infinicclUniqueId))return 1;return r470_internal_modern_glm_model_id(out);}
extern "C" void*modern_glm_global_create_r470(int node,int device,size_t rows,size_t capacity,const char*rawcfg,const void*world_id,const void*tp_id,size_t idbytes){std::unique_ptr<GlobalModel470>g;try{
 if(!rawcfg||!world_id||!tp_id||idbytes!=sizeof(infinicclUniqueId)||std::memcmp(world_id,tp_id,idbytes)==0)throw std::invalid_argument("r470 exactdistinctcommUIDs");auto cfg=nlohmann::json::parse(rawcfg);if(!cfg.value("glm_global_tp16_Model_r470",false)||!flag470("GLM_GLOBAL_TP16_MODEL_R470"))throw std::invalid_argument("r470 Model JSON/env optin");glm_global_tp16_r470::Scope scope(cfg);if(!model::v_on_demand_r403::requested())throw std::invalid_argument("r470 full32K requires explicit original NoV policy");auto geo=glm_global_tp16_r470::geometry(cfg,node,device,rows,capacity);if(cfg.value("glm_vocab_parallel_embedding_r471",false)){geo["replicated_embedding_BF16_bytes"]=0;geo["embedding_BF16_global_bytes"]=size_t(154880)*6144*2;geo["embedding_BF16_local_bytes"]=size_t(9680)*6144*2;geo["embedding_vocab_begin"]=size_t(node*8+device)*9680;geo["embedding_vocab_end"]=size_t(node*8+device+1)*9680;geo["embedding_reconstruction"]= "one masked local lookup plus existing same-stream TP SUM; no legacy ABI";}if(rows!=1||capacity!=32768)throw std::invalid_argument("r470 C1 full32K");
 // Fresh strongRuntime only; do not borrow the oldprivatecomponent or ownerspool.
 if(!owners().empty())throw std::logic_error("r470 do notmix legacy contexts");g=std::make_unique<GlobalModel470>();g->cfg=cfg;g->geo=geo;g->inner=std::make_unique<Context>();auto&c=*g->inner;c.owner=rt::Runtime::create({rt::Device::Type::kHygon,device});c.owner->enableOwnerDrainFailStop();c.owner->assertOwnerThread();g->weak=c.owner;c.stage=0;c.pp=1;c.capacity_r1406=capacity;c.rows_r5688=rows;c.bounded_rope_r4279=flag470("GLM_BOUNDED_ROPE_R4279");if(!c.bounded_rope_r4279)throw std::invalid_argument("r470 original bounded32768 rope required");c.shared_control_r4279=true;auto&r=*c.owner;infini::rt::DeviceGuard guard{r.device().type(),device,true};infinicclUniqueId world{},tp{};std::memcpy(&world,world_id,idbytes);std::memcpy(&tp,tp_id,idbytes);const int rank=node*8+device;
 if(infinicclCommInitRank(&c.world,16,world,rank)!=infinicclSuccess)throw std::runtime_error("r470 Modelworld16init");if(infinicclCommInitRank(&c.tp,16,tp,rank)!=infinicclSuccess)throw std::runtime_error("r470 ModelTP16init");c.control=c.world;
 infinilm::engine::distributed::RankInfo info(r.device());info.tp_size=16;info.tp_rank=rank;info.comm=c.tp;info.pp_size=1;info.pp_stage=0;info.world_size=16;info.world_rank=rank;info.world_comm=c.world;if(!glm_global_tp16_r470::admitted(info))throw std::logic_error("r470 exactglobalRankInfo");model::rope_bound_r4279::Scope rope(r,c.bounded_rope_r4279,capacity);c.module=std::make_unique<model::TextModelProjectedR22>(r,cfg,info,rows,capacity,&c.bypass_r1406,&c.owned_prefix_r1669);r.syncStream();return g.release();
 }catch(const std::exception&e){report470(e);if(g&&g->inner&&g->inner->owner){glm_global_tp16_r470::Scope cleanup_scope(g->cfg);auto&c=*g->inner;drain(&c);if(c.world&&c.tp){int rc=r470_internal_modern_glm_model_destroy(&c);if(rc)std::terminate();g->inner.release();}else{c.module.reset();if(c.tp&&infinicclCommDestroy(c.tp)!=infinicclSuccess)std::terminate();if(c.world&&infinicclCommDestroy(c.world)!=infinicclSuccess)std::terminate();g->inner.reset();}}return nullptr;}}
extern "C" const char*modern_glm_global_keys_r470(void*raw){try{auto&g=global470(raw);glm_global_tp16_r470::Scope scope(g.cfg);return r470_internal_modern_glm_model_keys(g.inner.get());}catch(const std::exception&e){report470(e);return nullptr;}}
extern "C" int modern_glm_global_load_batch_r470(void*raw,const char*desc){try{auto&g=global470(raw);glm_global_tp16_r470::Scope scope(g.cfg);return result470(g,r470_internal_modern_glm_model_load_batch(g.inner.get(),desc));}catch(const std::exception&e){report470(e);return 1;}}
extern "C" int modern_glm_global_finish_r470(void*raw){try{auto&g=global470(raw);glm_global_tp16_r470::Scope scope(g.cfg);return result470(g,r470_internal_modern_glm_model_finish(g.inner.get()));}catch(const std::exception&e){report470(e);return 1;}}
extern "C" int modern_glm_global_forward_r470(void*raw,size_t tokens,size_t count,const size_t*requests,const void*input,const void*positions,void*logits,int all_positions,void*stream){try{auto&g=global470(raw);glm_global_tp16_r470::Scope scope(g.cfg);return result470(g,r470_internal_modern_glm_model_forward(g.inner.get(),tokens,count,requests,input,positions,logits,all_positions,stream));}catch(const std::exception&e){report470(e);return 1;}}
extern "C" int modern_glm_global_control_r470(void*raw,size_t count,void*values,void*stream){try{auto&g=global470(raw);if(!count||!values)throw std::invalid_argument("r470 control count/pointer");glm_global_tp16_r470::Scope scope(g.cfg);return result470(g,r470_internal_modern_glm_model_control(g.inner.get(),count,values,stream));}catch(const std::exception&e){report470(e);return 1;}}
extern "C" int modern_glm_global_reset_r470(void*raw){try{auto&g=global470(raw);glm_global_tp16_r470::Scope scope(g.cfg);return result470(g,r470_internal_modern_glm_model_reset(g.inner.get()));}catch(const std::exception&e){report470(e);return 1;}}
extern "C" const char*modern_glm_global_geometry_r470(void*raw){try{auto&g=global470(raw,true);g.geo["Model_constructed"]=bool(g.inner->module);g.geo["cache_allocated"]=bool(g.inner->module);g.geo["weights_loaded"]=g.inner->ready;g.geo["Model_READY"]=g.inner->ready&&!g.poisoned;g.geo["poisoned"]=g.poisoned;g.geo["loaded_parameters"]=g.inner->loaded.size();g.geo["root_output_rank"]=0;g.geo["profile"]="GLM-globalTP16PP1-C1-NoV-explicit-off-r470";global_serialized470=g.geo.dump();return global_serialized470.c_str();}catch(const std::exception&e){report470(e);return nullptr;}}
extern "C" int modern_glm_global_destroy_r470(void*raw){try{auto&g=global470(raw,true);glm_global_tp16_r470::Scope scope(g.cfg);int rc=r470_internal_modern_glm_model_destroy(g.inner.get());if(rc)return result470(g,rc);g.inner.release();if(!g.weak.expired()){g.poisoned=true;throw std::runtime_error("r470 ModelRuntime retained");}g.magic=0;delete &g;return 0;}catch(const std::exception&e){report470(e);return 1;}}

// New typed diagnostic entry points. Every legacy Context API receives g.inner,
// while the sparse hook registry receives its actual Runtime owner pointer.
namespace {
void observer_optin_r474(GlobalModel470&g){if(!g.cfg.value("glm_c1_layer_observer_r474",false)||!flag470("GLM_C1_LAYER_OBSERVER_R474")||g.geo.at("TP")!=16||g.geo.at("PP")!=1||g.geo.at("rows")!=1||g.geo.at("capacity_per_row")!=32768)throw std::invalid_argument("r474 explicit globalTP16 C1/full32K observer optin");}
}
extern "C" int modern_glm_global_sparse_prepare_r474(void*raw){try{auto&g=global470(raw);observer_optin_r474(g);glm_global_tp16_r470::Scope scope(g.cfg);int rc=r470_internal_modern_glm_sparse_prepare_r131(g.inner.get());if(rc)return result470(g,rc);glm_sparse_r131::select_c1_r474(g.inner->owner.get());return 0;}catch(const std::exception&e){report470(e);return 1;}}
extern "C" int modern_glm_global_sparse_policy_r474(void*raw,int enabled){try{auto&g=global470(raw);observer_optin_r474(g);glm_global_tp16_r470::Scope scope(g.cfg);if(!glm_sparse_r131::c1_next_decode_r474.count(g.inner->owner.get()))throw std::logic_error("r474 prepare required");int rc=r470_internal_modern_glm_sparse_policy_r131(g.inner.get(),enabled);if(rc)return result470(g,rc);glm_sparse_r131::c1_next_decode_r474.at(g.inner->owner.get())=false;return 0;}catch(const std::exception&e){report470(e);return 1;}}
extern "C" int modern_glm_global_sparse_next_decode_r474(void*raw){try{auto&g=global470(raw);observer_optin_r474(g);glm_global_tp16_r470::Scope scope(g.cfg);auto&r=*g.inner->owner;if(!g.inner->ready||r.isGraphRecording()||model::decode_leases_r1354::active(r))throw std::logic_error("r474 loaded eager original owner");glm_sparse_r131::check_current_device(r.device().index());glm_sparse_r131::next_decode_r474(&r);return 0;}catch(const std::exception&e){report470(e);return 1;}}
extern "C" const char*modern_glm_global_sparse_read_r474(void*raw){try{auto&g=global470(raw,true);observer_optin_r474(g);glm_global_tp16_r470::Scope scope(g.cfg);auto j=glm_sparse_r131::read_c1_r474(g.inner->owner.get());j["global_tp_rank"]=g.geo.at("tp_rank");j["scope"]="at most3 diagnostic decode frames; overlapping originalstream event/host windows, not exclusiveGPUbusy";global_serialized470=j.dump();return global_serialized470.c_str();}catch(const std::exception&e){report470(e);return nullptr;}}

// Read only at drained Model boundaries. The layer registry is keyed by Runtime.
extern "C" int modern_glm_selected_value_layer0_read(void*, uint64_t*, size_t);

extern "C" int modern_glm_global_selected_value_layer0_read(
    void* handle, uint64_t* out, size_t count) {
    try {
        auto& context = global470(handle);
        if (!out || count != 3 || !context.inner->ready)
            throw std::invalid_argument("selected-value counters require a loaded Model and three outputs");
        auto& runtime = *context.inner->owner;
        if (runtime.isGraphRecording() || model::decode_leases_r1354::active(runtime))
            throw std::logic_error("selected-value counters require an eager owner");
        glm_sparse_r131::check_current_device(runtime.device().index());
        const int result = modern_glm_selected_value_layer0_read(&runtime, out, count);
        if (result != 0) throw std::runtime_error("selected-value layer zero registry unavailable");
        return 0;
    } catch (const std::exception& exc) {
        report470(exc);
        return 1;
    }
}

// Baseline and packed continuations share one loaded Model and one dispatch entry.
extern "C" int modern_glm_global_selected_value_layer0_forward(
    void* handle, size_t tokens, size_t count, const size_t* requests,
    const void* input, const void* positions, void* logits, int all_positions,
    void* stream, int use_packed) {
    try {
        auto& context = global470(handle);
        if (!context.inner->ready)
            throw std::logic_error("selected-value requires loaded Model");
        if (use_packed == 1 && (all_positions != 0 ||
            !context.cfg.value("glm_selected_value_packed_layer0", false)))
            throw std::invalid_argument("selected-value packed forward requires the loaded opt-in");
        glm_global_tp16_r470::Scope scope(context.cfg);
        ::glm_selected_value_layer0::ForwardScope phase(
            context.inner->owner.get(), use_packed, tokens, count, requests);
        return result470(context, r470_internal_modern_glm_model_forward(
            context.inner.get(), tokens, count, requests, input, positions, logits, all_positions, stream));
    } catch (const std::exception& exc) {
        report470(exc);
        return 1;
    }
}
