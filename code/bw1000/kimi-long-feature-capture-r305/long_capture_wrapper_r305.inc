
// Independent default-OFF long-feature entry. Existing API and AR never call it.
#include "long_capture_geometry_r305.hpp"
#include "long_capture_api_r305.hpp"
extern "C" int modern_kimi_long_features_begin_r305(void*raw,uint64_t epoch,int enabled,size_t count,const int32_t*logical,const int32_t*past,const int32_t*width){try{
 if(!raw||(enabled!=0&&enabled!=1))throw std::invalid_argument("r305 context/enabled");
 if(!enabled){if(count||logical||past||width)throw std::invalid_argument("r305 disabled metadata must be empty");return modern_kimi_batch_taps_begin_r9122(raw,epoch,0,0,nullptr,nullptr,nullptr);}
 const char*opt=std::getenv("INFINILM_KIMI_LONG_FEATURES_R305");if(!opt||std::strcmp(opt,"1")!=0)throw std::invalid_argument("r305 long features defaultOFF");
 auto*c=static_cast<Context*>(raw);if(!c->pair_r9006||!c->engine||!c->cache)throw std::invalid_argument("r305 incomplete Context");
 std::lock_guard<std::mutex>pair_lock(c->pair_mutex_r9006);std::lock_guard<std::mutex>engine_lock(c->mutex);c->pair_r9006->owner(c);
 const auto&g=c->pair_r9006->geometry;
 if(!c->batch_enabled_r9122||!c->speculation_r3632||c->ordinary_epoch_r4242)throw std::invalid_argument("r305 scoped batch pair Context required");
 const kimi_long_features_r305::ContextGeometry actual{c->long_agent_r179,g.long_agent,c->engine->append_ready_r247(),g.logical,c->max_rows,g.state_rows(),c->max_total_tokens_r179,c->pages_per_row_r4974,c->cache->num_blocks(),c->cache->block_size(),c->cache->max_batch_size(),c->tp_nodes,c->pp_size,c->stage};
 auto scope=kimi_long_features_r305::validate_context(actual);kimi_long_features_r305::validate_metadata(scope,epoch,count,logical,past,width);
 if(kimi_batch_live_r9122::begin_long_validated_r305(raw,epoch,count,logical,past,width))throw std::invalid_argument("r305 long capture arm failed");return 0;
 }catch(const std::exception&e){error=e.what();return 1;}catch(...){error="r305 unknown long feature failure";return 1;}}
extern "C" uint64_t modern_kimi_long_features_capabilities_r305(){return 0x01320080;}
