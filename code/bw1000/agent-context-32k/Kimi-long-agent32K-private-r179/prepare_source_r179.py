"""Deterministic local source transformation; never compiles or loads a DSO."""
import pathlib,json,hashlib
D=pathlib.Path(__file__).resolve().parent;O=D/'original';N=D/'native';log=[]
for p in O.rglob('*'):
 if p.is_file():q=N/p.relative_to(O);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
def edit(name,old,new,n=1):
 p=N/name;s=p.read_text();found=s.count(old);assert found==n,(name,old[:100],found,n);p.write_text(s.replace(old,new));log.append({'file':name,'anchor_SHA256':hashlib.sha256(old.encode()).hexdigest(),'occurrences':found})
# Private Geometry fields default to legacy1024/minphysical16. Explicit long=true
# is admitted only by the new factory and requires logical1/context32768.
edit('kimi_pair_state_r9006.hpp','size_t logical,physical,half;','size_t logical,physical,half,context;bool long_agent;')
edit('kimi_pair_state_r9006.hpp','explicit Geometry(size_t b):logical(b),physical(std::max(size_t(16),2*b)),half(physical/2){','explicit Geometry(size_t b,bool scoped_long=false):logical(b),physical(scoped_long?2*b:std::max(size_t(16),2*b)),half(physical/2),context(scoped_long?32768:1024),long_agent(scoped_long){\n  if(scoped_long&&b!=1)throw std::invalid_argument("r179 long scope requires logical1");')
edit('kimi_pair_state_r9006.hpp','State(const void*owner,size_t b):owner_(owner),geometry(b)','State(const void*owner,size_t b,bool scoped_long=false):owner_(owner),geometry(b,scoped_long)')
edit('kimi_pair_state_r9006.hpp','if(r.phase||!n||n>1024)','if(r.phase||!n||n>(geometry.long_agent?size_t(512):geometry.context))')
edit('kimi_pair_state_r9006.hpp','size_t(past)+ids.size()>1024','size_t(past)+ids.size()>geometry.context')
edit('kimi_pair_state_r9006.hpp','size_t pages()const{return 4*(physical+1);}','size_t pages()const{return long_agent?128*physical+4:4*(physical+1);}')
edit('kimi_pair_state_r9006.hpp','size_t extra_vs_ordinary16_per_layer()const{return','size_t extra_vs_ordinary16_per_layer()const{if(long_agent)throw std::invalid_argument("r179 legacy extra-vs16 diagnostic excluded");return')
edit('kimi_pair_state_r9006.hpp','size_t kv_bytes_per_MLA_layer()const{return','size_t kv_bytes_per_MLA_layer()const{if(long_agent)throw std::invalid_argument("r179 use actual expanded KV allocation metadata");return')
# Typed controls retain exact old default; pool3 is scoped to32768 only.
edit('batch_payload_r9122c.hpp','size_t expected_pool,int64_t op){','size_t expected_pool,int64_t op,size_t max_context=1024){\n if(!((max_context==1024&&(expected_pool==17||expected_pool==33||expected_pool==65))||(max_context==32768&&expected_pool==3)))throw std::invalid_argument("r179 context/pool scope");')
edit('batch_payload_r9122c.hpp','(expected_pool!=17&&expected_pool!=33&&expected_pool!=65)||','')
edit('batch_payload_r9122c.hpp','x[1]>1024','x[1]>int64_t(max_context)')
edit('batch_payload_r9122c.hpp','x[1]+x[5]>1024','x[1]+x[5]>int64_t(max_context)')
# Capture declaration and implementation are compiled together in delta, with
# worker recompiled against this signature. Capturefalse remains the live path.
edit('batch_state_r9122.hpp','void begin(rt::Runtime&,uint64_t,const std::vector<Request>&,size_t);','void begin(rt::Runtime&,uint64_t,const std::vector<Request>&,size_t,size_t=1024);')
edit('batch_state_r9122.hpp','const std::vector<Request>&req,size_t total){','const std::vector<Request>&req,size_t total,size_t max_context){\n if(max_context!=1024&&max_context!=32768)throw std::invalid_argument("r179 begin context");')
edit('batch_state_r9122.hpp','if(half!=8&&half!=16&&half!=32)','if(!((max_context==1024&&(half==8||half==16||half==32))||(max_context==32768&&half==1)))')
edit('batch_state_r9122.hpp','size_t(x.past)+x.width>1024','size_t(x.past)+x.width>max_context')
# Main CAPI: old factories cannot activate long scope just by raw JSON. Only the
# new entry's temporary host-thread authorization can create that exact geometry.
edit('pair_probe.cpp','thread_local std::string error,serialized;','thread_local std::string error,serialized;\nthread_local bool authorized_long_create_r179=false;\nstruct LongCreateScopeR179{LongCreateScopeR179(){if(authorized_long_create_r179)throw std::invalid_argument("r179 nested create");authorized_long_create_r179=true;}~LongCreateScopeR179(){authorized_long_create_r179=false;}};')
edit('pair_probe.cpp','int stage=0,pp_size=4,tp_nodes=1,tp_node_rank=0,node=0;};','int stage=0,pp_size=4,tp_nodes=1,tp_node_rank=0,node=0;bool long_agent_r179=false;size_t max_total_tokens_r179=1024;};')
edit('pair_probe.cpp','kimi_batch_payload_r9122c::validate(words.data(),words.size(),kimi_native_lengths_.size(),int64_t(op));','kimi_batch_payload_r9122c::validate(words.data(),words.size(),kimi_native_lengths_.size(),int64_t(op),kimi_native_lengths_.size()==3?32768:1024);')
edit('pair_probe.cpp','auto cfg=nlohmann::json::parse(raw);auto&text=cfg.at("text_config");auto quant=text.at("quantization_config");','auto cfg=nlohmann::json::parse(raw);const bool long_scope_r179=cfg.value("infinilm_kimi_long_agent_r179",false);if(long_scope_r179!=authorized_long_create_r179||(long_scope_r179&&(max_rows!=2||tp_nodes!=2||pp_size!=2)))throw std::invalid_argument("r179 explicit factory scope");auto&text=cfg.at("text_config");text["infinilm_kimi_long_agent_r179"]=long_scope_r179;auto quant=text.at("quantization_config");')
edit('pair_probe.cpp','c->node=node;c->max_rows=max_rows;c->vocab=text.at("vocab_size");c->cache=std::make_unique<infinilm::cache::PagedKVCacheConfig>(4*max_rows+4,256,max_rows);','c->node=node;c->max_rows=max_rows;c->long_agent_r179=long_scope_r179;c->max_total_tokens_r179=long_scope_r179?32768:1024;c->pages_per_row_r4974=long_scope_r179?128:4;c->vocab=text.at("vocab_size");c->cache=std::make_unique<infinilm::cache::PagedKVCacheConfig>(long_scope_r179?128*max_rows+4:4*max_rows+4,256,max_rows);')
edit('pair_probe.cpp','(logical!=8&&logical!=16&&logical!=32)||!raw||!address','(logical!=8&&logical!=16&&logical!=32&&!(authorized_long_create_r179&&logical==1))||!raw||!address')
edit('pair_probe.cpp','kimi_pair_r9006::Geometry g(logical);auto*p=modern_kimi_engine_create_r2475','kimi_pair_r9006::Geometry g(logical,authorized_long_create_r179);auto*p=modern_kimi_engine_create_r2475')
edit('pair_probe.cpp','c->pair_r9006=std::make_unique<kimi_pair_r9006::State>(c,logical);','c->pair_r9006=std::make_unique<kimi_pair_r9006::State>(c,logical,c->long_agent_r179);',2)
edit('pair_probe.cpp','(c->max_rows!=16&&c->max_rows!=32&&c->max_rows!=64)||','(c->max_rows!=16&&c->max_rows!=32&&c->max_rows!=64&&!(c->long_agent_r179&&c->max_rows==2))||')
edit('pair_probe.cpp','auto next=std::make_unique<kimi_pair_r9006::State>(c,c->max_rows/2);','auto next=std::make_unique<kimi_pair_r9006::State>(c,c->max_rows/2,c->long_agent_r179);')
# Pair reset is another generation reset entry, not a context downgrade.
edit('pair_probe.cpp','if(c->pages_per_row_r4974!=4)throw','if(c->pages_per_row_r4974!=4&&!(c->long_agent_r179&&c->pages_per_row_r4974==128))throw')
edit('pair_probe.cpp','table(count*4);','table(count*c->pages_per_row_r4974);')
edit('pair_probe.cpp','total[i]>1024','size_t(total[i])>c->max_total_tokens_r179')
edit('pair_probe.cpp','int32_t page=1+(rows[i]-1)*4;for(size_t j=0;j<4;++j)table[i*4+j]','int32_t page=1+(rows[i]-1)*int32_t(c->pages_per_row_r4974);for(size_t j=0;j<c->pages_per_row_r4974;++j)table[i*c->pages_per_row_r4974+j]')
edit('pair_probe.cpp','{count,4},rt::DataType::kInt32);','{count,c->pages_per_row_r4974},rt::DataType::kInt32);')
p=N/'pair_probe.cpp';s=p.read_text();s+='''
// No environment switch, kernel change, alternate math or hidden context growth.
extern "C" void*modern_kimi_engine_create_long_agent_r179(const char*raw,int node,int tp_nodes,int pp_size,const char*address,int port){try{
 if(!raw||!address||tp_nodes!=2||pp_size!=2)throw std::invalid_argument("r179 TP16PP2 only");
 LongCreateScopeR179 authorization;auto cfg=nlohmann::json::parse(raw);const auto&t=cfg.at("text_config");if(t.at("num_attention_heads")!=96||t.at("qk_nope_head_dim")!=128||t.value("max_position_embeddings",size_t(0))<32768)throw std::invalid_argument("r179 actual K3 expanded KV/RoPE metadata required");cfg["infinilm_kimi_long_agent_r179"]=true;
 return modern_kimi_engine_create_batch_r9122c(cfg.dump().c_str(),node,tp_nodes,pp_size,address,port,1);
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" const char*modern_kimi_long_agent_capabilities_r179(void*raw){try{
 auto*c=static_cast<Context*>(raw);if(!c||!c->long_agent_r179||c->max_rows!=2||c->pages_per_row_r4974!=128||c->max_total_tokens_r179!=32768)throw std::invalid_argument("r179 long context required");
 serialized=nlohmann::json{{"scoped_long_agent",true},{"logical_capacity",1},{"physical_rows",2},{"state_pool_rows",3},{"num_blocks",260},{"page_size",256},{"pages_per_request",128},{"max_total_tokens",32768},{"initial_prefill_max_tokens",512},{"continuation_input_width_max",8},{"implemented_max_flatten_tokens",64},{"expanded_MLA_cache_shape",{2,260,256,6,192}},{"expanded_MLA_payload_bytes_per_layer",uint64_t(306708480)},{"math_changed",false},{"GPU_qualified",false}}.dump();return serialized.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}}
''';p.write_text(s)
# Cache helpers preserve legacy defaults; only exact long=true/cap2/pages260 may
# accept128 pages. Their main engine calls pass the immutable model scope flag.
edit('kimi_engine_input_r145.hpp','#include "cache/kv_cache.hpp"','#include "cache/kv_cache.hpp"\n#include "long_agent_geometry_r179.hpp"')
edit('kimi_engine_input_r145.hpp','cache_pages_r4974(const cache::CacheConfig*base,bool allow_compact=false)','cache_pages_r4974(const cache::CacheConfig*base,bool allow_compact=false,bool allow_long_r179=false)')
edit('kimi_engine_input_r145.hpp','const size_t rows=c->max_batch_size();','const size_t rows=c->max_batch_size();\n if(allow_long_r179){::kimi_long_agent_r179::validate_cache(rows,c->num_blocks(),c->block_size());return 128;}')
edit('kimi_engine_input_r145.hpp','inline void validate_cache(const cache::CacheConfig*base,bool allow_compact=false){(void)cache_pages_r4974(base,allow_compact);}','inline void validate_cache(const cache::CacheConfig*base,bool allow_compact=false,bool allow_long_r179=false){(void)cache_pages_r4974(base,allow_compact,allow_long_r179);}')
edit('kimi_engine_input_r145.hpp','inline size_t request_capacity(const cache::CacheConfig*base,bool allow_compact=false){validate_cache(base,allow_compact);','inline size_t request_capacity(const cache::CacheConfig*base,bool allow_compact=false,bool allow_long_r179=false){validate_cache(base,allow_compact,allow_long_r179);')
edit('kimi_engine_input_r145.hpp','size_t pages_per_row=4){','size_t pages_per_row=4,bool long_scope_r179=false){')
edit('kimi_engine_input_r145.hpp','if(pages_per_row!=4&&pages_per_row!=3)','if(pages_per_row!=4&&pages_per_row!=3&&!(long_scope_r179&&pages_per_row==128&&lengths.size()==3))')
edit('kimi_engine_input_r145.hpp','if(spec&&pages_per_row!=4)','if(spec&&pages_per_row!=4&&!(long_scope_r179&&pages_per_row==128&&proof->capacity==2))')
needle='  size_t first_page=1+(size_t(spec?proof->kv[i]:rows[i])-1)*pages_per_row;'
edit('kimi_engine_input_r145.hpp',needle,'  if(long_scope_r179)::kimi_long_agent_r179::validate_row_table(size_t(spec?proof->kv[i]:rows[i]),past[i],total[i],offsets[i],offsets[i+1],pages_per_row,table.data()+i*pages_per_row,slots.data()+offsets[i],positions.data()+offsets[i]);\n'+needle)
# Four engine helper call sites and both freeze/template contexts must agree.
p=N/'kimi_compact_engine_r4974.cpp';s=p.read_text();flag='model_config_->get_config_json().value("infinilm_kimi_compact_cache_r4974",false)';assert s.count(flag)==5;s=s.replace(flag,flag+',model_config_->get_config_json().value("infinilm_kimi_long_agent_r179",false)');needle='this,models::kimi_native::cache_pages_r4974(cache_config_.get(),'+flag+',model_config_->get_config_json().value("infinilm_kimi_long_agent_r179",false)))';assert needle in s;s=s.replace(needle,needle[:-1]+',model_config_->get_config_json().value("infinilm_kimi_long_agent_r179",false))');p.write_text(s)
# Allocation zero-fill, tensor shapes/dtype/lifetime are unchanged. Only KDA
# pool selection is decoupled for exact scoped rows2/pages260.
edit('kimi_compact_allocate_r4974.cpp','const size_t state_pool_size = compact_capability ? config->max_batch_size()+1 : std::max<size_t>(2, config->num_blocks() / 4);','const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);\n    if(long_scope_r179){::kimi_long_agent_r179::validate_cache(config->max_batch_size(),config->num_blocks(),config->block_size());if(tp_size!=16||rank_info.pp_size!=2)throw std::invalid_argument("r179 cache TP16PP2 scope");}\n    const size_t state_pool_size = (compact_capability||long_scope_r179) ? config->max_batch_size()+1 : std::max<size_t>(2, config->num_blocks() / 4);')
# Worker/producer typed header and dynamic maxpast are recompiled together.
edit('worker.cpp','(v[6]!=17&&v[6]!=33&&v[6]!=65)||','(v[6]!=17&&v[6]!=33&&v[6]!=65&&v[6]!=3)||')
needle='::kimi_batch_payload_r9122c::validate(v,payload->numel(),pc->max_batch_size()+1,v[2]);'
edit('worker.cpp',needle,'const bool long_scope_r179=model_config_->get_config_json().value("infinilm_kimi_long_agent_r179",false);const size_t max_context_r179=256*infinilm::models::kimi_native::cache_pages_r4974(pc,model_config_->get_config_json().value("infinilm_kimi_compact_cache_r4974",false),long_scope_r179);::kimi_batch_payload_r9122c::validate(v,payload->numel(),pc->max_batch_size()+1,v[2],max_context_r179);')
edit('worker.cpp','x[1]>1024','x[1]>int64_t(max_context_r179)')
edit('worker.cpp','::kimi_batch_r9122::begin(*runtime_,uint64_t(v[3]),req,total);','::kimi_batch_r9122::begin(*runtime_,uint64_t(v[3]),req,total,max_context_r179);')
# Preserve legacy helper function signatures as delegating overloads as well.
edit('batch_state_r9122.hpp','void begin(rt::Runtime&,uint64_t,const std::vector<Request>&,size_t,size_t=1024);','void begin(rt::Runtime&,uint64_t,const std::vector<Request>&,size_t);\nvoid begin(rt::Runtime&,uint64_t,const std::vector<Request>&,size_t,size_t);')
edit('batch_state_r9122.hpp','void begin(rt::Runtime&r,uint64_t epoch,const std::vector<Request>&req,size_t total,size_t max_context){','void begin(rt::Runtime&r,uint64_t epoch,const std::vector<Request>&req,size_t total){begin(r,epoch,req,total,1024); }\nvoid begin(rt::Runtime&r,uint64_t epoch,const std::vector<Request>&req,size_t total,size_t max_context){')
edit('kimi_engine_input_r145.hpp','bool allow_compact=false,bool allow_long_r179=false','bool allow_compact,bool allow_long_r179',3)
edit('kimi_engine_input_r145.hpp','inline void validate_cache(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179)','inline size_t cache_pages_r4974(const cache::CacheConfig*base,bool allow_compact=false){return cache_pages_r4974(base,allow_compact,false); }\ninline void validate_cache(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179)')
edit('kimi_engine_input_r145.hpp','inline size_t request_capacity(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179)','inline void validate_cache(const cache::CacheConfig*base,bool allow_compact=false){validate_cache(base,allow_compact,false); }\ninline size_t request_capacity(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179)')
edit('kimi_engine_input_r145.hpp','inline rt::Tensor freeze_cpu(','inline size_t request_capacity(const cache::CacheConfig*base,bool allow_compact=false){return request_capacity(base,allow_compact,false); }\ninline rt::Tensor freeze_cpu(')
(D/'transformation-contract-r179.json').write_text(json.dumps({'changes':log,'old_default_context':1024,'new_explicit_context':32768,'new_HIP_TUs':0,'old_math_provider_unchanged':True,'long_scope_logical':1,'long_scope_physical':2,'long_scope_state_pool':3,'long_scope_num_blocks':260,'GPU_GO':False},indent=2)+'\n')
print(json.dumps({'source_edits':len(log),'GPU':False}))
