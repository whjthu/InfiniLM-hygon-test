"""New immutable AR-only persistent-storage tree; old r453/8df untouched."""
from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent;P=D.parent/'Kimi-long-geometry-optin-r453'
H=lambda b:hashlib.sha256(b).hexdigest()
def once(s,a,b):assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b,1)
def lf(s):return s.replace('\r\n','\n').replace('\r','')
def generate(raw_sources):
 q=json.loads(Path(raw_sources).read_bytes());src=q['files'];get=lambda end:next((p,v)for p,v in src.items()if p.endswith(end)and v.get('exists'))
 originals={};out={}
 for p in (P/'native').iterdir():
  if p.is_file():out[p.name]=p.read_text(encoding='utf8');originals[str(p.relative_to(P))]=H(p.read_bytes())
 # Existing private helpers and class definitions are copied unchanged. Only
 # input admission helper gains an explicit mode, with four actual MD consumers.
 for end,name in [('kimi_engine_input_r145.hpp','kimi_engine_input_r145.hpp'),('kimi_compact_engine_r4974.cpp','kimi_compact_engine_r4974.cpp'),('worker.cpp','worker.cpp')]:
  p,v=get(end);assert H(v['rawtext'].encode())==v['SHA256'];out[name]=lf(v['rawtext']);originals[p]=v['SHA256']
 out['ar_storage_r456.hpp']=(D/'ar_storage_r456.hpp').read_text()
 s=out['kimi_engine_input_r145.hpp'];s='#include "ar_storage_r456.hpp"\n'+s
 s=once(s,'cache_pages_r4974(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179){','cache_pages_r4974(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179,bool ar_storage_r456=false){')
 s=once(s,'if(allow_long_r179){::kimi_long_agent_r179::validate_cache(rows,c->num_blocks(),c->block_size());return 128;}','if(ar_storage_r456){if(!allow_long_r179)throw std::invalid_argument("r456 long flag required");::kimi_ar_storage_r456::cache_geometry(rows,c->num_blocks(),c->block_size());return 128;}\n if(allow_long_r179){::kimi_long_agent_r179::validate_cache(rows,c->num_blocks(),c->block_size());return 128;}')
 s=once(s,'validate_cache(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179){(void)cache_pages_r4974(base,allow_compact,allow_long_r179);}','validate_cache(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179,bool ar_storage_r456=false){(void)cache_pages_r4974(base,allow_compact,allow_long_r179,ar_storage_r456);}')
 s=once(s,'request_capacity(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179){validate_cache(base,allow_compact,allow_long_r179);','request_capacity(const cache::CacheConfig*base,bool allow_compact,bool allow_long_r179,bool ar_storage_r456=false){validate_cache(base,allow_compact,allow_long_r179,ar_storage_r456);')
 s=once(s,'size_t pages_per_row=4,bool long_scope_r179=false){','size_t pages_per_row=4,bool long_scope_r179=false,bool ar_storage_r456=false){')
 s=once(s,'pages_per_row==128&&lengths.size()==3','pages_per_row==128&&lengths.size()==(ar_storage_r456?2u:3u)')
 s=once(s,'const bool spec=proof&&proof->owner==owner;','const bool spec=proof&&proof->owner==owner;\n if(ar_storage_r456){if(!long_scope_r179)throw std::invalid_argument("r456 long flag required");::kimi_ar_storage_r456::state_geometry(lengths.size(),spec);}')
 s=once(s,'for(size_t i=0;i<n;++i){','for(size_t i=0;i<n;++i){\n  if(ar_storage_r456)::kimi_ar_storage_r456::request(size_t(rows[i]),size_t(writes[i]),lengths.size(),pages_per_row,spec);')
 out['kimi_engine_input_r145.hpp']=s
 # Engine still uses its existing vector/counter/lifetime layout. The new mode
 # is read from the model's immutable, factory-authorized JSON at each entry.
 flag='model_config_->get_config_json().value("infinilm_kimi_ar_storage_optin_r456",false)'
 s=out['kimi_compact_engine_r4974.cpp'];old='model_config_->get_config_json().value("infinilm_kimi_long_agent_r179",false)'
 assert s.count(old)==6,s.count(old)
 # Each occurrence is the final argument of validate_cache/request_capacity/
 # cache_pages, or the freeze_input long flag. Append the one explicit mode.
 s=s.replace(old+')',old+','+flag+')')
 out['kimi_compact_engine_r4974.cpp']=s
 # Batch control is unreachable for new AR contexts; leave all original batch
 # guards intact. This real includer is nevertheless rebuilt against the new
 # inline input helper, so no stale header implementation is retained.
 # MoE exact W2 ticket is distinct and only arms for new AR TP32 constructor.
 s='#include "ar_storage_r456.hpp"\n'+out['kimi_moe_scale_exact_r201.cpp']
 old='        register_mxfp4_experts(runtime);'
 new='        kimi_ar_storage_r456::Scope storage_scope_r456(model_config->get_config_json().value("infinilm_kimi_ar_storage_optin_r456",false),model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false),tp_size_,rank_info.pp_size,local_intermediate_size_,&runtime,this);\n'+old
 s=once(s,old,new)
 s=once(s,'auto w2 = infinilm::runtime::Tensor::empty(\n        runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 2},\n        infinilm::runtime::DataType::kUInt8);','auto w2 = kimi_ar_storage_r456::selected(&runtime,this)\n        ? infinilm::runtime::Tensor::empty_parameter(runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 2}, infinilm::runtime::DataType::kUInt8)\n        : infinilm::runtime::Tensor::empty(runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 2}, infinilm::runtime::DataType::kUInt8);')
 out['kimi_moe_scale_exact_r201.cpp']=s
 # Private exact cache helper retains the exact original zero/failure-drain
 # body and guards. It neither replaces nor interposes the shared cache API.
 cp,cv=get('cache/kv_cache.cpp');cache=lf(cv['rawtext']);start=cache.index('infinilm::runtime::Tensor create_layer_kv_cache(',cache.index('namespace PagedKVCache'))
 cache=cache[start:cache.index('\n}; // namespace PagedKVCache',start)]
 cache=once(cache,'create_layer_kv_cache(','exact_mla_cache_r456(')
 # Original rank_kv_heads checks are preserved under stronger exact geometry.
 cache=once(cache,'const size_t num_rank_k_heads = rank_kv_heads(k_dim, v_dim, num_k_heads, num_v_heads, static_cast<size_t>(rank.tp_size));','kimi_ar_storage_r456::cache_geometry(config.max_batch_size(),config.num_blocks(),config.block_size());\n    if(rank.tp_size!=32||rank.pp_size!=1||k_dim!=192||v_dim!=192||num_k_heads!=96||num_v_heads!=96||dtype!=infinilm::runtime::DataType::kBFloat16)throw std::invalid_argument("r456 exact MLA geometry");\n    const size_t num_rank_k_heads = 3;')
 cache=once(cache,'infinilm::runtime::Tensor::empty(runtime, kv_shape, dtype)','infinilm::runtime::Tensor::empty_parameter(runtime, kv_shape, dtype)')
 cache=once(cache,'    validate_cache_runtime(runtime, rank);','    infinilm::cache::validate_cache_runtime(runtime, rank);')
 cache=once(cache,'const PagedKVCacheConfig &config)','const infinilm::cache::PagedKVCacheConfig &config)')
 cache=once(cache,'const engine::distributed::RankInfo &rank','const infinilm::engine::distributed::RankInfo &rank')
 out['exact_mla_cache_r456.hpp']='#pragma once\n#include "ar_storage_r456.hpp"\n#include "cache/kv_cache.hpp"\n#include "execution/submit.hpp"\n#include "utils.hpp"\nnamespace kimi_ar_storage_r456 {\ninline '+cache+'\n}\n'
 originals[cp]=cv['SHA256']
 s='#include "exact_mla_cache_r456.hpp"\n'+out['kimi_compact_allocate_r4974.cpp']
 s=once(s,'const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);','const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);\n    const bool ar_storage_r456=model_config->get_config_json().value("infinilm_kimi_ar_storage_optin_r456",false);')
 s=once(s,'validate_cache(config,compact_capability,long_scope_r179);','validate_cache(config,compact_capability,long_scope_r179,ar_storage_r456);')
 s=once(s,'::kimi_long_agent_r179::validate_cache(config->max_batch_size(),config->num_blocks(),config->block_size());','if(ar_storage_r456){::kimi_ar_storage_r456::topology(true,long_scope_r179,tp_size,rank_info.pp_size,96);::kimi_ar_storage_r456::cache_geometry(config->max_batch_size(),config->num_blocks(),config->block_size());}else ::kimi_long_agent_r179::validate_cache(config->max_batch_size(),config->num_blocks(),config->block_size());')
 old='kv[i] = cache::PagedKVCache::create_layer_kv_cache(\n                    runtime, rank_info,\n                    head_dim, head_dim, num_heads, num_heads,\n                    model_config->get_kv_cache_dtype(), *config);'
 new='kv[i] = ar_storage_r456\n                    ? kimi_ar_storage_r456::exact_mla_cache_r456(runtime,rank_info,head_dim,head_dim,num_heads,num_heads,model_config->get_kv_cache_dtype(),*config)\n                    : cache::PagedKVCache::create_layer_kv_cache(runtime,rank_info,head_dim,head_dim,num_heads,num_heads,model_config->get_kv_cache_dtype(),*config);'
 s=once(s,old,new);out['kimi_compact_allocate_r4974.cpp']=s
 # Explicit new factory, using existing Context fields and original forward.
 s='#include "ar_storage_r456.hpp"\n'+out['pair_probe.cpp']
 s=once(s,'thread_local bool authorized_long_geometry_r453=false;','thread_local bool authorized_long_geometry_r453=false;\nthread_local bool authorized_ar_storage_r456=false;\nstruct ArStorageCreateScopeR456{ArStorageCreateScopeR456(){if(authorized_ar_storage_r456)throw std::invalid_argument("r456 nested create");authorized_ar_storage_r456=true;}~ArStorageCreateScopeR456(){authorized_ar_storage_r456=false;}};')
 s=once(s,'const bool geometry_r453=cfg.value("infinilm_kimi_long_geometry_optin_r453",false);','const bool geometry_r453=cfg.value("infinilm_kimi_long_geometry_optin_r453",false);const bool ar_storage_r456=cfg.value("infinilm_kimi_ar_storage_optin_r456",false);')
 s=once(s,'if(geometry_r453!=authorized_long_geometry_r453||','if(ar_storage_r456!=authorized_ar_storage_r456||(ar_storage_r456&&(!geometry_r453||tp_nodes!=4||pp_size!=1))||geometry_r453!=authorized_long_geometry_r453||')
 s=once(s,'(max_rows!=2||!kimi_long_geometry_r453::topology','(max_rows!=(ar_storage_r456?1u:2u)||!kimi_long_geometry_r453::topology')
 s=once(s,'text["infinilm_kimi_long_geometry_optin_r453"]=geometry_r453;','text["infinilm_kimi_long_geometry_optin_r453"]=geometry_r453;text["infinilm_kimi_ar_storage_optin_r456"]=ar_storage_r456;')
 start=s.index('extern "C" const char*modern_kimi_long_geometry_capabilities_r453(');at=s.index('if(!geometry_context_r453(c)||',start);s=s[:at]+s[at:].replace('if(!geometry_context_r453(c)||','if(!geometry_context_r453(c)||c->max_rows!=2||',1)
 s+=r'''
extern "C" void*modern_kimi_engine_create_ar_storage_r456(const char*raw,int node,const char*address,int port){try{
 const char*opt=std::getenv("INFINILM_KIMI_AR_STORAGE_R456");const char*geo=std::getenv("INFINILM_KIMI_LONG_GEOMETRY_R453");
 if(!opt||std::string(opt)!="1"||!geo||std::string(geo)!="1"||!raw||!address||node<0||node>=4)throw std::invalid_argument("r456 defaultOFF AR TP32PP1 factory");
 pair_math0_r9006();ArStorageCreateScopeR456 ar;LongGeometryScopeR453 geometry;LongCreateScopeR179 long_scope;
 auto cfg=nlohmann::json::parse(raw);const auto&t=cfg.at("text_config");
 if(t.at("num_attention_heads")!=96||t.at("qk_nope_head_dim")!=128||t.at("kv_lora_rank")!=512||t.at("qk_rope_head_dim")!=64||t.at("linear_attn_config").at("num_heads")!=96||t.at("linear_attn_config").at("head_dim")!=128||t.value("max_position_embeddings",size_t(0))<32768)throw std::invalid_argument("r456 original checkpoint geometry");
 cfg["infinilm_kimi_long_agent_r179"]=true;cfg["infinilm_kimi_long_geometry_optin_r453"]=true;cfg["infinilm_kimi_ar_storage_optin_r456"]=true;
 return modern_kimi_engine_create_r2475(cfg.dump().c_str(),node,4,1,address,port,1);
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_kimi_engine_forward_ar_storage_r456(void*raw,int kind,size_t tokens,size_t count,int flags,const int32_t*rows,const int32_t*past,const int32_t*total,const int32_t*offsets,const int64_t*ids,const int64_t*positions,void*out_logits,void*out_ids,int*out_flags){try{
 auto*c=static_cast<Context*>(raw);if(!geometry_context_r453(c)||c->tp_nodes!=4||c->pp_size!=1||c->max_rows!=1||c->pages_per_row_r4974!=128||c->max_total_tokens_r179!=32768||c->pair_r9006||c->speculation_r3632||c->batch_enabled_r9122||c->batch_capture_r9122||c->state_only_r9114||!rows||!past||!total||!offsets)throw std::invalid_argument("r456 typed AR storage context");
 kimi_long_geometry_r453::call_scope(kind,count,rows[0],past[0],tokens,flags);kimi_ar_storage_r456::request(rows[0],rows[0],2,128,false);
 if(int64_t(total[0])!=int64_t(past[0])+int64_t(tokens)||offsets[0]!=0||offsets[1]!=int32_t(tokens))throw std::invalid_argument("r456 exact input interval");
 std::lock_guard<std::mutex>lock(c->mutex);if(!c->engine->append_ready_r247())throw std::invalid_argument("r456 weights must be Ready");
 return input_forward_locked_r247(c,tokens,count,flags,rows,past,total,offsets,ids,positions,out_logits,out_ids,out_flags);
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_ar_storage_capabilities_r456(void*raw){try{
 auto*c=static_cast<Context*>(raw);if(!geometry_context_r453(c)||c->tp_nodes!=4||c->pp_size!=1||c->max_rows!=1||c->pair_r9006||c->speculation_r3632)throw std::invalid_argument("r456 typed AR context required");
 serialized=nlohmann::json{{"version",1},{"TP",32},{"PP",1},{"logical_capacity",1},{"physical_rows",1},{"state_pool_rows",2},{"context",32768},{"page_size",256},{"pages_per_request",128},{"num_blocks",132},{"MLA_shape",{2,132,256,3,192}},{"W2_exact_parameter",true},{"MLA_exact_parameter",true},{"capture",false},{"replay",false},{"GPU_qualified",false}}.dump();return serialized.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}}
'''
 out['pair_probe.cpp']=s
 N=D/'native';N.mkdir(exist_ok=False)
 for n,s in out.items():(N/n).write_bytes(s.encode())
 receipt=dict(schema=1,parent_source_manifest_SHA256=H((P/'CPU_SOURCE_MANIFEST.json').read_bytes()),parent_Native_SHA256='fc3def9a539f3e9c9e1fc4c4ef2c2f0bbee03ddbe32300c689a8113310068056',originals_SHA256=originals,private_SHA256={n:H(s.encode())for n,s in out.items()},new_production_HOST_TUs=['pair_probe','allocate','moe','worker','infer_engine'],new_GPU_TUs=0,public_class_fields_changed=False,old_8df_r453_sources_changed=False,GPU_GO=False,Model_GO=False)
 (D/'DERIVATION.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode());return receipt
if __name__=='__main__':print(json.dumps(generate(sys.argv[1]),indent=2))
