"""Exact retained CPU622/179/201/9183 source -> private five-HOST tree.

No SDK or GPU kernel changes. Existing class data members are retained.
"""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def once(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a))
 return s.replace(a,b,1)
def generate(audit,partition):
 q=json.loads(Path(audit).read_bytes());src=q['source_files']
 if isinstance(src,list):src={v['path'].replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014','/refactor'):v for v in src}
 else:src={k.replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014','/refactor'):v for k,v in src.items()}
 def get(path):
  v=src[path];s=v.get('rawtext',v.get('text'));assert sha(s.encode())==v['SHA256'];return s.replace('\r\n','\n')
 pairs='/refactor/incoming/Kimi-append512-r247/native/pair_probe.cpp'
 alloc='/refactor/incoming/Kimi-long-agent32K-r179/native/kimi_compact_allocate_r4974.cpp'
 scale='/refactor/incoming/Kimi-long-scale-exact-r201/'
 head='/refactor/incoming/Kimi-shard-only-Model-r9183/'
 originals={p:get(p)for p in [pairs,alloc,scale+'kimi_moe_scale_exact_r201.cpp',scale+'scale_exact_scope_r201.hpp',*[head+x for x in ['kimi_shard_causal_lm_r9183.hpp','kimi_shard_head_r9183.hpp','kimi_shard_model_r9183.cpp','kimi_shard_model_r9183.hpp','kimi_shard_policy_r9183.cpp','kimi_shard_stats_r9183.hpp']]]}
 s=originals[pairs];assert sha(s.encode())=='38dbbe54e90d4150ee510acac774254b9e9e1ff5da36d210e8e36eebf84240ce'
 s='#include "long_geometry_r453.hpp"\n'+s
 s=once(s,'thread_local bool authorized_long_create_r179=false;',
  'thread_local bool authorized_long_create_r179=false;\nthread_local bool authorized_long_geometry_r453=false;\nstruct LongGeometryScopeR453{LongGeometryScopeR453(){if(authorized_long_geometry_r453)throw std::invalid_argument("r453 nested factory");authorized_long_geometry_r453=true;}~LongGeometryScopeR453(){authorized_long_geometry_r453=false;}};')
 old='const bool long_scope_r179=cfg.value("infinilm_kimi_long_agent_r179",false);if(long_scope_r179!=authorized_long_create_r179||(long_scope_r179&&(max_rows!=2||tp_nodes!=2||pp_size!=2)))throw std::invalid_argument("r179 explicit factory scope");auto&text=cfg.at("text_config");text["infinilm_kimi_long_agent_r179"]=long_scope_r179;'
 new='const bool long_scope_r179=cfg.value("infinilm_kimi_long_agent_r179",false);const bool geometry_r453=cfg.value("infinilm_kimi_long_geometry_optin_r453",false);if(geometry_r453!=authorized_long_geometry_r453||long_scope_r179!=authorized_long_create_r179||(geometry_r453&&!long_scope_r179)||(long_scope_r179&&(max_rows!=2||!kimi_long_geometry_r453::topology(geometry_r453,8*tp_nodes,pp_size))))throw std::invalid_argument("r453 explicit factory scope");auto&text=cfg.at("text_config");text["infinilm_kimi_long_agent_r179"]=long_scope_r179;text["infinilm_kimi_long_geometry_optin_r453"]=geometry_r453;'
 s=once(s,old,new)
 # No Context field added: new contexts are unambiguously recognized from
 # existing admitted topology and long/C1 metadata; old pair objects untouched.
 context_end='thread_local const Context*pair_dispatch_r9006=nullptr;'
 s=once(s,context_end,'bool geometry_context_r453(const Context*c){return c&&c->long_agent_r179&&kimi_long_geometry_r453::topology(true,size_t(8*c->tp_nodes),size_t(c->pp_size));}\n'+context_end)
 for entry in ['modern_kimi_engine_forward_r2475','modern_kimi_engine_resize_cache_r2475','modern_kimi_engine_set_cache_pages_r4974','modern_kimi_long_agent_capabilities_r179']:
  start=s.index(' '+entry+'(') if ' '+entry+'(' in s else s.index('*'+entry+'(')
  at=s.index('auto*c=static_cast<Context*>(raw);',start)
  s=s[:at]+s[at:].replace('auto*c=static_cast<Context*>(raw);','auto*c=static_cast<Context*>(raw);if(geometry_context_r453(c))throw std::invalid_argument("r453 typed AR-only fixed-capacity entry required");',1)
 s+=r'''
// Explicit opt-in only. No batch/speculation/capture/Replay state constructed.
extern "C" void*modern_kimi_engine_create_long_geometry_r453(const char*raw,int node,int tp_nodes,int pp,const char*address,int port){try{
 const char*opt=std::getenv("INFINILM_KIMI_LONG_GEOMETRY_R453");
 if(!opt||std::string(opt)!="1"||!raw||!address||tp_nodes<1||pp<1||node<0)
  throw std::invalid_argument("r453 defaultOFF typed factory");
 const auto g=kimi_long_geometry_r453::geometry(true,size_t(8*tp_nodes),size_t(pp));
 if(size_t(node)>=g.tp_nodes*g.pp)throw std::invalid_argument("r453 node");
 pair_math0_r9006();LongGeometryScopeR453 geometry;LongCreateScopeR179 long_scope;
 auto cfg=nlohmann::json::parse(raw);const auto&t=cfg.at("text_config");
 if(t.at("num_attention_heads")!=96||t.at("qk_nope_head_dim")!=128||t.at("kv_lora_rank")!=512||t.at("qk_rope_head_dim")!=64||t.at("linear_attn_config").at("num_heads")!=96||t.at("linear_attn_config").at("head_dim")!=128||t.value("max_position_embeddings",size_t(0))<32768)
  throw std::invalid_argument("r453 checkpoint MLA/KDA/position geometry");
 cfg["infinilm_kimi_long_agent_r179"]=true;cfg["infinilm_kimi_long_geometry_optin_r453"]=true;
 return modern_kimi_engine_create_r2475(cfg.dump().c_str(),node,tp_nodes,pp,address,port,2);
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_kimi_engine_forward_long_geometry_r453(void*raw,int kind,size_t tokens,size_t count,int flags,const int32_t*rows,const int32_t*past,const int32_t*total,const int32_t*offsets,const int64_t*ids,const int64_t*positions,void*out_logits,void*out_ids,int*out_flags){try{
 auto*c=static_cast<Context*>(raw);
 if(!geometry_context_r453(c)||!rows||!past||!total||!offsets||c->pair_r9006||c->speculation_r3632||c->batch_enabled_r9122||c->batch_capture_r9122||c->state_only_r9114||c->max_rows!=2||c->pages_per_row_r4974!=128||c->max_total_tokens_r179!=32768)
  throw std::invalid_argument("r453 AR context and input metadata");
 kimi_long_geometry_r453::call_scope(kind,count,rows[0],past[0],tokens,flags);
 if(int64_t(total[0])!=int64_t(past[0])+int64_t(tokens)||offsets[0]!=0||offsets[1]!=int32_t(tokens))throw std::invalid_argument("r453 exact input interval");
 std::lock_guard<std::mutex>lock(c->mutex);
 if(!c->engine->append_ready_r247())throw std::invalid_argument("r453 weights must be Ready");
 return input_forward_locked_r247(c,tokens,count,flags,rows,past,total,offsets,ids,positions,out_logits,out_ids,out_flags);
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_long_geometry_capabilities_r453(void*raw){try{
 auto*c=static_cast<Context*>(raw);if(!geometry_context_r453(c)||c->pair_r9006||c->speculation_r3632||c->batch_enabled_r9122)throw std::invalid_argument("r453 typed AR context required");
 const auto g=kimi_long_geometry_r453::geometry(true,8*c->tp_nodes,c->pp_size);const auto r=kimi_long_geometry_r453::rank(true,g.tp,g.pp,c->node,0);
 serialized=nlohmann::json{{"version",1},{"TP",g.tp},{"PP",g.pp},{"C",1},{"physical_rows",2},{"state_rows",3},{"pages",260},{"page_size",256},{"context",32768},{"prompt_chunk",512},{"decode_width",1},{"MLA_local_heads",g.heads},{"KDA_local_heads",g.heads},{"local_MoE_I",g.expert_i},{"local_shared_I",g.shared_i},{"head_rows",g.head_rows},{"head_bytes",g.head_bytes},{"stage",r.stage},{"layer_begin",r.begin},{"layer_end",r.end},{"sampling_node",(g.pp-1)*g.tp_nodes},{"speculation",false},{"capture",false},{"replay",false},{"GPU_qualified",false}}.dump();return serialized.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}}
'''
 outputs={'pair_probe.cpp':s}
 a=originals[alloc]
 a='#include "long_geometry_r453.hpp"\n'+a
 a=once(a,'if(compact_capability)infinilm::models::kimi_native::validate_cache(config,true);','const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);\n    if(compact_capability||long_scope_r179)infinilm::models::kimi_native::validate_cache(config,compact_capability,long_scope_r179);')
 a=once(a,'const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);\n    if(long_scope_r179)','if(long_scope_r179)')
 a=once(a,'if(tp_size!=16||rank_info.pp_size!=2)throw std::invalid_argument("r179 cache TP16PP2 scope");','kimi_long_geometry_r453::require_topology(model_config->get_config_json().value("infinilm_kimi_long_geometry_optin_r453",false),tp_size,rank_info.pp_size);')
 outputs['kimi_compact_allocate_r4974.cpp']=a
 a=originals[scale+'scale_exact_scope_r201.hpp'];a='#include "long_geometry_r453.hpp"\n'+a
 a=once(a,'inline bool enabled(const Geometry&g){','inline bool enabled(const Geometry&g,bool optin=false){')
 a=once(a,'g.tp!=16||g.pp!=2||g.experts!=896||g.hidden!=3584||g.local_intermediate!=192','!kimi_long_geometry_r453::topology(optin,g.tp,g.pp)||g.experts!=896||g.hidden!=3584||g.local_intermediate!=3072/g.tp')
 a=once(a,'Scope(const Geometry&g,const void*runtime,const void*object):','Scope(const Geometry&g,const void*runtime,const void*object,bool optin=false):')
 a=once(a,'if(!enabled(g))return;','if(!enabled(g,optin))return;')
 outputs['scale_exact_scope_r201.hpp']=a
 a=originals[scale+'kimi_moe_scale_exact_r201.cpp'];a=once(a,'&runtime,this);','&runtime,this,model_config->get_config_json().value("infinilm_kimi_long_geometry_optin_r453",false));')
 outputs['kimi_moe_scale_exact_r201.cpp']=a
 # Reuse the already reviewed dynamic-row implementation, preserving class
 # member/signature layout. Extend its pure geometry to TP32; outer Model
 # construction enforces the new explicit JSON capability before child bind.
 dynamic=D.parent/'Kimi-TP8PP4-ordinary-r410/cpu-v4/native'
 for name in ['kimi_shard_causal_lm_r9183.hpp','kimi_shard_head_r9183.hpp','kimi_shard_model_r9183.cpp','kimi_shard_model_r9183.hpp','kimi_shard_policy_r9183.cpp','kimi_shard_stats_r9183.hpp']:
  a=(dynamic/name).read_text(encoding='utf8')
  if name=='kimi_shard_causal_lm_r9183.hpp':
   a='#include "long_geometry_r453.hpp"\n'+a
   a=once(a,'models::validate_runtime_construction(model_config, runtime, rank_info);','models::validate_runtime_construction(model_config, runtime, rank_info);\n        if(rank_info.tp_size!=16)kimi_long_geometry_r453::require_topology(model_config->get_config_json().value("infinilm_kimi_long_geometry_optin_r453",false),rank_info.tp_size,rank_info.pp_size);')
  if name=='kimi_shard_stats_r9183.hpp':
   a='#include "long_geometry_r453.hpp"\n'+a
   a=a.replace('std::array<std::array<uint64_t,3>,16>','std::array<std::array<uint64_t,3>,32>').replace('std::array<bool,16>','std::array<bool,32>')
   a=once(a,'if(TP!=8&&TP!=16)','if(TP!=8&&TP!=16&&TP!=32)')
   a=a.replace('Private r410 geometry admission','Private r453 geometry admission').replace('r410 head TP8 or TP16 only','r453 head TP8/16/32 only')
   a=once(a,'struct HeadGeometryR410 { size_t TP,rows,bytes;std::array<size_t,3>samples; };','using HeadGeometryR410=kimi_long_geometry_r453::HeadGeometry;')
   a=once(a,' if(TP!=8&&TP!=16&&TP!=32)throw std::invalid_argument("r453 head TP8/16/32 only");\n const size_t rows=163840/TP;return{TP,rows,rows*7168*2,{0,rows/2,rows-1}};',' return kimi_long_geometry_r453::head_geometry(TP);')
  outputs[name]=a
 n=D/'native';n.mkdir(exist_ok=False)
 for name,s in outputs.items():(n/name).write_bytes(s.encode())
 (n/'long_geometry_r453.hpp').write_bytes((D/'long_geometry_r453.hpp').read_bytes())
 p=json.loads(Path(partition).read_bytes());assert sha(p['rawtext'].encode())==p['SHA256']=='084ab999daa97d5b5b2cfc5e2db5e254616d6b136f37df3a282c7af258de249d'
 (D/'reference_pipeline_partition.hpp').write_bytes(p['rawtext'].encode())
 deriv={'schema':1,'original_raw_source_SHA256':{p:src[p]['SHA256']for p in originals},'original_LF_source_SHA256':{p:sha(s.encode())for p,s in originals.items()},'private_source_SHA256':{p:sha(s.encode())for p,s in outputs.items()},'Context_fields_added':0,'Geometry_fields_added':0,'Scope_fields_added':0,'head_class_fields_added':0,'changed_HOST_units':['pair_probe','allocate','moe','head','policy'],'new_GPU_units':0,'GPU_GO':False,'Model_GO':False,'forward_math_body_unchanged':True,'old_TP16_pair_batch_append_factory_unchanged':True,'head_dynamic_implementation_parent':'Kimi-TP8PP4-ordinary-r410/cpu-v4/native'}
 (D/'DERIVATION.json').write_text(json.dumps(deriv,indent=2)+'\n',encoding='utf8')
 return deriv
if __name__=='__main__':
 import sys
 print(json.dumps(generate(sys.argv[1],sys.argv[2]),indent=2))
