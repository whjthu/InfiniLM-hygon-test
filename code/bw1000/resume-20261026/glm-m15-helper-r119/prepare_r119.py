import pathlib,json,hashlib,difflib
r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群');o=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runtime=r/'glm_fp8_runtime_batch_r3068.cpp';assert sha(runtime)=='7f994d97fc3b2a50173cf318b24d08d0b714a2258e4c4e183aab1d2f237a9a44'
base=(r/'glm_fp8_cache_batch_r3039.cpp').read_text(encoding='utf8')
private=base.replace('glm_fp8_cache_r807.hpp','glm_fp8_cache_m15_r119.hpp').replace('_r807','_r119')
assert private.count('(m!=8&&m!=16&&m!=32&&m!=64)')==1
private=private.replace('(m!=8&&m!=16&&m!=32&&m!=64)','(m!=15)').replace('3039 bounded BF16 M8/16/32/64 geometry invalid','r119 private BF16 M15 geometry invalid')
(o/'glm_fp8_cache_m15_r119.cpp').write_text(private,encoding='utf8')
header=(r/'glm_fp8_cache_r807.hpp').read_text(encoding='utf8').replace('_r807','_r119')
(o/'glm_fp8_cache_m15_r119.hpp').write_text(header,encoding='utf8')
s=runtime.read_text(encoding='utf8');s=s.replace('#include "glm_fp8_runtime_r809.hpp"','#include "glm_fp8_runtime_r809.hpp"\n#include "glm_fp8_cache_m15_r119.hpp"')
s=s.replace('bool batch_enabled{false};uint64_t','bool batch_enabled{false};bool m15_enabled{false};void*m15_handle{};uint64_t',1)
s=s.replace('~Owner(){if(handle&&','~Owner(){if(m15_handle&&glm_fp8_cache_destroy_r119(m15_handle,runtime,stream))std::terminate();if(handle&&',1)
needle='bool fp8_cached_r809(rt::Runtime&r'
setter='''void glm_fp8_m15_policy_r119(rt::Runtime&r,bool enabled){
 r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("r119 policy during graph");
 owner(r).m15_enabled=enabled;
}
'''
assert s.count(needle)==1;s=s.replace(needle,setter+needle)
old='if(m!=8&&m!=16&&m!=32&&m!=64)return false;'
new='''auto m15_owner=store.owners.find(&r);
 const bool m15_admitted=m==15&&m15_owner!=store.owners.end()&&m15_owner->second->m15_enabled;
 if(m!=8&&m!=16&&m!=32&&m!=64&&!m15_admitted)return false;'''
assert s.count(old)==1;s=s.replace(old,new)
old=' auto&c=owner(r);\n if(glm_fp8_cache_run_r807('
new=''' auto&c=owner(r);
 if(m15_admitted){
  if(!c.m15_handle){c.m15_handle=glm_fp8_cache_create_r119(c.device,c.stream,&r);if(!c.m15_handle)throw std::runtime_error(glm_fp8_cache_error_r119());}
  if(glm_fp8_cache_run_r119(c.m15_handle,&r,r.stream(),1,1,m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_cache_error_r119());
  modern_glm_fp8_count_r780(r.device().index());return true;
 }
 if(glm_fp8_cache_run_r807('''
assert s.count(old)==1;s=s.replace(old,new)
old=' auto&p=*i->second;r->syncStream();if(glm_fp8_cache_clear_r807('
new=''' auto&p=*i->second;r->syncStream();
 if(p.m15_handle){
  if(glm_fp8_cache_clear_r119(p.m15_handle,r,p.stream))throw std::runtime_error(glm_fp8_cache_error_r119());
  uint64_t v[10]{};if(glm_fp8_cache_stats_r119(p.m15_handle,v,10)||v[0]!=v[1]||v[5]||v[6])throw std::logic_error("r119 private cache not fully released");
 }
 if(glm_fp8_cache_clear_r807('''
assert s.count(old)==1;s=s.replace(old,new)
(o/'glm_fp8_runtime_m15_r119.cpp').write_text(s,encoding='utf8')
patch=''.join(difflib.unified_diff(runtime.read_text(encoding='utf8').splitlines(True),s.splitlines(True),fromfile='actual-r3068-private15.cpp',tofile='private-r119-runtime.cpp'))
patch+=''.join(difflib.unified_diff(base.splitlines(True),private.splitlines(True),fromfile='known-r3039-cache-wrapper.cpp',tofile='private-r119-cache-wrapper.cpp'))
(o/'source-delta-r119.patch').write_text(patch,encoding='utf8')
# Bridge is a draft integration fragment, not compiled or installed.
bridge='''namespace infinilm::models::glm_moe_dsa {void glm_fp8_m15_policy_r119(rt::Runtime&,bool);}
extern "C" int modern_glm_model_m15_cache_policy_r119(void*raw,int enabled){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("r119 policy context/value");
  auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("r119 policy requires reset and idle owner");
  r.syncStream();model::glm_fp8_m15_policy_r119(r,enabled!=0);return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}
'''
(o/'glm_m15_bridge_r119.inc').write_text(bridge,encoding='utf8')
index={'schema':1,'scope':'CPU source draft; no compile/dlopen/upload/GPU; currentcacheDSO source binding UNKNOWN, privatewrapper from knownr3039 source, not an assertedcopyofcurrentDSO',
 'inputs':[dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size) for p in [runtime,r/'glm_fp8_cache_batch_r3039.cpp',r/'glm_fp8_cache_r807.hpp',r/'glm_fp8_batched_r776.cu',r/'glm_batch_cache_bridge_r3068.inc',r/'probe_glm_batch_cache_r3039.py',r/'audit_glm_batch_cache_r3042.py',r/'modern_glm_fp8_cache_r807.py',r/'glm_fp8_prefill_cache_r3124.cpp',r/'r5681-GLM-transient-CPU-status-a.json',r/'GLM-B32-Model-private-r9183/CPU-build.json']],
 'outputs':[dict(path=p.name,SHA256=sha(p),bytes=p.stat().st_size) for p in o.iterdir() if p.is_file() and p.suffix in ('.cpp','.hpp','.patch','.inc')]}
(o/'source-hash-index-r119.json').write_text(json.dumps(index,indent=2),encoding='utf8')
print(json.dumps(index['outputs']))
