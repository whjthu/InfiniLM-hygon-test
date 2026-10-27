import pathlib,hashlib,shutil,difflib
o=pathlib.Path(__file__).resolve().parent;r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群');base=r/'glm_fp8_runtime_batch_r3068.cpp';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(base)=='7f994d97fc3b2a50173cf318b24d08d0b714a2258e4c4e183aab1d2f237a9a44'
s=base.read_text();s=s.replace('#include "glm_fp8_runtime_r809.hpp"','#include "glm_fp8_runtime_r809.hpp"\n#include "glm_fp8_cache_m15_r119.hpp"\n#include "glm_m15_oproj_geometry_r125.hpp"')
s=s.replace('bool batch_enabled{false};uint64_t','bool batch_enabled{false};bool m15_enabled{false};void*m15_handle{};uint64_t m15_eligible{},m15_selected{},m15_off{},m15_rejected{},m15_errors{},m15_switches{};uint64_t',1)
s=s.replace('~Owner(){if(handle&&','~Owner(){if(m15_handle&&glm_fp8_cache_destroy_r119(m15_handle,runtime,stream))std::terminate();if(handle&&',1)
s=s.replace('std::array<std::array<uint64_t,10>,8>released{};','std::array<std::array<uint64_t,10>,8>released{};\n std::array<std::array<uint64_t,17>,8>released_m15{};',1)
needle='bool fp8_cached_r809(rt::Runtime&r'
new='''void glm_m15_oproj_policy_r125(rt::Runtime&r,bool enabled){
 r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("r125 policy recording");auto&c=owner(r);if(c.m15_enabled!=enabled){c.m15_enabled=enabled;++c.m15_switches;}
}
'''
assert s.count(needle)==1;s=s.replace(needle,new+needle)
old='if(m!=8&&m!=16&&m!=32&&m!=64)return false;'
new='''auto m15=store.owners.find(&r);const bool geometry15=m==15&&w->ndim()==2&&x->ndim()==2&&ws->ndim()==2&&glm_m15_oproj_r125::fits(m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co);
 bool select15=false;
 if(m==15&&m15!=store.owners.end()){
  auto&v=*m15->second;if(geometry15){++v.m15_eligible;select15=v.m15_enabled;if(!select15)++v.m15_off;}else ++v.m15_rejected;
 }
 if(m!=8&&m!=16&&m!=32&&m!=64&&!select15)return false;'''
assert s.count(old)==1;s=s.replace(old,new)
old=' auto&c=owner(r);\n if(glm_fp8_cache_run_r807('
new=''' auto&c=owner(r);
 if(select15){try{
  if(!c.m15_handle){c.m15_handle=glm_fp8_cache_create_r119(c.device,c.stream,&r);if(!c.m15_handle)throw std::runtime_error(glm_fp8_cache_error_r119());}
  if(glm_fp8_cache_run_r119(c.m15_handle,&r,r.stream(),1,1,m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_cache_error_r119());
  ++c.m15_selected;modern_glm_fp8_count_r780(r.device().index());return true;
 }catch(...){++c.m15_errors;throw;}}
 if(glm_fp8_cache_run_r807('''
assert s.count(old)==1;s=s.replace(old,new)
needle='extern "C" int modern_glm_fp8_runtime_clear_r809(rt::Runtime*r)'
stats='''extern "C" int modern_glm_m15_oproj_stats_r125(int device,uint64_t*out,size_t count){try{
 if(device<0||device>=8||!out||count!=17)throw std::invalid_argument("r125 stats bounds");const Owner*found=nullptr;
 for(const auto&[id,p]:store.owners)if(p->device==device){id->assertOwnerThread();if(found)throw std::logic_error("r125 ambiguous owner");found=p.get();}
 if(!found){std::copy(store.released_m15[device].begin(),store.released_m15[device].end(),out);return 0;}
 const auto&c=*found;const std::array<uint64_t,7>head{uint64_t(c.m15_enabled),c.m15_eligible,c.m15_selected,c.m15_off,c.m15_rejected,c.m15_errors,c.m15_switches};std::copy(head.begin(),head.end(),out);std::fill(out+7,out+17,0);
 if(c.m15_handle&&glm_fp8_cache_stats_r119(c.m15_handle,out+7,10))throw std::runtime_error(glm_fp8_cache_error_r119());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
'''
assert s.count(needle)==1;s=s.replace(needle,stats+needle)
old=' auto&p=*i->second;r->syncStream();if(glm_fp8_cache_clear_r807('
new=''' auto&p=*i->second;r->syncStream();
 if(p.m15_handle&&glm_fp8_cache_clear_r119(p.m15_handle,r,p.stream))throw std::runtime_error(glm_fp8_cache_error_r119());
 auto&released15=store.released_m15.at(p.device);if(modern_glm_m15_oproj_stats_r125(p.device,released15.data(),17))throw std::runtime_error(error);
 if(released15[7]!=released15[8]||released15[12]||released15[13])throw std::logic_error("r125 private cache not released");released15[0]=0;p.m15_enabled=false;
 if(glm_fp8_cache_clear_r807('''
assert s.count(old)==1;s=s.replace(old,new)
(o/'glm_fp8_runtime_oproj_r125.cpp').write_text(s,encoding='utf8');(o/'runtime-delta-r125.patch').write_text(''.join(difflib.unified_diff(base.read_text().splitlines(True),s.splitlines(True),fromfile='actual-r3068-private15.cpp',tofile='r125-oproj-only-private15.cpp')),encoding='utf8')
for n in ['glm_fp8_cache_m15_r119.cpp','glm_fp8_cache_m15_r119.hpp']:shutil.copyfile(o.parent/'glm-m15-component-r121'/n,o/n)
assert sha(o/'glm_fp8_cache_m15_r119.cpp')=='9a1eff3876a0271385be78be679e556e8ed2202dd582196ab0a495bb4a5fcb96'
print('runtime/cache source prepared; bridge pending exact source binding')
