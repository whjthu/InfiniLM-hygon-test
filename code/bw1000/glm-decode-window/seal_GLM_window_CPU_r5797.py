import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent;d=r/'glm-window-r5797-src';assert not d.exists();d.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((r/'r5796-GLM-window-build-inputs.json').read_text());prior=json.loads((r/'GLM-model-window-r5795-source-closure.json').read_text())
for t in inputs['targets']:
 n=pathlib.Path(t['record']['source']).name;assert t['record']['source_SHA256']==prior['sources'][n]['base_SHA256'];assert hashlib.sha256(t['source_text'].encode()).hexdigest()==t['record']['source_SHA256']
 s=(r/'glm-window-r5795-src'/n).read_text().replace('glm_model_window_r5795.hpp','glm_model_window_r5797.hpp')
 if n=='modern_glm_cached_model_bridge_r809.cpp':
  s+='''
extern "C" int modern_glm_window_prepare_r5797(void*raw){try{auto*c=static_cast<Context*>(raw);if(!c||!c->ready)throw std::invalid_argument("5797 prepare context");c->owner->assertOwnerThread();if(model::decode_leases_r1354::active(*c->owner)||!c->module->all_sequences_empty(*c->owner))throw std::invalid_argument("5797 prepare EMPTY/drained only");glm_window_r5795::prepare(*c->owner);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
'''
 (d/n).write_text(s,encoding='utf8',newline='\n')
h=(r/'glm_model_window_r5795.hpp').read_text().replace('void arm(rt::Runtime&,','void prepare(rt::Runtime&);\nvoid arm(rt::Runtime&,');(r/'glm_model_window_r5797.hpp').write_text(h,encoding='utf8',newline='\n')
s=(r/'glm_model_window_r5795.cpp').read_text().replace('glm_model_window_r5795.hpp','glm_model_window_r5797.hpp').replace('cudaEvent_t,64','cudaEvent_t,512').replace('Stamp,64','Stamp,512')
old='s.runtime=&r;s.owner=std::this_thread::get_id();try{for(auto&e:s.events)if(!e)ck(cudaEventCreate(&e));}catch(...){for(auto&e:s.events)if(e){cudaEventDestroy(e);e=nullptr;}throw;}'
assert s.count(old)==1;s=s.replace(old,'if(s.runtime!=&r||!s.events.back())throw std::logic_error("5797 prepare pool before decode");')
pos=s.index('void arm(');s=s[:pos]+'''void prepare(rt::Runtime&r){owner(r);if(s.active||s.pending||s.armed)throw std::logic_error("5797 prepare pending");s.runtime=&r;s.owner=std::this_thread::get_id();try{for(auto&e:s.events)if(!e)ck(cudaEventCreate(&e));}catch(...){for(auto&e:s.events)if(e){cudaEventDestroy(e);e=nullptr;}throw;}}
'''+s[pos:];(r/'glm_model_window_r5797.cpp').write_text(s,encoding='utf8',newline='\n')
names=['glm_model_window_r5797.cpp','glm_model_window_r5797.hpp','build_GLM_window_CPU_r5797.py','GLM_window_CPU_r5797_inside.sh','launch_GLM_window_CPU_r5797.py','r5796-GLM-window-build-inputs.json','resource_wave_local_r3097_outer.sh']
files={n:sha(r/n)for n in names};files.update({str(p.relative_to(r)).replace('\\','/'):sha(p)for p in d.iterdir()})
m={'sources_SHA256':files,'parent_DSO_SHA256':inputs['parent']['library_SHA256'],'parent_receipt_SHA256':inputs['parent_receipt_SHA256'],'input_receipt':'r5796-GLM-window-build-inputs.json','resource':'910406 n19 CPU4/4096M/5min gresnone','scope':'Three changed existing TUs plus one bounded queue-only observer; defaultOFF, same math/order/stream, no CPUdlopen/GPU/weights. Fixed512 events precreated at EMPTY phase; arm/read around original CAPI only.','targets_base':prior['sources'],'new_API':['modern_glm_window_prepare_r5797','modern_glm_window_arm_r5795','modern_glm_window_read_r5795'],'GPU_NOT_RUN':True}
(r/'GLM-window-CPU-manifest-r5797.json').write_text(json.dumps(m,indent=2),encoding='utf8');print('SEALED',len(files))
