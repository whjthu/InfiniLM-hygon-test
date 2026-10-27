import pathlib,json,hashlib,difflib,shutil
o=pathlib.Path(__file__).resolve().parent;r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群');q=json.loads((o/'actual-bridge-binding-r125.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
copy=next(v for v in q['source_copies'].values()if v['SHA256']=='1d50a250f5b1f82687c879e3f320b97e3d4ebeefd97a0505375700d60b2c15dd');s=copy['text'];assert hashlib.sha256(s.encode()).hexdigest()==q['bridge_record']['source_SHA256']
extra='''
// r125 appends one reset/idle/lease-guarded policy API; Context/layout is unchanged.
#include "glm_m15_oproj_policy_r125.hpp"
extern "C" int modern_glm_model_m15_oproj_policy_r125(void*raw,int enabled){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("r125 policy context/value");auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("r125 policy requires reset/idle/empty rows");
  r.syncStream();model::glm_m15_oproj_policy_r125(r,enabled!=0);return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}
'''
(o/'modern_glm_bridge_oproj_r125.cpp').write_text(s+extra,encoding='utf8');(o/'bridge-policy-only-r125.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),(s+extra).splitlines(True),fromfile='actual-private3-1d50.cpp',tofile='private3-r125.cpp')),encoding='utf8')
qualified=r/'GLM-down-row2-Model-private-r9284b/native-CPU-qualified.json';z=json.loads(qualified.read_bytes());assert z['completed']and z['library_SHA256']=='c6506898f7d718e7257253b51bf720ff8fc84ed733d9b7306edd989308d37514';shutil.copyfile(qualified,o/'parent-native-qualified-r752.json')
parent=q['current_parent_receipt'];link=next(a for a in parent['commands']if '-shared'in a);assert link[link.index('-o')+1]==z['library'];pins=dict(z['reused_objects_SHA256']);pins.update(z['reused_qualified_GPU_object_SHA256']);pins.update({c['object']:c['object_SHA256']for c in z['compiled']});assert {x for x in link if x.endswith('.o')}==set(pins)
assert pins['/refactor/candidates/glm-M16-r8861/private-3.o']==q['bridge_record']['object_SHA256']==q['bridge_object_actual_SHA256'];assert pins['/refactor/candidates/glm-transient-layer-r5681/private-15.o']==q['runtime_record']['object_SHA256']
binding=dict(schema=1,parent_native_SHA256=z['library_SHA256'],parent_native_path=z['library'],qualified_receipt_SHA256=sha(qualified),historical_failed_receipt_SHA256=q['current_parent_receipt_SHA256'],historical_failed_receipt_status=parent['completed'],historical_failed_receipt_used_for='successful original compile/link commands only; qualified752receipt suppliesactualobjectpins',link_command=link,parent_objects_SHA256=pins,bridge_record=q['bridge_record'],bridge_compile_command=q['bridge_compile_command'],bridge_MD_SHA256=q['bridge_MD_SHA256'],runtime_record=q['runtime_record'],runtime_compile_command=q['runtime_compile_command'],helper776_SHA256='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8',old807_SHA256='1bd9ddbe75aad77a1ee59fa45784037a4aa00b7d2a44fc033a0a5cc41803268c')
(o/'parent-binding-r125.json').write_text(json.dumps(binding,indent=2),encoding='utf8');print(json.dumps({'bridge_closed':True,'parent_objects':len(pins),'bridge_new_SHA256':sha(o/'modern_glm_bridge_oproj_r125.cpp')}))
