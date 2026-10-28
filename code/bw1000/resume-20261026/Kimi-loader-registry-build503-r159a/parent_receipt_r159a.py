"""Strict schema for actual CPU503; no guessing links/objects/compile commands."""
PARENT_NATIVE='d559f6a60081b50aa43dcb5a442119c38d62cc757a5c3d35619c599dcef2dc2b'
PARENT_RECEIPT='c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd'
OLDWORKER='/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o'
OLDWORKER_SHA='42b891322186149b0018ddbf8044f2724080e41549ac95950e9bac8e6d176bb9'
PRESERVED={
 '/refactor/candidates/Kimi-MoE-phase-component-r155/moe.o':'79185c1d02b48ed64bd946e9d87be2ca8b925b77565a3f9e20be4f33832b1864',
 '/refactor/candidates/Kimi-MoE-phase-component-r155/provider.o':'7f6ccd4b97011c07b3fedd818ddded1d8be66888866ef74f2be41a612be7fbaa',
}
def normalize(parent):
 assert parent['completed'] is True and parent['CPU_only'] is True and parent['GPU'] is False and parent['dlopen'] is False
 assert parent['actual_step']=='910407.503' and parent['native_SHA256']==PARENT_NATIVE
 assert len(parent['unchanged_object_SHA256'])==31 and len(parent['new_object_SHA256'])==2
 objects=dict(parent['unchanged_object_SHA256'])
 for path,h in parent['new_object_SHA256'].items():
  assert path not in objects,'new object ambiguously replaces an unchanged entry';objects[path]=h
 links=[c for c in parent['commands']if '-shared'in c and '-o'in c and c[c.index('-o')+1]==parent['native']]
 assert len(links)==1,'need the actual Model link, not the later component adapter link'
 link=links[0][:];assert link[0]=='g++'
 linked=[x for x in link if x.endswith('.o')]
 assert len(linked)==len(set(linked))==33 and set(linked)==set(objects),'Model link/object closure mismatch'
 assert objects[OLDWORKER]==OLDWORKER_SHA
 for path,h in PRESERVED.items():assert objects[path]==h
 assert set(parent['new_MD'])=={'provider.d','adapter.d','moe.d'} and all(isinstance(v,dict)and v for v in parent['new_MD'].values())
 return {'objects':objects,'link':link,'new_MD':parent['new_MD'],'LD_LIBRARY_PATH':parent['LD_LIBRARY_PATH']}

def audit_symbols(oldraw,newraw,unique):
 old={x.split()[-1]:x.split()[-2]for x in oldraw.splitlines()if len(x.split())>=3}
 new={x.split()[-1]:x.split()[-2]for x in newraw.splitlines()if len(x.split())>=3}
 old_capi={name for name,kind in old.items()if kind=='T'and not name.startswith('_Z')}
 assert old_capi<=set(new),sorted(old_capi-set(new))
 required={'modern_kimi_engine_load_r2475','modern_kimi_engine_finish_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_loading_registry_work_r159','modern_kimi_loading_binding_counts_r9212',unique}
 assert required<=set(new),sorted(required-set(new))
 return {'required_C_API_and_symbols_preserved':True,'removed_exports_by_type':{name:kind for name,kind in old.items()if name not in new},'removed_weak_classification':'Weak inline/template exports are not a public C API gate; required unresolved references/version closure checked by ldd-r. Preserve names/types for review, including possible nlohmann vector<int> C1/C2.'}
