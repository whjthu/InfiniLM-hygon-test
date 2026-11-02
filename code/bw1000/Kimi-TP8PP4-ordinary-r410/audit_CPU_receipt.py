"""Read-only stdlib audit of the original remote observation and frozen source."""
from pathlib import Path
import hashlib,json,sys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(observation,source,base_audit):
 r=json.loads(Path(observation).read_text(encoding='utf-8-sig'));q=r['receipt'];D=Path(source)
 assert hashlib.sha256(r['receipt_text'].encode()).hexdigest()==r['receipt_SHA256']
 assert json.loads(r['receipt_text'])==q and q['completed']and q['passed']
 assert q['actual_CPU']=='910407.819'and not q['GPU']and not q['Native_dlopen']and not q['Runtime_created']
 assert any(x.startswith('910407.819|Kimi410-TP8PP4-ordinary-CPU-v1|COMPLETED|0:0|')for x in r['accounting'])
 mf=D/'CPU_SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());assert sha(mf)==q['source_manifest_SHA256']
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 base=json.loads(json.loads(Path(base_audit).read_text(encoding='utf-8-sig'))['files']['/refactor/candidates/Kimi-append512-r247/CPU-build.json']['text'])
 assert base['native_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
 assert len(q['all_linked_object_SHA256'])==33 and len(q['unchanged_objects_SHA256'])==30
 assert all(base['all_linked_object_SHA256'][p]==h for p,h in q['unchanged_objects_SHA256'].items())
 names={'pair_probe':'pair_probe.cpp','allocate':'kimi_compact_allocate_r4974.cpp','moe':'kimi_moe_scale_exact_r201.cpp'}
 remote='/refactor/incoming/Kimi-TP8PP4-ordinary-r410/native/'
 for unit,name in names.items():
  row=q['HOST'][unit];md=row['all_new_MD'];assert row['fresh_MD_matches']
  assert row['source_SHA256']==sha(D/'native'/name)==md[remote+name]
  assert md[remote+'topology_r410.hpp']==sha(D/'native/topology_r410.hpp')
  if unit=='moe':assert md[remote+'scale_exact_scope_r201.hpp']==sha(D/'native/scale_exact_scope_r201.hpp')
  assert row['object_SHA256']==q['all_linked_object_SHA256']['/refactor/candidates/Kimi-TP8PP4-ordinary-r410-CPU-v1/'+unit+'.o']
 assert q['old_CAPI_retained']==130 and set(q['new_CAPI'])=={
  'modern_kimi_engine_create_ordinary_tp8_pp4_r410','modern_kimi_engine_forward_ordinary_r410','modern_kimi_ordinary_capabilities_r410'}
 assert q['actual_dependency_library_SHA256']==base['actual_dependency_library_SHA256']
 g=q['geometry_contract'];assert g['passed']and g['call_positives']==68 and g['I384_scale_alias_bounds']==3584
 return {'passed':True,'actual_CPU':q['actual_CPU'],'receipt_SHA256':r['receipt_SHA256'],
 'observation_SHA256':sha(observation),'source_manifest_SHA256':sha(mf),'source_members_checked':len(m['files_SHA256']),
 'Native_SHA256':q['native_SHA256'],'elapsed_s':q['elapsed_s'],'HOST_TUs':3,'GPU_TUs':0,
 'all_objects':33,'original_objects_unchanged':30,'MD_counts':{u:len(v['all_new_MD'])for u,v in q['HOST'].items()},
 'old_CAPI_retained':130,'added_CAPI':3,'exact_dynamic_providers':q['actual_dependency_library_SHA256'],
 'GPU':0,'Native_dlopen':False,'Runtime_created':False,'Model_qualified':False,
 'qualification':'Actual HOST/link and shared C++ geometry only; no physical device/weight/quality/performance qualification'}
if __name__=='__main__':print(json.dumps(audit(*sys.argv[1:]),indent=2))
