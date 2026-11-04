"""Standalone stdlib audit of CPU966. No candidate/SDK module executes."""
from pathlib import Path
import base64,gzip,hashlib,json,posixpath,shlex,sys
H=lambda b:hashlib.sha256(b).hexdigest()
ARCHIVE='38e33a3e04592b46448be83cc63b0da06110959be379f8b56258b721d6ed3a02'
SOURCE='e9ee27a8574c712d48709e7729a47413f779d3af326d95c9de125361cac0f94c'
PRODUCTION='919b888d4d8c5bb994d683509298d9116d79544cb72f4973a387e44eaf79c16d'
NATIVE='0d4eb39aa2f566876b4defa90747188839724c99ac4db02e8735749956b35eaf'
B='/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v2/'
O='/refactor/incoming/Kimi-AR-persistent-storage-optin-r456/';D=O+'cpu-repair-v2/';N=O+'native/';P='/refactor/incoming/Kimi-long-geometry-optin-r453/'
def symbols(s):return{w[-1]:w[-2]for line in s.splitlines()if len(w:=line.split())>=2}
def body(s,marker):
 i=s.index(marker);i=s.index('{',i);depth=1;j=i+1
 while depth:depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[i:j]
def lf(s):return s.replace('\r\n','\n').replace('\r','')
def audit(path):
 t=json.loads(Path(path).read_bytes());z=base64.b64decode(t['gzip_base64']);assert H(z)==t['archive_SHA256']==ARCHIVE;raw=gzip.decompress(z);assert len(raw)==t['uncompressed_bytes']and H(raw)==t['uncompressed_SHA256'];a=json.loads(raw)
 assert a['CPU']=='910407.966'and a['source_manifest_SHA256']==SOURCE and a['production_manifest_SHA256']==PRODUCTION
 assert not a['Native_execution']and not a['SDK_execution']and a['GPU']==a['Model']==0
 fs=a['files'];assert len(fs)==t['rawfiles']==113
 for p,v in fs.items():b=v['rawtext'].encode();assert len(b)==v['bytes']and H(b)==v['SHA256'],p
 get=lambda p:fs[p]['rawtext'];js=lambda p:json.loads(get(p))
 q=js(B+'CPU-result.json');assert q['completed']and q['passed']and q['actual_CPU']=='910407.966'and q['source_manifest_SHA256']==SOURCE and q['native_SHA256']==NATIVE
 assert not q['Native_dlopen']and not q['Runtime_created']and q['GPU']==q['Model']==0
 for root,h in [(D,SOURCE),(O,PRODUCTION),(P,'f37bef41ef98b216c1c3b5a1a4a3015a0a161a64bd5d5842f96be5a96e0827cd')]:
  assert fs[root+'CPU_SOURCE_MANIFEST.json']['SHA256']==h
  for p,v in js(root+'CPU_SOURCE_MANIFEST.json')['files_SHA256'].items():assert fs[root+p]['SHA256']==v,(root,p)
 oldfail=js('/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v1/CPU-result.json');assert oldfail['actual_CPU']=='910407.964'and not oldfail['passed']and oldfail['commands']==[]and 'AssertionError'in oldfail['error']
 assert len(q['HOST'])==5 and len(q['all_linked_object_SHA256'])==33 and len(q['unchanged_objects_SHA256'])==28 and len(a['artifacts'])==35
 for p,h in q['all_linked_object_SHA256'].items():assert a['artifacts'][p]['SHA256']==h,p
 assert a['artifacts'][q['native']]['SHA256']==NATIVE
 base=js('/refactor/candidates/Kimi-long-geometry-optin-r453-CPU-v1/CPU-result.json');assert fs['/refactor/candidates/Kimi-long-geometry-optin-r453-CPU-v1/CPU-result.json']['SHA256']=='d567329a41cc5c33e6f6bf39aef2b89209b3ef5e1756bebad4116538b2330c96'
 for p,h in q['unchanged_objects_SHA256'].items():assert base['all_linked_object_SHA256'][p]==h
 assert {k:v['SHA256']for k,v in q['actual_dependency_library_SHA256'].items()}=={k:v['SHA256']for k,v in base['actual_dependency_library_SHA256'].items()}
 for label,v in q['logs'].items():assert v['rc']==0 and fs[B+label+'.log']['SHA256']==v['SHA256'],label
 prod=[c for c in q['commands']if '-c'in c];assert len(prod)==5 and {Path(c[c.index('-c')+1]).name for c in prod}=={'pair_probe.cpp','kimi_compact_allocate_r4974.cpp','kimi_moe_scale_exact_r201.cpp','worker.cpp','kimi_compact_engine_r4974.cpp'}
 old179=js('/refactor/candidates/Kimi-long-agent32K-r179/CPU-build.json');mds={**{k:v['all_new_MD']for k,v in base['HOST'].items()},**{k:v for k,v in old179['all_new_MD'].items()if k in ['worker','infer_engine','delta','trace']}}
 r179='/refactor/incoming/Kimi-long-agent32K-r179/native/';input_origins={'pair_probe':'/refactor/incoming/Kimi-append512-r247/native/kimi_engine_input_r145.hpp',**{k:r179+'kimi_engine_input_r145.hpp'for k in ['allocate','worker','infer_engine']}}
 assert q['exact_historical_input_header_origins']==input_origins
 for unit,md in mds.items():assert {p:h for p,h in md.items()if Path(p).name=='kimi_engine_input_r145.hpp'}==({input_origins[unit]:'8efaddde3995d25b6ef0a0985338b4bd8f7a247ca2e47a62f51d15c160c11168'}if unit in input_origins else{})
 known={posixpath.normpath(p):h for md in mds.values()for p,h in md.items()};source_map={Path(p).name:P+p for p in js(P+'CPU_SOURCE_MANIFEST.json')['files_SHA256']if p.startswith('native/')}
 source_map.update({p:r179+p for p in ['worker.cpp','kimi_compact_engine_r4974.cpp']});extras={N+'ar_storage_r456.hpp',N+'exact_mla_cache_r456.hpp'}
 adds={}
 for unit,v in q['HOST'].items():
  dep=lambda path:set(shlex.split(get(path).replace('\\\n',' ').split(':',1)[1]));actual=dep(B+unit+'.d');assert actual==dep(B+unit+'.fresh.d')==set(v['all_new_MD'])
  required={N+'ar_storage_r456.hpp'}
  if unit in input_origins:required.add(N+'kimi_engine_input_r145.hpp')
  if unit=='allocate':required.add(N+'exact_mla_cache_r456.hpp')
  assert required<=actual,(unit,'actual private header absent')
  for p,h in v['all_new_MD'].items():
   if p.startswith(N):assert fs[p]['SHA256']==h,p
  parentmd=mds[unit];oldcanon={posixpath.normpath(p):h for p,h in parentmd.items()};normalized={}
  for p,h in v['all_new_MD'].items():
   if p in extras:continue
   if p.startswith(N):origin=input_origins[unit]if Path(p).name=='kimi_engine_input_r145.hpp'else source_map[Path(p).name]
   else:origin=p
   normalized[posixpath.normpath(origin)]=h
  assert set(oldcanon)<=set(normalized),(unit,'old dependency omitted')
  changed=set(input_origins.values())|{P+'native/'+p for p in ['pair_probe.cpp','kimi_compact_allocate_r4974.cpp','kimi_moe_scale_exact_r201.cpp']}|{r179+p for p in ['worker.cpp','kimi_compact_engine_r4974.cpp']}
  for p,h in normalized.items():
   if p not in changed:assert h==oldcanon.get(p,known.get(p)),(unit,'unchanged dependency SHA',p)
  added=set(normalized)-set(oldcanon);assert added<=set(known)
  assert {p:known[p]for p in added}==v['added_external_dependencies_from_successful_parent_MD']
  adds[unit]=len(added)
 old=symbols(get(B+'old-defined.log'));new=symbols(get(B+'new-defined.log'));strong={p for p,t in old.items()if t.upper()in {'T','D','B','R','G','S','A'}};assert strong<=set(new)and len(strong)==389
 oldc={p for p in old if p.startswith('modern_')};newc={p for p in new if p.startswith('modern_')};added={'modern_kimi_engine_create_ar_storage_r456','modern_kimi_engine_forward_ar_storage_r456','modern_kimi_ar_storage_capabilities_r456'};assert oldc<=newc and newc-oldc==added and len(oldc)==133
 link=next(c for c in q['commands']if '-shared'in c);assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link and {p for p in link if p.endswith('.o')}==set(q['all_linked_object_SHA256'])
 assert 'not found'not in get(B+'ldd-r.log')and 'undefined symbol:'not in get(B+'ldd-r.log')
 old=get(P+'native/pair_probe.cpp');new=get(N+'pair_probe.cpp');same={}
 for marker in ['struct Context','int input_forward_locked_r247(','modern_kimi_engine_reset_r2475(','modern_kimi_engine_destroy_r2475(','modern_kimi_engine_create_long_agent_r179(','modern_kimi_pair_append_true_input_r247(']:assert body(old,marker)==body(new,marker);same[marker]=H(body(old,marker).encode())
 assert 'ar_storage_r456!=authorized_ar_storage_r456'in new and 'INFINILM_KIMI_AR_STORAGE_R456'in new
 for name in ['kimi_shard_causal_lm_r9183.hpp','kimi_shard_head_r9183.hpp','kimi_shard_stats_r9183.hpp','kimi_shard_model_r9183.cpp','kimi_shard_model_r9183.hpp','kimi_shard_policy_r9183.cpp','long_geometry_r453.hpp','scale_exact_scope_r201.hpp']:assert get(N+name)==get(P+'native/'+name)
 assert lf(get(N+'worker.cpp'))==lf(get(r179+'worker.cpp'))
 oldmoe=get(P+'native/kimi_moe_scale_exact_r201.cpp');newmoe=get(N+'kimi_moe_scale_exact_r201.cpp')
 for marker in ['KimiK3MoE::forward(','KimiK3MLP::forward(','auto register_packed = [&]']:
  assert body(oldmoe,marker)==body(newmoe,marker);same[marker]=H(body(oldmoe,marker).encode())
 assert '? infinilm::runtime::Tensor::empty_parameter(runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 2}, infinilm::runtime::DataType::kUInt8)'in newmoe
 cache=lf(get('/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/cache/kv_cache.cpp'));cache=cache[cache.index('namespace PagedKVCache'):];exact=get(N+'exact_mla_cache_r456.hpp');assert body(cache,'    try {')==body(exact,'    try {');same['MLA_zero_and_failuredrain']=H(body(exact,'    try {').encode())
 old=lf(get(r179+'kimi_compact_engine_r4974.cpp'));new=get(N+'kimi_compact_engine_r4974.cpp');flag=',model_config_->get_config_json().value("infinilm_kimi_ar_storage_optin_r456",false)';assert new.count(flag)==6 and new.replace(flag,'')==old
 src=get(N+'kimi_engine_input_r145.hpp');i=src.index('inline rt::Tensor freeze_cpu(');j=src.rindex('\n}');actual_body=src[i:j];identity=q['actual_template_recording_leaf_identity'];assert H(actual_body.encode())==identity['actual_template_body_SHA256']=='3cfb65c8f9746ea181a22b428a24ce64073b3f487b9f9d0e68eccdabc71e5867'
 assert actual_body in get(B+'actual_freeze_input_recording.hpp')and fs[B+'actual_freeze_input_recording.hpp']['SHA256']==identity['generated_header_SHA256']
 c=js(B+'pure-contract-run.log');assert c==q['pure_address_contract']and c['passed']and c['checks']==32990 and c['actual_freeze_input_template_cases']==66 and c['Tensor_leaf_recording_not_SDK']
 assert c['full_context']==32768 and c['active_pages']==128 and c['blocks']==132 and c['state_rows']==2 and c['last_active_slot']==33023 and c['last_storage_slot']==33791
 acct=get('accounting964966');assert '910407.966|Kimi456-AR-persistent-storage-CPU-v2|COMPLETED|0:0|00:01:26|a14r2n20'in acct and '910407.964|Kimi456-AR-persistent-storage-CPU-v1|FAILED|1:0'in acct
 assert not a['owned_srun_PID2610528_cmdline']or 'Kimi456-AR-persistent-storage-CPU-v2'not in a['owned_srun_PID2610528_cmdline']
 return dict(schema=1,passed=True,scope='Actual HOST/new private header/freshMD/static ELF dependency closure plus actual freeze_input template with recording Tensor CPUleaf; not actualSDK/device execution',actual_CPU='910407.966',state='COMPLETED',exit='0:0',elapsed_s=q['elapsed_s'],source_manifest_SHA256=SOURCE,production_manifest_SHA256=PRODUCTION,CPU_receipt_SHA256=fs[B+'CPU-result.json']['SHA256'],Native=q['native'],Native_SHA256=NATIVE,rawfiles=113,artifacts_metadata=35,production_HOST_TUs=5,pure_metadata_TUs=1,reused_objects=28,old_strong_exports=389,old_CAPI=133,new_CAPI=sorted(added),actual_template_body_SHA256=identity['actual_template_body_SHA256'],recording_template_cases=66,address_checks=32990,recording_Tensor_not_SDK=True,added_pinned_external_header_counts=adds,source_same=same,old964FAIL_preserved=True,old964_compiled_TUs=0,archive_path=t['sealed_path'],archive_SHA256=ARCHIVE,archive_uncompressed_SHA256=t['uncompressed_SHA256'],GPU=0,Model=0,Native_dlopen=False,SDK_execution=False,Runtime_created=False,GPU_GO=False,Model_GO=False,fit_numeric_H3_quality_performance='UNKNOWN',physical_budget_limit='Conservative logical parameter+exact pool block scenario leaves355652804B before driver/CCL/workspace; pool block bytes are not a driver allocation upper bound.')
if __name__=='__main__':
 q=audit(sys.argv[1]);b=(json.dumps(q,indent=2)+'\n').encode()
 if len(sys.argv)>2:p=Path(sys.argv[2]);assert not p.exists();p.write_bytes(b)
 else:print(b.decode())
