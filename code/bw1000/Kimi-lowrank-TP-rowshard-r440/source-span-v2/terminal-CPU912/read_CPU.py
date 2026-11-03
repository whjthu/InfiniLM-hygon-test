"""Independent stdlib audit of original CPU912 static artifacts, no SDK execution."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,shlex
ARCHIVE_SHA='4511d3441594524c55c807c0ec64ad758325662605d14f3f435af60743d27018'
MF_SHA='ba0a985b647443c725f51ef5ead0fb8fa8e57b08619c5bf05e222c521fa51470'
RECEIPT_SHA='e11cac669cb6c650879c4b32d8e8d06afe407e125d5a7e18471f61fb72cc62e8'
LIB_SHA='fe1d0b59f69b74fa973f10a1623216423b00e07c55d91eb6d5afde82df041ad9'
ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
B=ROOT+'/candidates/Kimi-lowrank-TP-rowshard-r440-CPU-source-span-header-v4'
D=ROOT+'/incoming/Kimi-lowrank-TP-rowshard-r440/source-span-v2/header-fix-v4'
sha=lambda b:hashlib.sha256(b).hexdigest()
def symbols(text):return {p[-1]:p[-2]for line in text.splitlines()if len(p:=line.split())>=3}
def audit(path):
 raw=Path(path).read_bytes();assert sha(raw)==ARCHIVE_SHA
 a=json.loads(gzip.decompress(raw));assert a['actual_CPU']=='910407.912'and a['new_jobs']==0 and a['GPU']==0 and not a['Runtime_execution']and not a['Native_dlopen']
 f={}
 for p,v in a['files'].items():
  b=base64.b64decode(v['base64'],validate=True);assert len(b)==v['bytes']and sha(b)==v['SHA256'],p;f[p]=b
 assert len(f)==40 and '910407.912|Kimi-r440-source-span-header-CPU-v4|COMPLETED|0:0|00:00:38|a14r2n20'in a['accounting']
 assert sha(f[B+'/CPU-build.json'])==RECEIPT_SHA;q=json.loads(f[B+'/CPU-build.json']);assert q['passed']and q['completed']and q['actual_CPU']=='910407.912'
 assert q['new_production_HOST_TUs']==2 and q['pure_contract_HOST_TUs']==0 and q['new_GPU_TUs']==0 and not q['old_successful_TUs_recompiled']
 assert q['GPU']==0 and not q['Native_dlopen']and not q['Runtime_created']and q['weights']==0 and not q['GPU_GO']and not q['Model_GO']and not q['new_device_or_math_qualification']
 assert q['library_SHA256']==LIB_SHA and q['source_manifest_SHA256']==MF_SHA and not q['old432GPU_linked']
 m=json.loads(f[D+'/SOURCE_MANIFEST.json']);assert sha(f[D+'/SOURCE_MANIFEST.json'])==MF_SHA and len(m['files_SHA256'])==19
 for n,h in m['files_SHA256'].items():assert sha(f[D+'/'+n])==h,n
 contract=json.loads(f[D+'/BUILD_CONTRACT.json']);assert q['reused_original33']==contract['parent_native']['all_linked_object_SHA256']and len(q['reused_original33'])==33
 identities=a['actual_artifact_identities'];assert all(identities[p]==h for p,h in q['reused_original33'].items())
 assert len(q['compiled'])==2
 def host(p):
  assert p.startswith('/');return ROOT+p[len('/refactor'):]if p.startswith('/refactor/')else p
 for c in q['compiled']:
  assert identities[c['object']]==c['object_SHA256']and identities[c['MD']]==c['MD_SHA256']
  md=f[host(c['MD'])];assert sha(md)==c['MD_SHA256'];deps=set(shlex.split(md.decode().replace('\\\n',' ').split(':',1)[1]));assert deps==set(c['dependency_SHA256'])
  assert sha(f[host(c['source'])])==c['source_SHA256']
  for p,h in c['dependency_SHA256'].items():
   if host(p)in f:assert sha(f[host(p)])==h,p
 source=q['actual_source_contract'];assert all(source[k]for k in ['passed','whole_original_MoE_body_except_exact_five_projection_calls','caller_owns_leases_through_original_sync','release_and_scope_destruction_in_original_primary_interval'])and not source['SDK_execution']
 for c in q['commands']:assert c['rc']==0 and sha(f[host(c['log'])])==c['SHA256'],c['label']
 expected=['changed-actual-full-consumer-source-contract','private_moe.cpp-preMD','private_moe.cpp-compile','component.cpp-preMD','component.cpp-compile','before-private-global','localize-exact-seven-clone-wrappers','after-private-global','isolated-full-native-plus-new-consumer-link-no432GPU','new-symbols','qualified8df-symbols','new-dynamic','actual-Common-AllGather-provider','new-ldd-r'];assert [c['label']for c in q['commands']]==expected
 old=symbols(f[B+'/qualified8df-symbols.log'].decode());new=symbols(f[B+'/new-symbols.log'].decode());strong={k:v for k,v in old.items()if v in ['T','D','B','R']};assert len(strong)==q['old_strong_symbols_preserved']and all(new.get(k)==v for k,v in strong.items())
 for name in ['modern_kimi_lowrank_full_moe_r440','modern_kimi_lowrank_full_moe_result_r440']:assert new.get(name)=='T'
 before=symbols(f[B+'/before-private-global.log'].decode());after=symbols(f[B+'/after-private-global.log'].decode());loc=q['localized_private_object'];exact=loc['only_localized'];assert len(exact)==7 and all(before.get(k)=='T'for k in exact)and after=={k:v for k,v in before.items()if k not in exact}
 assert loc['other_globals_and_all_text_bytes_unchanged']and loc['all_text_sections_SHA256']and identities[loc['object']]==loc['object_SHA256']
 dynamic=f[B+'/new-dynamic.log'].decode();ldd=f[B+'/new-ldd-r.log'].decode();assert 'SYMBOLIC'in dynamic and 'not found'not in ldd and 'undefined symbol'not in ldd
 link=next(c['argv']for c in q['commands']if c['label']=='isolated-full-native-plus-new-consumer-link-no432GPU');assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link and contract['old_GPU_object']['object']not in link
 ag='_ZN8infinilm6engine11distributed10allgather_ERNS_7runtime7RuntimeENS2_6TensorERKS5_Pv';assert symbols(f[B+'/actual-Common-AllGather-provider.log'].decode()).get(ag)=='T'
 reused=q['reused_CPU910_pure_contract'];assert not reused['recompiled']and not reused['reexecuted']and reused['result']['passed']and reused['result']['changed_contract_cases']==27
 old910path=ROOT+'/candidates/Kimi-lowrank-TP-rowshard-r440-CPU-source-span-repair-v3/CPU-build.json';assert sha(f[old910path])==reused['receipt_SHA256'];old910=json.loads(f[old910path]);assert not old910['passed']and old910['actual_pure_contract']==reused['result']
 assert identities[q['library']]==LIB_SHA and identities[q['actual_AllGather_provider']['path']]==q['actual_AllGather_provider']['SHA256']
 # Actual full consumer source is evidence for timing/lifetime ordering, not device execution.
 c=f[D+'/component.cpp'].decode();t=c[c.index('const auto begin=Clock::now();rt::Tensor y;'):c.index('const double native_ms=ms(begin);')];assert 'scope.complete_original_forward()'in t and 'retained_handles=scope.retained_before_complete();}'in t and 'r.syncStream()'not in t
 return dict(passed=True,actual_CPU='910407.912',COMPLETED0=True,elapsed_s=q['elapsed_s'],sealed_files=len(f),actual_artifact_identities=len(identities),source_members=19,source_manifest_SHA256=MF_SHA,receipt_SHA256=RECEIPT_SHA,library=q['library'],library_SHA256=LIB_SHA,new_production_HOST_TUs=2,reused_original33=len(q['reused_original33']),old_strong_CABI_preserved=len(strong),exact_localized_C_wrappers=len(exact),recording_lease_span_CPU910_cases_reused=27,new_pure_cases_reexecuted=0,all_actual_command_raw_hashes_checked=True,primary_timer_includes_original_completion_lease_clear_scope_destroy=True,old432_GPU_object_linked=False,GPU_GO=False,Model_GO=False,SDK_execution=False,device_numeric_performance_collective_qualification='UNKNOWN',decision='CPU_STATIC_COMPLETE_ONLY; concrete device claim and actual TP16 qualification remain separate')
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');a=p.parse_args();q=audit(a.archive);s=json.dumps(q,indent=2)+'\n'
 if a.output:Path(a.output).write_text(s)
 print(s)
if __name__=='__main__':main()
