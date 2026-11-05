OLD_READER_SHA256='d1a3c829f814d9a323ecee136b6c24557632e5cd6d13e58207fb5068bf6358d0'
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,shlex
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def read(old_archive,archive):
 reader=Path(__file__).parent.parent/'terminal-CPU1070-1073-v1/read_original.py';assert hashlib.sha256(reader.read_bytes()).hexdigest()==OLD_READER_SHA256
 spec=importlib.util.spec_from_file_location('original1073',reader);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);history=old.read(old_archive)
 raw=Path(archive).read_bytes();assert len(raw)==709487 and hashlib.sha256(raw).hexdigest()=='9fc8e4877bef24a53cfcf7c82f65e92f76b6ba03437fc542d9e55aabe1d999b5'
 q=json.loads(gzip.decompress(raw));files=q['files'];art=q['artifacts'];assert len(files)==28 and len(art)==3
 for p,v in files.items():
  b=v['rawtext'].encode();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['SHA256'],p
 prior=json.loads(gzip.decompress(Path(old_archive).read_bytes()))
 assert q['prior_archive']['SHA256']==hashlib.sha256(Path(old_archive).read_bytes()).hexdigest()
 b=R+'/candidates/Kimi-split-V128-CPU-v5/';v=json.loads(files[b+'CPU-result.json']['rawtext']);assert v['completed']and v['passed']and v['actual_CPU']=='910407.1075'
 assert q['accounting_original']==['910407.1075|Kimi-split-V128-CPU-v5|COMPLETED|0:0|00:00:25']
 def physical(p):
  assert p.startswith('/refactor/');return R+p[len('/refactor'):]
 def file(p):
  p=physical(p);return files[p]if p in files else prior['files'][p]
 old_result=prior['files'][R+'/candidates/Kimi-split-V128-CPU-v4/CPU-result.json'];reuse=v['reused_original_failed_CPU'];assert reuse['receipt']['SHA256']==old_result['SHA256']and not reuse['passed']and reuse['actual_CPU']=='910407.1073'
 t=json.loads(old_result['rawtext']);assert reuse['successful_HOST_objects']==t['compiled']and reuse['successful_contracts']=={k:t[k]for k in ['CPU_contract','CPU_cache','binding_recording']}
 assert not v['reused_source_contracts_executed_this_job']and v['new_HOST_TUs']==1 and v['new_GPU_TUs']==1 and v['pure_CPU_TUs']==0
 assert set(v['compiled'])=={'fa_backend','fa_kernels'}
 units={}
 for name,c in v['compiled'].items():
  assert file(c['source'])['SHA256']==c['source_SHA256']and art[physical(c['object'])]['SHA256']==c['object_SHA256']
  md=file(c['object'][:-2]+'.d');assert md['SHA256']==c['MD_SHA256'];paths=set(shlex.split(md['rawtext'].replace('\\\n',' ').split(':',1)[1]));assert paths==set(c['all_MD'])
  before=file(c['object'][:-2]+'.before.d');assert set(shlex.split(before['rawtext'].replace('\\\n',' ').split(':',1)[1]))==paths
  units[name]={'source_SHA256':c['source_SHA256'],'object_SHA256':c['object_SHA256'],'MD_SHA256':c['MD_SHA256'],'MD_records':len(paths)}
 assert len(v['all_linked_object_SHA256'])==36 and len(v['reused_objects_SHA256'])==31
 for c in t['compiled'].values():assert v['all_linked_object_SHA256'][c['object']]==c['object_SHA256']
 for p,h in v['reused_objects_SHA256'].items():assert v['all_linked_object_SHA256'][p]==h
 native=art[physical(v['native'])];assert native['SHA256']==v['native_SHA256']=='c78a27cab1cefd9385f4b423b6b45282898e641ec28efe5b52968a298bed0e21'
 def strong(log):return {x.split()[-1]for x in files[b+log]['rawtext'].splitlines()if len(x.split())>=3 and x.split()[-2]in ['T','D','B','R']}
 previous=strong('parent-symbols.log');current=strong('new-symbols.log');assert previous<=current and len(previous)==v['old_strong_exports_retained']==398
 assert len([x for x in previous if x.startswith('modern_')])==v['old_CAPI_retained']==142
 assert set(v['new_CAPI'])<=current and len(v['new_CAPI'])==2
 elf=files[b+'new-ELF.log']['rawtext'];assert 'SYMBOLIC'in elf and 'NEEDED'in elf
 ldd=files[b+'relocation-trace-no-entry.log'];assert ldd['SHA256']==v['logs']['relocation-trace-no-entry']['SHA256'];assert v['logs']['relocation-trace-no-entry']['rc']==0 and 'not found'not in ldd['rawtext']and 'undefined symbol'not in ldd['rawtext']
 assert v['project_maps']==[]and not v['Torch_imported']and v['GPU']==0 and not v['Native_dlopen']and not v['SDK_device_execution']
 return dict(audit_passed=True,CPU_compile_link_passed=True,actual_CPU='910407.1075',elapsed_s=v['elapsed_s'],accounting=q['accounting_original'],raw_files_verified=len(files)+history['raw_files_verified'],artifact_SHA_metadata_records=len(art)+history['artifact_SHA_metadata_records'],artifact_bytes_recomputed_locally=False,prior_failures_preserved=history['original_CPU_results'],reused1073_production_HOST=3,reused_contracts_executed=False,new_units=units,total_linked_objects=36,reused_b8_objects=31,old_strong_exports=398,old_CAPI=142,new_CAPI=v['new_CAPI'],Native_SHA256=v['native_SHA256'],ldd_full_combined_stdout_stderr_SHA256=ldd['SHA256'],project_maps=v['project_maps'],Torch_imported=False,SDK_planning_supports_dynamic='UNKNOWN; compile only',source_payload_saving=311427072,explicit_scratch_pool_slot_hypothesis=10092544,net_excess_over_643_deficit_hypothesis=5636096,FA_driver_Torch_workspace_unknown=True,GPU_GO=False,Model_GO=False,qualification='actual isolated compile/link and reused pureCPU contracts; no SDKdevice/Runtime/FA execution, fit, numeric or performance qualification')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('old_archive');p.add_argument('archive');a=p.parse_args();print(json.dumps(read(a.old_archive,a.archive),indent=2,sort_keys=True))
