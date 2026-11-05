from pathlib import Path
import argparse,gzip,hashlib,json,shlex
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def load(path,size,sha):
 raw=Path(path).read_bytes();assert len(raw)==size and hashlib.sha256(raw).hexdigest()==sha;q=json.loads(gzip.decompress(raw))
 for p,v in q['files'].items():
  b=v['rawtext'].encode();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['SHA256'],p
 return q
def read(failed_archive,linked_archive):
 a=load(failed_archive,83998,'96ef4bfda699ec2038258861da583c460eca47c0371ff73f831bd4bce6fabd23');b=load(linked_archive,51545,'c5f834c8121b615f661ab54cb34cb2dfec5685391068f863a77d5dd9f3509f0c');assert len(a['files'])==25 and len(b['files'])==18 and len(a['artifacts'])==len(b['artifacts'])==1
 A=R+'/candidates/Kimi-split-V128-component-CPU-v1/';B=R+'/candidates/Kimi-split-V128-component-link-v2/'
 oldfile=a['files'][A+'CPU-result.json'];old=json.loads(oldfile['rawtext']);new=json.loads(b['files'][B+'CPU-result.json']['rawtext'])
 assert old['actual_CPU']=='910407.1076'and old['completed']and not old['passed']and old['phase']=='probe-link'
 assert a['accounting_original']==['910407.1076|Kimi-split-V128-component-CPU-v1|FAILED|1:0|00:00:12']
 assert new['actual_CPU']=='910407.1080'and new['completed']and new['passed']and new['new_HOST']==new['new_GPU']==0
 assert b['accounting_original']==['910407.1080|Kimi-split-V128-component-link-v2|COMPLETED|0:0|00:00:01']
 c=old['compiled'];obj='/refactor/candidates/Kimi-split-V128-component-CPU-v1/probe.o';assert a['artifacts'][A+'probe.o']['SHA256']==c['object_SHA256']==new['reused1076']['object_SHA256']
 assert c['MD_SHA256']==a['files'][A+'probe.d']['SHA256']==new['reused1076']['MD_SHA256']
 paths=set(shlex.split(a['files'][A+'probe.d']['rawtext'].replace('\\\n',' ').split(':',1)[1]));assert paths==set(c['all_MD'])and len(paths)==new['reused1076']['MD_records']
 assert set(shlex.split(a['files'][A+'probe.before.d']['rawtext'].replace('\\\n',' ').split(':',1)[1]))==paths
 assert new['reused1076']['receipt']['SHA256']==oldfile['SHA256']and not new['reused1076']['successful_HOST_recompiled']
 local=old['local_entry_reused'];assert local['actual_stdout']['passed']and local['actual_stdout']['actual_high64_CAPI']and local['actual_stdout']['recorded_query_calls']==20 and local['actual_stdout']['recorded_tokens']==8196
 assert new['local_entry_reused']==local and not new['old_CPU_contracts_reexecuted']
 before=old['commands'][-1];after=new['commands'][0];native=new['parent_Native']['path'];assert after[1:3]==[obj,native]
 assert [x for x in before if x.startswith('-l')]==[x for x in after if x.startswith('-l')]
 def normalize(cmd):return [x for i,x in enumerate(cmd)if x not in [obj,native]and not x.startswith('-Wl,-Map,')and(i==0 or cmd[i-1]!='-o')]
 assert normalize(before)==normalize(after)
 symbols=b['files'][B+'probe-symbols.log']['rawtext'];required=new['CAPI'];assert len(required)==10
 for name in required:assert any(len(x.split())>=3 and x.split()[-2]=='T'and x.split()[-1]==name for x in symbols.splitlines())
 for marker in ['_ZN15kimi_split_v1288allocate','_ZN15kimi_split_v1287forward','kimi_split_store']:assert any(marker in x and ' U 'in x for x in symbols.splitlines())
 assert 'SYMBOLIC'in b['files'][B+'probe-ELF.log']['rawtext']and 'libKimi_split_v128_private.so'in b['files'][B+'probe-ELF.log']['rawtext']
 log=b['files'][B+'probe-relocation-full-streams.log'];assert log['SHA256']==new['logs']['probe-relocation-full-streams']['SHA256']and new['logs']['probe-relocation-full-streams']['rc']==0
 assert 'undefined symbol'not in log['rawtext']and 'not found'not in log['rawtext']
 assert b['artifacts'][B+'libKimi_split_v128_component.so']['SHA256']==new['probe_SHA256']=='2985ce19f42e5daf3a9d460dfd95f97a669c5113b648b8970f79d52585af7555'
 assert new['project_maps']==[]and not new['Torch_imported']and not new['Native_dlopen']and not new['SDK_device_execution']
 return dict(audit_passed=True,CPU_link_passed=True,raw_files_SHA_verified=43,artifact_SHA_metadata_records=2,artifact_raw_recomputed_locally=False,failed_CPU1076=a['accounting_original'][0],passed_CPU1080=b['accounting_original'][0],failed_elapsed_s=old['elapsed_s'],link_elapsed_s=new['elapsed_s'],successful_HOST_object_SHA256=c['object_SHA256'],actual_MD_records=len(paths),CPU1080_compile=0,old_contracts_rerun=False,actual_local_entry_recording=local['actual_stdout'],CAPI=required,production_exports_undefined_to_actual_c78=True,Native_c78=new['parent_Native'],probe=new['probe'],probe_SHA256=new['probe_SHA256'],ldd_full_stdoutstderr_SHA256=log['SHA256'],project_maps=[],GPU_GO=False,Model_GO=False,qualification='1actual HOST compile then0compilelink repair +recording only; no SDK/Runtime/FA/device/numeric/fit qualification')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('failed_archive');p.add_argument('linked_archive');a=p.parse_args();print(json.dumps(read(a.failed_archive,a.linked_archive),indent=2,sort_keys=True))
