"""Offline stdlib audit of the original first-baseline timeout; no job/device."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
SEAL='39f4dd8aafb492671f9a4d533d07c7bdc28c94cc2ad746496efd889d706961cb'
def audit(path):
 b=Path(path).read_bytes();assert hashlib.sha256(b).hexdigest()==SEAL
 z=json.loads(gzip.decompress(b));files={}
 for name,v in z['files'].items():
  raw=base64.b64decode(v['bytes'],validate=True);assert len(raw)==v['nbytes']and hashlib.sha256(raw).hexdigest()==v['SHA256'];files[name]=raw
 def doc(suffix):
  found=[v for p,v in files.items()if p.endswith(suffix)];assert len(found)==1;return json.loads(found[0])
 w=doc('/Kimi439-watch-v1.json');a=doc('/Kimi439-910371.760-main.json')
 assert w['completed']and not w['passed']and w['actual_step']=='910371.760'
 assert w['publication_commit']=='d51e6c321b28d8f3ac6385d34e50786cb618de8b'
 assert a['completed']and not a['passed']and a['Runtime_processes_normal']==0
 assert a['source_manifest_SHA256']==w['GPU_source_manifest_SHA256']=='c61d457a847e7d32825be8f42ded2d6e1835b64237ed4cef75f179a18a59126b'
 assert len(a['records'])==1;r=a['records'][0]
 assert r['arm']=='baseline'and r['operation']=='check'and r['width']==0 and r['warm']and r['timeout']and r['rc']==-9
 assert 25<=r['process_inclusive_wall_s']<27 and 'actual_result'not in r
 for key in ['stdout','stderr']:
  name='/public/home/aclsylqidf/hygon-inference-refactor-20261014'+r[key]['path'][len('/refactor'):]
  assert files[name]==b''and hashlib.sha256(files[name]).hexdigest()==r[key]['SHA256']
 post=w['failure_post'];assert post['passed']and post['actual_step']=='910371.761'
 assert post['KFD_pids']==post['matching_GPU_driver_PIDs']==post['expected_PIDs_still_exist']==[]
 assert set(post['expected_owner_PIDs'])=={a['PID'],r['PID']}and w['owned_failure_cleanup']['closed']
 account={v[0]:v[2:5]for v in z['account']};assert account['910371.760'][:2]==['FAILED','1:0']and account['910407.893'][:2]==['FAILED','1:0']and account['910371.761'][:2]==['COMPLETED','0:0']
 return dict(offline_original_evidence_audit=True,component_passed=False,status='STOP_FIRST_BASELINE_PROCESS_CAP',actual_account=account,seal_SHA256=SEAL,files=len(files),source_bytes=sum(map(len,files.values())),attempted_processes=1,normal_Runtime_processes=0,ON_processes=0,formal_samples=0,first_process_wall_s=r['process_inclusive_wall_s'],whole_GPU_epoch_elapsed_s=a['elapsed_since_original_GPU_epoch_s'],actual_visible_env=a['actual_visible_env'],actual_CPU_affinity=a['CPU_affinity'],Cpp_report_available=False,exact_blocked_phase='UNKNOWN: no C++ progress before terminal report; empty stdout/stderr. Cannot separate ELF initialization, Runtime creation or H2D check progress.',actual_device_mapping='UNKNOWN: device properties were not persisted',actual_completed_H2D_calls='UNKNOWN',normal_Runtime_destroy=False,performance='UNKNOWN',Model=False,no_retry=True,failure_post=post)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('seal');p.add_argument('--output');args=p.parse_args();result=audit(args.seal);s=json.dumps(result,indent=2,allow_nan=False)+'\n'
 if args.output:Path(args.output).write_text(s,encoding='utf8')
 print(s)
