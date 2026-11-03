"""Read original938 bytes and exact accounting, no Native/Runtime imports."""
import argparse,base64,hashlib,json
from pathlib import Path
RAW='f7c05b962cc8e5a226725732256e54775611771872c275b5ed662f369577ed0c'
MF='03021a3669848d8e4c494679999d119bc7a3eba15d6556fdddc39c2782c262bd'

def analyze(observation,accounting):
 o=json.loads(Path(observation).read_bytes());a=json.loads(Path(accounting).read_bytes())
 for name,f in o['files'].items():
  raw=base64.b64decode(f['base64']);assert len(raw)==f['bytes']and hashlib.sha256(raw).hexdigest()==f['SHA256'],name
 f=o['files']['candidates/Deep449-public-pair-CPU-v1/result.json'];assert f['SHA256']==RAW;q=json.loads(base64.b64decode(f['base64']))
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.938'and q['source_manifest_SHA256']==MF
 assert not any(q[k]for k in ['Native_dlopen','Runtime_execution','Model_loaded','GPU','new_HOST_TUs','new_GPU_TUs'])
 rows=[s.split('|')for s in a['accounting938']['stdout'].splitlines()if s];assert a['accounting938']['rc']==0 and len(rows)==1 and rows[0][:4]==['910407.938','Deep449-public-pair-CPU-v1','COMPLETED','0:0']and rows[0][5]=='a14r2n20'
 c=q['worker_control'];assert c['passed']and c['recording_CABI_only']and not c['actual_Model_or_HTTP']and not c['GPU']and not c['Native_dlopen']
 assert [v['scenario']for v in c['cases']]==['six-modes','no-seventh','missing-hit','forward-failure','reset-drain-failure','disconnect','timeout']
 for v in c['cases']:assert v['passed']and v['typed_libffi']and v['high64_counter_buffers']and v['high64_context']>2**32 and not v['actual_Native_or_GPU']
 assert [v['recording_Native_calls']for v in c['cases']]==[66,66,22,1,11,10,10]
 for cmd in q['commands']:
  assert cmd['rc']==0
  for ext in ['stdout','stderr']:
   f=o['files']['candidates/Deep449-public-pair-CPU-v1/'+cmd['name']+'.'+ext];assert f['SHA256']==cmd[ext+'_SHA256']
 b=q['actual_binding'];assert b['objects']==50 and b['guard_retained']and not b['project_maps']and set(b['hidden'].values())=={'-1'}and b['library_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
 assert q['actual_whole_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 return dict(passed=True,actual_CPU='910407.938',Slurm='COMPLETED0/1s',raw_receipt_SHA256=RAW,source_manifest_SHA256=MF,actual50object_derivation=True,actual_public_CLI_nonserve=True,changed_worker_typed_CABI_cases=7,actual_Native_Runtime_GPU_Model=False,Model_quality_or_performance='UNMEASURED')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('observation');p.add_argument('accounting');a=p.parse_args();print(json.dumps(analyze(a.observation,a.accounting),sort_keys=True,indent=2))
