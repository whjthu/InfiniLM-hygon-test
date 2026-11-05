"""Offline original1057 and prior invoker parseFAIL. No benchmark rerun."""
from pathlib import Path
import hashlib,json,argparse,statistics
H=lambda b:hashlib.sha256(b).hexdigest()
def read(path,proof):
 q=json.loads(Path(path).read_bytes());files=q['files'];prefix='candidates/Kimi-loader-TP32-copy-CPU-v1/'
 for v in files.values():
  if 'text'in v:assert len(v['text'].encode())==v['bytes']and H(v['text'].encode())==v['SHA256']
 v=json.loads(files[prefix+'result.json']['text']);assert v['completed']and v['passed']and v['actual_CPU']=='910407.1057'and v['source_manifest_SHA256']=='c6e06909395e0e4b6d44aba55753d0d93e2dd1e926db9470c00b1b365f7f3262'
 assert v['CPU_only']and v['GPU']==0 and not v['Native_dlopen']and not v['SDK_execution']and not v['weights']and not v['project_maps']
 assert q['accounting']['rows']==[['910407.1057','Kimi-loader-TP32-copy-CPU-v1','COMPLETED','0:0','00:00:01']]
 assert len(v['actual_mask'])==1 and len([c for c in v['commands']if c[0]=='/usr/bin/g++'and '-o'in c])==1
 consumer=v['consumer'];assert consumer['passed']and consumer['fallback_PASS'];out=[]
 for g in consumer['geometries']:
  assert g['width']in[48,3]and g['rows']==3584 and g['stride']==g['width']*32 and g['all32offsets_byte_canary_PASS']
  arms=g['arms'];assert [a['mode']for a in arms]==[0,1,0,1,1,0]and [a['warm']for a in arms]==[True,True,False,False,False,False]
  assert all(a['calls']==256 and a['blocks']==3584*256 and a['bytes']==3584*256*g['width']for a in arms)
  off=[a['seconds']for a in arms[2:]if a['mode']==0];on=[a['seconds']for a in arms[2:]if a['mode']==1]
  out.append(dict(width=g['width'],stride=g['stride'],rows=3584,all32offsets_byte_canary_PASS=True,formal_OFF_s=off,formal_ON_s=on,reduction_fraction=1-statistics.mean(on)/statistics.mean(off),OFF_endpoint_drift_fraction=abs(off[-1]/off[0]-1),scope='one CPU core, synthetic warm RAM; not Native load/H2D/Model speed'))
 old=files['results/Kimi-loader-TP32-copy-CPU-v1/invoker.stderr'];assert old['SHA256']=='51a54f139eba78181ad759422c04f9c1d3176acea8a0bff694bf06ce8eddce31'and 'SyntaxError'in old['text']
 p=json.loads(Path(proof).read_bytes());assert p['passed']and p['compile_only']and p['old_failure_preserved']and not p['Native_source_changed']
 return dict(audit_PASS=True,CPU_PASS=True,actual_CPU=v['actual_CPU'],elapsed_s=v['elapsed_s'],source_manifest_SHA256=v['source_manifest_SHA256'],geometries=out,prior_invoker_failed_before_intent_or_srun=True,prior_failure_SHA256=old['SHA256'],exact_final_control_compile=p,fallback_PASS=True,GPU=0,Runtime_changed=False,Model_gain_proven=False,new_tests_by_reader=0,raw_files=len(files),original_files_SHA256=H(Path(path).read_bytes()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('files');p.add_argument('control_proof');a=p.parse_args();print(json.dumps(read(a.files,a.control_proof),indent=2,sort_keys=True))
