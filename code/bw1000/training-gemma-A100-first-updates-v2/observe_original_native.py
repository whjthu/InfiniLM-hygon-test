from pathlib import Path
import json,subprocess
B=Path(__file__).parent;D=B/'training-gemma-A100-first-updates-v2';assert not (D/'first-native-observation.json').exists();intent=json.loads((D/'launch-intent.json').read_bytes())
remote=r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-A100-first-updates-v2');out=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ranks=[])
for name in ['installed.json','source-CPU-tests.json','supervisor-intent.json','gpu-intent.json','child-owner.json','prelaunch-result.json','outcome.json']:
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=300000;out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
assert out['installed.json']['value']['claim']==CLAIM and out['installed.json']['value']['request_id']==REQUEST
child=out.get('child-owner.json',{}).get('value',{});identity=child.get('identity');out['owned_native_root_live']=None
if identity:
 p=Path('/proc',str(child['PID']),'stat')
 if p.is_file():
  parts=p.read_text().split(') ',1)[1].split();out['owned_native_root_live']=int(parts[19])==identity['startticks'] and parts[0]!='Z'
 else:out['owned_native_root_live']=False
for rank in range(8):
 root=W/'capture'/('rank-%02d'%rank);r=dict(rank=rank,receipt=(root/'receipt.json').is_file(),native25=(root/'native25-complete.json').is_file())
 p=root/'progress.json'
 if p.is_file():
  v=json.loads(p.read_bytes());r.update({k:v.get(k) for k in ['inputs_saved','outputs_saved','completed_updates','clip_calls','failed','serialization_and_hash_seconds']});r['files']=len(v.get('files',[]))
 p=root/'failure.json'
 if p.is_file():r['failure_error_type']=json.loads(p.read_bytes())['error_type']
 out['ranks'].append(r)
out['actual_GPU_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=12).strip().splitlines()
print(json.dumps(out))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');a=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];r=subprocess.run(a,input=('CLAIM='+repr(intent['claim'])+'\nREQUEST='+repr(intent['request_id'])+'\n'+remote).encode(),capture_output=True,timeout=25);(D/'first-native-observe-stdout.private.json').write_bytes(r.stdout);(D/'first-native-observe-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(D/'first-native-observation.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());o=v.get('outcome.json',{}).get('value',{});print(json.dumps(dict(source_CPU_tests=v.get('source-CPU-tests.json',{}).get('value'),native_root_live=v['owned_native_root_live'],GPU_apps=len(v['actual_GPU_apps']),ranks=v['ranks'],native_returncode=o.get('native_returncode'),keeper_passed=o.get('keeper_passed'))))
