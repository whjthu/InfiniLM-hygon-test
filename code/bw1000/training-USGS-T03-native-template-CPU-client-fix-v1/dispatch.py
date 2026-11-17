import sys,re
from pathlib import Path
import json,hashlib,subprocess
CLIENT=Path(__file__).resolve().parent
CLIENT_PUB=sys.argv[1]
assert re.fullmatch('[0-9a-f]{40}',CLIENT_PUB)
CG=Path('.').resolve()
def client_h(b):return hashlib.sha256(b).hexdigest()
cm_raw=(CLIENT/'SOURCE_FILES.json').read_bytes();cm=json.loads(cm_raw)
for name in [*cm['files'],'SOURCE_FILES.json']:
 raw=(CLIENT/name).read_bytes()
 if name!='SOURCE_FILES.json':assert len(raw)==cm['files'][name]['bytes'] and client_h(raw)==cm['files'][name]['SHA256']
 assert subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+CG.as_posix(),'show',CLIENT_PUB+':'+(CLIENT/name).relative_to(CG).as_posix()],cwd=CG,timeout=5)==raw
client_claim=json.loads((CLIENT/'CLAIM.json').read_bytes())
assert client_claim['root_GO'] and client_claim['CPU_GO_once'] and client_claim['case']=='T03.sft'
failed=CG/client_claim['prelaunch_failure']['path'];oldraw=failed.read_bytes()
assert client_h(oldraw)==client_claim['prelaunch_failure']['SHA256']
previous=json.loads(oldraw)
assert previous['request_id']=='01fee01c01bd4545ac96629d63d23d70'
assert previous['local_adapter_attempted'] is False and previous['exec_request_attempted'] is False and previous['remote_outcome']=='NOT_ATTEMPTED' and previous['client_error_type']=='FileNotFoundError'

import base64,hashlib,json,pathlib,subprocess,time,uuid
G=pathlib.Path('.').resolve();D=G/'code/bw1000/training-USGS-T03-T04-native-template-CPU-v1'
PY=pathlib.Path('C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
case='T03.sft';pub='477ed0d5720d8caf6d5da8c434462d56cad69818'
R=G/'local-only/training-USGS-T03-native-template-CPU-v1-original-client-fixed';R.mkdir()
epoch=time.time();mono=time.monotonic()+295;request=uuid.uuid4().hex
def h(b):return hashlib.sha256(b).hexdigest()
def left():return max(.001,min(epoch+295-time.time(),mono-time.monotonic()))
def write(name,x):
 b=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode();(R/name).write_bytes(b)
record=dict(client_publication=CLIENT_PUB,prelaunch_failure_SHA256=client_h(oldraw),case=case,request_id=request,publication=pub,epoch=epoch,deadline=epoch+300,local_adapter_attempted=False,exec_request_attempted=False,remote_outcome='NOT_ATTEMPTED',replay_allowed=False)
write('intent.json',record)
try:
 mf=(D/'SOURCE_FILES.json').read_bytes();m=json.loads(mf)
 assert h(mf)=='cf185df13ffd6d0403a34c6b29f4c5c11ac593c392353fc9e6540f7118cf3a4a'
 package={}
 for name in [*m['files'],'SOURCE_FILES.json']:
  raw=(D/name).read_bytes()
  assert subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'show',pub+':'+(D/name).relative_to(G).as_posix()],cwd=G,timeout=min(5,left()))==raw
  if name!='SOURCE_FILES.json':assert h(raw)==m['files'][name]['SHA256'] and len(raw)==m['files'][name]['bytes']
  package[name]=base64.b64encode(raw).decode()
 plan=json.loads((D/'PLAN.json').read_bytes());target='/home/whj/nvidia-baseline/alignment-assets/training-USGS-T03-native-template-CPU-v1/data'
 data_source=G/'local-only/training-USGS-reported-event-charts-v1-original/dataset'
 record['local_adapter_attempted']=True;write('intent.json',record)
 cp=subprocess.run([str(PY),'-B','-X','utf8',str(D/'prepare_inputs.py'),'--audit',str(G/'results/bw1000/2026-11-17/training-USGS-reported-event-chart-data-independent-terminal.json'),'--dataset',str(data_source),'--output',str(R/'adapter'),'--target-data',target],capture_output=True,timeout=min(15,left()),cwd=G)
 (R/'prepare.stdout').write_bytes(cp.stdout);(R/'prepare.stderr').write_bytes(cp.stderr);record['local_adapter_rc']=cp.returncode
 assert cp.returncode==0,'Local domain adapter failed; no transport'
 data={}
 for name in [*plan['media'],'train-swift.jsonl','dev-swift.jsonl','data-binding-private.json']:
  raw=(data_source/name).read_bytes() if name in plan['media'] else (R/'adapter'/name).read_bytes()
  if name in plan['media']:assert h(raw)==plan['media'][name]['SHA256'] and len(raw)==plan['media'][name]['bytes']
  data[name]=base64.b64encode(raw).decode()
 payload=dict(case=case,request_id=request,publication=pub,epoch=epoch,source_manifest_SHA256=h(mf),claim_SHA256=h((D/'CLAIM.json').read_bytes()),package=package,data=data)
 raw=json.dumps(payload,separators=(',',':')).encode();assert len(raw)<2*1024**2
 (R/'input-bundle.private.json').write_bytes(raw)
 body="INSTALL_PAYLOAD_B64 = "+repr(base64.b64encode(raw).decode())+"\n"+(D/'install_and_run.py').read_text(encoding='utf-8')
 (R/'body.py').write_bytes(body.encode());record.update(input_bundle_bytes=len(raw),input_bundle_SHA256=h(raw),body_SHA256=h(body.encode()),exec_request_attempted=True,remote_outcome='UNKNOWN');write('intent.json',record)
 cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-F','C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config','a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -']
 cp=subprocess.run(cmd,input=body.encode(),capture_output=True,timeout=left())
 (R/'stdout.private').write_bytes(cp.stdout);(R/'stderr.private').write_bytes(cp.stderr)
 record.update(rc=cp.returncode,stdout_bytes=len(cp.stdout),stdout_SHA256=h(cp.stdout),stderr_bytes=len(cp.stderr),stderr_SHA256=h(cp.stderr),elapsed_seconds=time.monotonic()-(mono-295))
 for line in reversed(cp.stdout.splitlines()):
  try:q=json.loads(line)
  except (ValueError,TypeError):continue
  if isinstance(q,dict) and q.get('request_id')==request:
   record.update(remote_outcome='RETURNED',passed=q.get('passed'),worker_returncode=q.get('worker_returncode'),native_result=q.get('native_result'),state=q.get('state'),error_type=q.get('error_type'));break
except subprocess.TimeoutExpired as e:
 record['error_type']='TimeoutExpired'
 if e.stdout:(R/'partial.stdout.private').write_bytes(e.stdout)
 if e.stderr:(R/'partial.stderr.private').write_bytes(e.stderr)
except BaseException as e:
 record.update(client_error_type=type(e).__name__,client_error=str(e)[:160])
finally:write('receipt.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ['client_error']}))
