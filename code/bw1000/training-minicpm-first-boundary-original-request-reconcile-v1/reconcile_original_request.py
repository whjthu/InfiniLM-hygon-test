"""Read exact original request/namespace/accounting; never replay its bootstrap."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;B=D.parent;o=json.loads((D/'transport-outcome.json').read_bytes());launch=json.loads((D/'launch-intent.json').read_bytes())
assert o['request_id']==launch['request_id'] and launch['exec_request_attempted'] and not o['replay_allowed']
remote=('CLAIM='+repr(launch['claim_commit'])+'\nREQUEST='+repr(launch['request_id'])+'\n').encode()+br'''
from pathlib import Path
import datetime,hashlib,json,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native-first-boundary-pair-v1');label=W.name;files={};owner_status=None
for name in ('intent.json','keeper-owner.json','controller-ready.json','outcome.json'):
 p=W/name
 if p.is_file():
  assert 0<p.stat().st_size<=2*1024**2;raw=p.read_bytes();files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
if 'intent.json' in files:
 v=files['intent.json']['value'];assert v['claim_commit']==CLAIM and v['request_id']==REQUEST
if 'keeper-owner.json' in files:
 v=files['keeper-owner.json']['value'];assert v['claim_commit']==CLAIM and v['request_id']==REQUEST
 same=socket.gethostname()==v['host'];present=None
 if same:
  p=Path('/proc/%d/stat'%v['PID']);present=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==v['startticks'] and p.read_text().split(') ',1)[1].split()[0]!='Z'
 owner_status=dict(same_host=same,original_identity_alive=present)
rows=[r.split('|') for r in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if any('|'+label+s+'|' in r for s in ('','-pre','-post','-audit'))]
active=[r for r in subprocess.check_output(['squeue','--steps','-h','-j','1087656','-o','%i|%j|%N'],text=True,timeout=15).splitlines() if label in r]
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,namespace_exists=W.exists(),source_directory_exists=(W/'src').exists(),files=files,owner_status=owner_status,accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'],accounting=rows,matching_active_steps=active,replay_allowed=False,new_Slurm_steps=0,scope='Readonly originalrequest reconciliation;missingreceipt/namespace/accounting alone isnotproof ofnonexecution.')))
'''
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_cluster_SSH','exec'),ns)
tag=str(time.time_ns());folder=D/'reconciliation';folder.mkdir(exist_ok=True);start=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=60)
for suffix,raw in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
 with (folder/(tag+'.'+suffix)).open('xb') as f:f.write(raw)
v=dict(claim=launch['claim_commit'],request_id=launch['request_id'],transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),new_execution_attempted=False,replay_allowed=False)
if r.returncode==0:v['value']=json.loads(r.stdout)
with (folder/(tag+'-receipt.json')).open('x') as f:json.dump(v,f,indent=2)
value=v.get('value',{});print(json.dumps(dict(receipt=tag+'-receipt.json',transport_rc=r.returncode,namespace_exists=value.get('namespace_exists'),original_intent_present='intent.json' in value.get('files',{}),owner_status=value.get('owner_status'),accounting=value.get('accounting'),new_execution_attempted=False,replay_allowed=False)))
