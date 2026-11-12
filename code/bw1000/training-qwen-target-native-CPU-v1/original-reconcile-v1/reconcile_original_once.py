"""Reconcile original request after channel loss; never invoke its receiver again."""
from pathlib import Path
import hashlib,json,subprocess,sys
D=Path(__file__).resolve().parent;B=D.parent
intent=json.loads((D/'launch-intent.json').read_bytes());expected={k:intent[k] for k in ('claim','request_id')}
remote=('EXPECTED='+repr(expected)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-target-native-CPU-v1')
rows=[r.split('|') for r in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if '|training-qwen-target-native-CPU-v1|' in r]
result=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request_id=EXPECTED['request_id'],claim=EXPECTED['claim'],namespace_exists=W.exists(),accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'],accounting=rows,replay_allowed=False)
for name in ('intent.json','owned-srun.json','outcome.json','evidence/CPU-result.json','evidence/failure-private.json'):
 p=W/name
 if not p.is_file():continue
 assert 0<p.stat().st_size<=2*1024**2
 raw=p.read_bytes();value=json.loads(raw)
 if name in ('intent.json','outcome.json'):assert all(value[k]==EXPECTED[k] for k in ('claim','request_id'))
 result[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=value)
print(json.dumps(result))
'''
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
tag=sys.argv[1];assert tag.isalnum()
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=60)
for name,raw in [('reconcile-'+tag+'.stdout.private.json',r.stdout),('reconcile-'+tag+'.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/('reconciliation-'+tag+'.json')).open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});o=v.get('outcome.json',{}).get('value',{});cpu=v.get('evidence/CPU-result.json',{}).get('value',{})
print(json.dumps(dict(transport_rc=r.returncode,namespace_exists=v.get('namespace_exists'),accounting=v.get('accounting'),outcome_present='outcome.json' in v,passed=o.get('passed'),CPU_phase=cpu.get('phase'),error_type=cpu.get('error_type'),replay_allowed=False)))
