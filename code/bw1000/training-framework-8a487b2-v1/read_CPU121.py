"""Independently audit the exact new CPU step, runtime fields and source bytes."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
task=json.loads((BASE/'task.json').read_bytes());ns={'__file__':str(Path(__file__).resolve())}
prelude=(BASE.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8')
exec(compile(prelude.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
code=r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-native-dropout-CPU-r2')
raw=(root/'outcome.json').read_bytes();out=json.loads(raw);intent=json.loads((root/'intent.json').read_bytes());owner=json.loads((root/'owner.json').read_bytes())
assert out['claim_commit']=='a7f6b958934fe018bd82c6e6aa8436cc916051c3' and out['returncode']==0
assert out['deadline']-out['epoch']==180 and out['wall_seconds']<180 and out['GPU_allocation']==0
assert out['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw']
assert out['accounting']==[['1087656.121','training-vlm-native-dropout-CPU-r2','COMPLETED','0:0','4']]
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656.121','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],text=True,timeout=15).strip().splitlines()]
assert rows==out['accounting']
for name,pin in intent['files'].items():
    assert pin==TASK['source_SHA256'][name]
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==pin
result_raw=(root/'CPU-result.json').read_bytes();result=json.loads(result_raw)
assert hashlib.sha256(result_raw).hexdigest()==out['CPU-result.json']['SHA256'] and result==out['CPU-result.json']['value']
assert result['tests']==4 and result['successful'] and result['failures']==result['errors']==0 and not result['device_initialized']
assert result['torch']=='2.10.0' and result['torch_distribution']=='2.10.0+das.opt1.dtk2604.2606161421.g91a580'
assert not result['GPU_kernel_qualification'] and not result['training_qualification']
assert hashlib.sha256((root/'CPU.log').read_bytes()).hexdigest()==out['log_SHA256']
live=False
try:
    values=(Path('/proc')/str(owner['PID'])/'stat').read_text().split(') ',1)[1].split();live=int(values[19])==owner['startticks'] and values[0]!='Z'
except (FileNotFoundError,ProcessLookupError):pass
assert not live
value={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'TERMINAL_CPU121_DTK_NATIVE_DROPOUT_CONTRACTS_PASS','outcome_SHA256':hashlib.sha256(raw).hexdigest(),'actual_accounting_columns':out['accounting_columns'],'actual_accounting':rows,'source_files_rehashed':len(intent['files']),'CPU_result_SHA256':hashlib.sha256(result_raw).hexdigest(),'result':result,'owned_PID':owner,'owned_live':False,'GPU_allocation':0}
print(json.dumps(value))
'''
r=subprocess.run(ns['ssh'],input=('TASK='+repr(task)+'\n'+code).encode(),capture_output=True,timeout=50)
assert r.returncode==0,r.stderr.decode(errors='replace')[-800:]
v=json.loads(r.stdout)
with (BASE/'independent-original-CPU121.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k!='result'}))
