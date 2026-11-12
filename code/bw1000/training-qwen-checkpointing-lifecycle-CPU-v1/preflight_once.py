"""Check the new CPU namespace and exact original dependencies before execution."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
pins=json.loads(subprocess.check_output(git+['show',claim+':code/bw1000/training-qwen-checkpointing-lifecycle-CPU-v1/ORIGINAL_SOURCE_METADATA.json']))
runtime=json.loads(subprocess.check_output(git+['show',claim+':code/bw1000/training-qwen-checkpointing-lifecycle-CPU-v1/RUNTIME_SOURCE_PINS.json']))
remote=('PINS='+repr(pins)+'\nRUNTIME='+repr(runtime)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,importlib.metadata as metadata,json
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-checkpointing-lifecycle-CPU-v1')
assert not W.exists()
for tag,pin in PINS['files'].items():
 p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
for pin in RUNTIME['files'].values():
 p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,original_metadata_files=len(PINS['files']),versions={k:metadata.version(k) for k in ['torch','transformers','ms-swift','peft','accelerate']},GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
with (D/'preflight.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'preflight.stderr.private.txt').open('xb') as f:f.write(r.stderr)
result=dict(claim=claim,transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'preflight-result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=result.get('value',{}).get('passed'),namespace_absent=result.get('value',{}).get('namespace_absent'),GPU_calls=0)))
