"""Original namespace and exact source/metadata admission without Torch."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-v2-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
base=claim+':code/bw1000/training-llava-A100-native-contract-CPU-v1/'
bound={n:json.loads(subprocess.check_output(git+['show',base+n])) for n in ('ORIGINAL_SOURCES.json','MODEL_METADATA.json')}
remote=('BOUND='+repr(bound)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,importlib.metadata as md,json,sys
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-llava-native-update-contract-CPU-v1');assert not W.exists()
for name,pin in BOUND['ORIGINAL_SOURCES.json']['files'].items():
 p=Path(md.distribution('ms-swift').locate_file(name));assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
meta=BOUND['MODEL_METADATA.json'];root=Path(meta['model_root'])
for name,pin in meta['files'].items():
 p=root/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,source_files=5,metadata_files=14,Torch_imported=False,GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('preflight-v2.stdout.private.json',r.stdout),('preflight-v2.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(claim=claim,transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'preflight-v2-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),namespace_absent=v.get('namespace_absent'),source_files=v.get('source_files'),metadata_files=v.get('metadata_files'),Torch_imported=False,GPU_calls=0)))
