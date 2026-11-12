"""Read exact model metadata and namespace state, without reading weight bytes."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
metadata=json.loads(subprocess.check_output(git+['show',claim+':code/bw1000/training-llava-original-model-identity-v1/ORIGINAL_MODEL_METADATA.json']))
remote=('EXPECTED='+repr(metadata)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-llava-original-model-identity-v1');assert not W.exists()
for tag in ('config','index'):
 pin=EXPECTED['files'][tag];p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
root=Path(EXPECTED['files']['index']['path']).parent
for pin in EXPECTED['shards']:
 p=root/pin['name'];s=p.stat();assert not p.is_symlink() and p.is_file() and s.st_size==pin['bytes'] and s.st_mtime_ns==pin['mtime_ns'] and s.st_ino==pin['inode'] and s.st_dev==pin['device']
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,metadata_files=2,shard_stats=4,weight_bytes_read=False,GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
with (D/'preflight.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'preflight.stderr.private.txt').open('xb') as f:f.write(r.stderr)
value=dict(claim=claim,transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:value['value']=json.loads(r.stdout)
with (D/'preflight-result.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=value.get('value',{}).get('passed'),namespace_absent=value.get('value',{}).get('namespace_absent'),weight_bytes_read=False)))
