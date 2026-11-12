"""Read exact file stat differences and host context; no model bytes."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
remote=b'''
from pathlib import Path
import hashlib,json,socket
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vggt-target-full-SHA-CPU-v1');out=json.loads((W/'outcome.json').read_bytes());p=Path(out['model_path']);s=p.stat();now=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode)
print(json.dumps(dict(current_host=socket.gethostname(),hash_owner_host=out['owner']['host'],recorded_stat=out['after_stat'],current_stat=now,differences={k:[out['after_stat'][k],v] for k,v in now.items() if out['after_stat'][k]!=v},directory_names=sorted(x.name for x in p.parent.iterdir()),symlink=p.is_symlink(),weight_file_bytes_read=0)))
'''
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=60)
for n,raw in [('stat-diagnosis.stdout.private.json',r.stdout),('stat-diagnosis.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
result=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'stat-diagnosis-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
