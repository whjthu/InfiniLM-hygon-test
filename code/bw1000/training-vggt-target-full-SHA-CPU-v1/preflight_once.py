"""Read exact target terminal metadata and namespace, without weight payload."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;old=D.parent/'training-vggt-original-public-model-v1';original=json.loads((old/'transfer-outcome.json').read_bytes());assert original['passed']
remote=('ORIGINAL='+repr(original)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,socket,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results');old=R/'training-vggt-original-public-model-v1';W=R/'training-vggt-target-full-SHA-CPU-v1';assert not W.exists()
raw=(old/'outcome.json').read_bytes();assert len(raw)<=65536;out=json.loads(raw);assert out==ORIGINAL['receiver_result'] and out['passed']
intent=json.loads((old/'intent.json').read_bytes());assert all(intent[k]==out[k]==ORIGINAL[k] for k in ('claim','request_id','epoch','deadline'))
p=old/'model/model.pt';s=p.stat();assert p.is_file() and not p.is_symlink() and s.st_size==5026874952
parent=subprocess.run(['squeue','-h','-j','1087656','-o','%T|%D'],capture_output=True,text=True,timeout=12);assert parent.returncode==0 and parent.stdout.strip()=='RUNNING|40'
receiver_owner=out['owner'];same_host=receiver_owner['host']==socket.gethostname();owner_present=None
if same_host:
 try:owner_present=int(Path('/proc',str(receiver_owner['PID']),'stat').read_text().split(') ',1)[1].split()[19])==receiver_owner['startticks']
 except (FileNotFoundError,ProcessLookupError):owner_present=False
 assert owner_present is False
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,parent_state='RUNNING',parent_nodes=40,original_receiver_outcome_matches_local_terminal=True,original_receiver_outcome_SHA256=hashlib.sha256(raw).hexdigest(),receiver_owner_same_host=same_host,original_receiver_identity_present=owner_present,file_stat=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev),weight_file_bytes_read=0,Torch_imported=False,GPU_calls=0)))
'''
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
with (D/'preflight-intent.json').open('x') as f:json.dump(dict(read_only=True,payload_SHA256=hashlib.sha256(remote).hexdigest()),f)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=60)
for name,raw in [('preflight.stdout.private.json',r.stdout),('preflight.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/'preflight-result.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),namespace_absent=v.get('namespace_absent'),parent_nodes=v.get('parent_nodes'),original_receiver_identity_present=v.get('original_receiver_identity_present'),weight_file_bytes_read=0)))
