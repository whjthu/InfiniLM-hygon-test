
import hashlib,json,os,socket,sys
from pathlib import Path
R=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-initial-trainables-CPU-v1');task=json.loads(sys.argv[1]);v=json.loads((R/'outcome.json').read_bytes());intent=json.loads((R/'intent.json').read_bytes());owner=json.loads((R/'owner.json').read_bytes())
assert v['claim_commit']==intent['claim_commit']=='c9aa94a159a606c7dc2c66881affa8f0287274d2'
assert v['returncode']==0 and not v['forced_cleanup'] and v['deadline']==v['epoch']+180 and v['wall_seconds']<180
for n,h in intent['files'].items():assert h==task['source_SHA256'][n] and hashlib.sha256((R/n).read_bytes()).hexdigest()==h
raw=(R/'CPU-result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==v['CPU-result.json']['SHA256']=='640d71517a3245cc19e61f3423affe1bdf8f018e81925a692b0fcd92fcbec96b'
r=json.loads(raw);assert r['tests_run']==4 and r['failures']==r['errors']==0 and r['malformed_fixture_rejections']==20
assert r['callback_registry_imported'] and not r['GPU_initialized'] and not r['original_model_or_weight_loaded']
assert hashlib.sha256((R/'CPU.log').read_bytes()).hexdigest()==v['CPU_log_SHA256']
assert socket.gethostname()==owner['host'],'do not infer ownedPID exit on another host'
p=Path('/proc')/str(owner['PID'])/'stat';live=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==owner['startticks'];assert not live
out=dict(independent_passed=True,original_result_SHA256=hashlib.sha256(raw).hexdigest(),outcome_SHA256=hashlib.sha256((R/'outcome.json').read_bytes()).hexdigest(),intent_SHA256=hashlib.sha256((R/'intent.json').read_bytes()).hexdigest(),source_files_verified=len(intent['files']),owned_PID_gone=True,original_CPU=v,reader_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Model_GO=False,GPU_initialized=False)
with (R/'independent-CPU.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
