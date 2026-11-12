"""Read only the original sender's exact terminal artifacts and process identity."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');old=D.parent/'training-vggt-original-public-model-v1'
local=json.loads((old/'transfer-outcome.json').read_bytes());assert local['passed'] and local['source_rc']==0
remote=('ORIGINAL='+repr({k:local[k] for k in ('claim','request_id','epoch','deadline')})+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,socket,sys
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-original-public-model-v1');files={};values={}
for name in ('intent.json','outcome.json'):
 p=W/name;assert p.is_file() and p.stat().st_size<=65536;raw=p.read_bytes();values[name]=json.loads(raw);files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
intent=values['intent.json'];out=values['outcome.json']
assert all(out[k]==intent[k]==ORIGINAL[k] for k in ('claim','request_id','epoch','deadline'))
assert out['passed'] and out['weight_bytes']==5026874952 and out['files']=={'model.pt':{'bytes':5026874952,'SHA256':'d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}}
owner=out['owner'];assert owner==intent['owner'] and owner['PID']==2894680 and owner['startticks']==687131943 and owner['host']==socket.gethostname()
try:now=int(Path('/proc',str(owner['PID']),'stat').read_text().split(') ',1)[1].split()[19])
except (FileNotFoundError,ProcessLookupError):now=None
assert now!=owner['startticks'] and 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,original_source_outcome=out,original_source_identity_present=False,weight_payload_bytes_read=0,Torch_imported=False,GPU_cards=0)))
'''
with (D/'source-terminal-v2-read-intent.json').open('x') as f:json.dump(dict(read_only=True,payload_SHA256=hashlib.sha256(remote).hexdigest()),f)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for n,raw in [('source-terminal-v2.stdout.private.json',r.stdout),('source-terminal-v2.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/'source-terminal-v2-result.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,source_terminal_passed=v.get('passed'),original_source_identity_present=v.get('original_source_identity_present'),weight_payload_bytes_read=0)))
