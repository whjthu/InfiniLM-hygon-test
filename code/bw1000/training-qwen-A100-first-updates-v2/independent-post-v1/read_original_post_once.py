"""Independently verify original GPU owners,device identity and port release."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
original=json.loads((D/'transport-outcome.json').read_bytes());expected=original['value']['outcome.json']
remote=('EXPECTED_SHA='+repr(expected['SHA256'])+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,socket,subprocess
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-A100-first-updates-v2');raw=(W/'outcome.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED_SHA
outcome=json.loads(raw);binding=json.loads((W/'src/GPU_BINDING.json').read_bytes());owners={}
for key,pin in outcome['owners'].items():
 try:current=int(Path('/proc/%d/stat'%int(key)).read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:current=None
 owners[key]=dict(original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks']);assert current!=pin['startticks']
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=12).strip().splitlines()
rows=[[v.strip() for v in line.split(',')] for line in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])]
assert len(rows)==8 and {r[0]:r[1] for r in rows}==binding['UUID_by_index'] and {r[0]:r[2] for r in rows}==binding['PCI_by_index']
apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']);assert not apps
assert all(int(r[5])<=16 for r in rows)
with socket.socket() as s:s.bind(('127.0.0.1',29895))
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_outcome_SHA256=EXPECTED_SHA,
 original_native_returncode=outcome['native_returncode'],original_keeper_passed=outcome['keeper_passed'],
 observed_original_owners=len(owners),owners=owners,original_owners_absent=True,UUID_and_PCI_exact=True,GPUs=rows,compute_apps=apps,master_port_bindable=True,
 model_result_changed=False,GPU_launches=0,qualification='Independent cleanup only; original training result is independently recorded.')))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
with (D/'original-post.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'original-post.stderr.private.txt').open('xb') as f:f.write(r.stderr)
result=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'independent-original-post.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),original_owners_absent=v.get('original_owners_absent'),observed_original_owners=v.get('observed_original_owners'),compute_apps=len(v.get('compute_apps',[])),master_port_bindable=v.get('master_port_bindable'))))
