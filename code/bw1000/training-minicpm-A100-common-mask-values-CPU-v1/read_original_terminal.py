from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-minicpm-A100-common-mask-values-CPU-v1';A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');o=json.loads((D/'transport-outcome.json').read_bytes());assert o['transport_rc']==0 and o['remote_exec_acknowledged'] and o['value']['passed'];manifest=json.loads((D/'SOURCE_FILES.json').read_bytes())
E=dict(claim=o['claim'],request=o['request_id'],source={n:manifest[n] for n in ('vlm/runtime/first_update_audit.py','vlm/runtime/initial_trainables.py','PLAN.json','audit_values.py')},result=o['value']['CPU-result.json']['SHA256'],progress=o['value']['progress.json']['SHA256'])
remote=r'''from pathlib import Path
import datetime,hashlib,json,socket
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-reference-values-CPU-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,pin in E['source'].items():assert sha(W/'source'/n)==pin
v=json.loads((W/'outcome.json').read_bytes());intent=json.loads((W/'intent.json').read_bytes());result=json.loads((W/'evidence/CPU-result.json').read_bytes());progress=json.loads((W/'evidence/progress.json').read_bytes())
assert v['passed'] and v['returncode']==0 and not v['forced_cleanup'] and not v['owned_child_alive'] and result['passed']
assert intent['claim']==v['claim']==E['claim'] and intent['request_id']==v['request_id']==E['request']
assert intent['host']==v['host']==result['owner']['host']==socket.gethostname()
assert intent['epoch']==v['epoch']==result['epoch'] and intent['deadline']==v['deadline']==v['epoch']+720 and result['deadline']==v['epoch']+600
assert sha(W/'CPU.log')==v['log_SHA256']
for name,k in [('CPU-result.json','result'),('progress.json','progress')]:
 p=W/'evidence'/name;assert sha(p)==v[name]['SHA256']==E[k] and p.stat().st_size==v[name]['bytes']
assert result['completed_value_files']==progress['completed_value_files']==136 and progress['completed_ranks']==8 and not progress['failed'] and progress['stage']=='complete'
assert result['module_SHA256']==E['source']['vlm/runtime/first_update_audit.py'] and result['plan_SHA256']==E['source']['PLAN.json']
assert result['CPU_tests']==0 and result['CPU_unit_tests_reused']==4 and not result['GPU_initialized'] and result['GPU_cards']==0 and result['raw_tensor_transfer_bytes']==0
assert result['peak_RSS_KiB']<=8*1024**2 and len(result['owner']['CPU_affinity'])==4
owned=v['owned_child'];assert result['owner']['PID']==owned['PID'] and result['owner']['startticks']==owned['startticks']
p=Path('/proc',str(owned['PID']),'stat');same=False
if p.exists():same=int(p.read_text().split(') ',1)[1].split()[19])==owned['startticks']
assert not same
ranks=result['rank_summaries'];assert len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8))
assert all(len(r['forwards'])==8 and len(r['updates'])==2 and r['at_least_one_nonzero_saved_parameter_update'] for r in ranks)
print(json.dumps(dict(independent_A100_values_verified=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pins=E['source'],outcome_SHA256=sha(W/'outcome.json'),intent_SHA256=sha(W/'intent.json'),result_SHA256=E['result'],progress_SHA256=E['progress'],log_SHA256=v['log_SHA256'],owned_child_absent=True,original=v,raw_tensor_transfer_bytes=0,GPU_cards=0,cross_platform_numerical_accepted=False,Model_GO=False)))
'''
with (D/'independent-read-intent.json').open('x') as f:json.dump(dict(scope='Exact A100 CPU values receipt/source/log/ownedPID independent closure; no repeated raw reads',expected=E),f)
cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];r=subprocess.run(cmd,input=('E='+repr(E)+'\n'+remote).encode(),capture_output=True,timeout=35)
if r.returncode:print(json.dumps(dict(rc=r.returncode,error_tail=r.stderr.decode(errors='replace')[-900:])));raise SystemExit(1)
v=json.loads(r.stdout)
with (D/'independent-original.json').open('x') as f:json.dump(v,f,indent=2)
res=v['original']['CPU-result.json']['value'];print(json.dumps(dict(independent_A100_values_verified=True,returncode=v['original']['returncode'],completed_value_files=res['completed_value_files'],peak_RSS_KiB=res['peak_RSS_KiB'],owned_child_absent=True,raw_tensor_transfer_bytes=0)))
