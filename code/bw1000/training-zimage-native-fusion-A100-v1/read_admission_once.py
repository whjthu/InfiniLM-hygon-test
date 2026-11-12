"""One readonly single-card reference admission; no Torch or model execution."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
code=b'''from pathlib import Path
import datetime,json,os,shutil,signal,socket,subprocess,sys
signal.alarm(30)
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=10).strip().splitlines()
cards=[[v.strip() for v in line.split(',')] for line in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]
apps=[[v.strip() for v in line.split(',')] for line in query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]
eligible=[c for c in cards if c[0] in ('2','3','4','5','6','7') and c[3]=='NVIDIA A100-SXM4-40GB' and int(c[5])<=16 and int(c[6])==0 and not any(a[0]==c[1] for a in apps)]
mem={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
w=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-native-fusion-A100-v1')
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),cards=cards,compute_apps=apps,eligible_cards=eligible,selected=eligible[0] if eligible else None,
 namespace_absent=not w.exists(),available_affinity_CPUs=len(os.sched_getaffinity(0)),host_memory=mem,free_disk_bytes=shutil.disk_usage(w.parent).free,
 Torch_imported='torch' in sys.modules,GPU_launched=False)))
'''
start=time.monotonic();r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -'],input=code,capture_output=True,timeout=45)
(D/'admission.stdout.private.json').write_bytes(r.stdout);(D/'admission.stderr.private.txt').write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
with (D/'admission-result.json').open('x') as f:json.dump(dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,value=v,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest()),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,eligible_cards=[c[0] for c in v['eligible_cards']] if v else None,selected_index=v['selected'][0] if v and v['selected'] else None,
 namespace_absent=v['namespace_absent'] if v else None,host_memory=v['host_memory'] if v else None,GPU_launched=False)))
