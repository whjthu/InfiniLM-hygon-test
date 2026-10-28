"""Assigned CPU-only resource metadata; no GPU runtime or key/env contents."""
import pathlib,json,os,socket,datetime
NODES={'a02r4n12','a02r4n13','a02r4n14','a02r4n15'};node=socket.gethostname().split('.')[0];assert node in NODES
k=pathlib.Path('/sys/class/kfd/kfd/proc');assert k.is_dir();pids=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit())
available=None
for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
 if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as s:
 # A closed old service may leave TIME_WAIT. It is not an active listener.
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:s.bind(('127.0.0.1',43843));port_free=True
 except OSError:port_free=False
row={'node':node,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_runtime_initialized':False,'kfd_pids':pids,'KFD_empty':not pids,'port43843_bindable':port_free,'affinity_CPU_count':len(os.sched_getaffinity(0)),'logical_CPU_count':os.cpu_count(),'MemAvailable_bytes':available}
print(json.dumps(row),flush=True);assert not pids and port_free and row['logical_CPU_count']>=64
