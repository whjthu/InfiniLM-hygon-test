"""Independent CPU-only KFD/PID/43843 probe. No connection/GPU import."""
from pathlib import Path
import datetime,json,os,socket
node=socket.gethostname().split('.')[0];assert node in ['a02r4n12','a02r4n13']
kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir()
pids=sorted(int(p.name) for p in kfd.iterdir() if p.name.isdigit());matching=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if any(x.endswith(b'/GLM-C1-PP2-constructor-r395/GPU_entry.py') for x in args):matching.append(int(proc.name))
with socket.socket() as s:
    try:s.bind(('0.0.0.0',43843));available=True
    except OSError:available=False
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
       CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=pids,matching_component_processes=matching,ports_bindable={'43843':available},TCP_connections_made=0)
print(json.dumps(q));assert not pids and not matching and available
