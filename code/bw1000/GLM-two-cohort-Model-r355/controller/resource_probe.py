"""Assigned host CPU-only resource metadata; never opens GPU runtime."""
import os,json,socket,datetime
from pathlib import Path
from contract import STAGES
node=socket.gethostname().split('.')[0];assert node in {n for _,n in STAGES}
k=Path('/sys/class/kfd/kfd/proc');assert k.is_dir();pids=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit())
with socket.socket()as s:
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try:s.bind(('127.0.0.1',43843));free=True
    except OSError:free=False
print(json.dumps(dict(node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU_initialized=False,
    KFD_pids=pids,KFD_empty=not pids,port43843_bindable=free)),flush=True)
assert not pids and free
