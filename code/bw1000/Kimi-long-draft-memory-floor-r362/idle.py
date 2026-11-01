from pathlib import Path
import datetime,json,os,socket

root=Path('/sys/class/kfd/kfd/proc');assert root.is_dir()
pids=sorted(int(p.name)for p in root.iterdir()if p.name.isdigit())
owned=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit()or int(p.name)==os.getpid():continue
    try:cmd=(p/'cmdline').read_bytes()
    except(OSError,PermissionError):continue
    if b'/Kimi-long-draft-memory-floor-r362/'in cmd and(b'worker.py'in cmd or b'pressure.py'in cmd):owned.append(int(p.name))
bind={}
for port in [46947,46962]:
    with socket.socket()as s:
        try:s.bind(('0.0.0.0',port));bind[str(port)]=True
        except OSError:bind[str(port)]=False
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname().split('.')[0],
    actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,
    TCP_connections_made=0,kfd_pids=pids,owned_PIDs=owned,ports_bindable=bind,passed=not pids and not owned and all(bind.values()))))
