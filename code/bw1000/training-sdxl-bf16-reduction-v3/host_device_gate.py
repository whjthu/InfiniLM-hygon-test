"""Run process/KFD-holder admission in the host UID view of this GPU step."""
from pathlib import Path
import hashlib,json,os,socket,sys,time
from snapshot import snapshot
W=Path(__file__).resolve().parents[1]
assert len(sys.argv)>2 and sys.argv[1]=='--'
intent=json.loads((W/'intent.json').read_bytes());assert intent['epoch']<=time.time()<intent['epoch']+660
before=json.loads((W/'before.json').read_bytes());gate=snapshot(False);assert gate['node']==before['node'] and gate['boot_id']==before['boot_id'] and gate['pci']==before['pci'] and gate['kfd_holder_count']==0 and 0<=time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000
gate.update(host_UID=os.getuid(),host_PID=os.getpid(),host_PGID=os.getpgrp(),host_startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),before_SHA256=hashlib.sha256((W/'before.json').read_bytes()).hexdigest(),context='Actual host process/proc/KFD view inside the same one-GPU Slurm step; before container exec.')
with (W/'lease-before.json').open('x') as f:json.dump(gate,f,indent=2)
os.execvp(sys.argv[2],sys.argv[2:])
