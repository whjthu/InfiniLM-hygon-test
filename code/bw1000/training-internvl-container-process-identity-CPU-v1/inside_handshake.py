"""Stdlib handshake only: no Torch, model, processor, data or weights."""
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);deadline=float(sys.argv[2]);remaining=deadline-time.time()
if not 2.1<=remaining<=20:raise SystemExit('HANDSHAKE_DEADLINE')
if any(os.environ.get(k)!='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')):raise SystemExit('VISIBLE_DEVICE_MASK')
mono=time.monotonic()+remaining
v=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
record=dict(PID=os.getpid(),PPID=os.getppid(),PGID=os.getpgrp(),startticks=int(v[19]),UID=os.getuid(),Torch_imported='torch' in sys.modules,phase='STDLIB_HANDSHAKE')
with (root/'handshake.json').open('x') as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
end=min(mono,time.monotonic()+2)
while time.monotonic()<end:time.sleep(min(.05,end-time.monotonic()))
with (root/'handshake-done.json').open('x') as f:json.dump(dict(record,phase='STDLIB_NORMAL_EXIT'),f);f.flush();os.fsync(f.fileno())
