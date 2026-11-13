"""Validate loader closure first, then acquire a fresh same-step HOST lease."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from scope import ROOT,INSIDE_SECONDS,EpochClock,PREFLIGHT_SECONDS
from snapshot import snapshot
assert sys.argv[1]=='--' and len(sys.argv)>2
args=sys.argv[2:];assert args[-1]=='device_entry.py'
i=json.loads((ROOT/'intent.json').read_bytes());clock=EpochClock(i['epoch'])
masked=os.environ.copy();masked.update(HIP_VISIBLE_DEVICES='-1',CUDA_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
pre=args[:-1]+['link_preflight.py'];r=subprocess.run(pre,env=masked,timeout=clock.left(PREFLIGHT_SECONDS,10));assert r.returncode==0
clock.left(PREFLIGHT_SECONDS,5);raw=(ROOT/'link-preflight.json').read_bytes();v=json.loads(raw);assert v['passed'] and not v['CUDA_initialized']
before=json.loads((ROOT/'before.json').read_bytes());gate=snapshot(False)
assert gate['node']==before['node'] and gate['boot_id']==before['boot_id'] and gate['pci']==before['pci'] and gate['kfd_holder_count']==0
assert i['epoch']<=time.time()<i['epoch']+INSIDE_SECONDS
s=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
gate.update(host_UID=os.getuid(),host_PID=os.getpid(),host_PGID=os.getpgrp(),host_startticks=int(s[19]),before_SHA256=hashlib.sha256((ROOT/'before.json').read_bytes()).hexdigest(),link_preflight_SHA256=hashlib.sha256(raw).hexdigest(),context='Fresh actual same-step HOST process/proc/KFD view after masked library validation,before container GPU exec.')
with (ROOT/'lease-before.json').open('x') as f:json.dump(gate,f,indent=2)
os.execvp(args[0],args)
