"""Record host KFD and DRM metadata before entering the assigned container."""
import json,os,socket,sys,time
from pathlib import Path
from snapshot import snapshot
root=Path(sys.argv[1])
phase=sys.argv[2]
assert phase in {'pre','gpu-gate','post'}
row=snapshot(False)
assert row['kfd_holder_count']==0
if phase!='pre':
    before=json.loads((root/'pre'/f"{row['node']}.json").read_bytes())
    assert before['boot_id']==row['boot_id'] and before['pci']==row['pci']
    if phase=='gpu-gate':
        assert 0<=row['monotonic_ns']-before['monotonic_ns']<=90_000_000_000
with (root/phase/f"{row['node']}.json").open('x') as f:
    json.dump(row,f,indent=2)
if phase=='gpu-gate':
    assert len(sys.argv)>3
    os.execv(sys.argv[3],sys.argv[3:])
