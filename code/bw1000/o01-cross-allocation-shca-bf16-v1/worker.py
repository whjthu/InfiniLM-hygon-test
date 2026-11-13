"""Validate and time two-rank BF16 all-reduce on assigned HIP devices."""
import ctypes
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import sys
import time

root=Path(sys.argv[1])
intent=json.loads((root/'intent.json').read_bytes())
node=socket.gethostname().split('.')[0]
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
assert node in intent['nodes'] and step.split('.')[0]==intent['parents_by_node'][node]
assert time.time()<intent['work_deadline']
from snapshot import drm_inventory
before=json.loads((root/'pre'/f'{node}.json').read_bytes())
assert before['pci']==drm_inventory()
assert before['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
rank=int(os.environ['O01_GLOBAL_RANK'])
assert rank==intent['nodes'].index(node) and os.environ['SLURM_NTASKS']=='1' and os.environ['SLURM_PROCID']=='0'
pins=json.loads((root/'src/PROVIDERS.json').read_bytes())['files']
for path,pin in pins.items():
    raw=Path(path).read_bytes()
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'], path
from trace_provider import capture, open_from_mapped_verbs

def trace(stage, **details):
    path=root/'traces'/f'{rank}-{stage}.json'
    value=dict(rank=rank,node=node,step=step,stage=stage,providers=capture(),**details)
    with path.open('x') as f:
        json.dump(value,f,indent=2)

alias_target=Path('/rdma-provider-stack/libibverbs.so.1').resolve(strict=True)
for soname in ('libibverbs.so','libibverbs.so.1'):
    assert (root/'verbs-aliases'/soname).resolve(strict=True)==alias_target
assert os.environ['NCCL_NET']=='IBext_v8' and os.environ['NCCL_NET_PLUGIN']=='shca'
trace('before-torch')
import torch
import torch.distributed as dist
trace("after-torch")
assert torch.version.hip and torch.cuda.device_count()==1
hip=ctypes.CDLL('libamdhip64.so')
pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,len(pci),0)==0
name=pci.value.decode().lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',name):
    name='0000:'+name
assert name in before['pci']
from assigned_kfd import assigned_kfd
kfd=assigned_kfd(name)
torch.cuda.set_device(0)
trace('before-group')
rows=[]
stage='group'
try:
    dist.init_process_group(backend='nccl',init_method=(root/'rendezvous').as_uri(),
                            rank=rank,world_size=2,timeout=timedelta(seconds=35))
    trace('after-group')
    stage='first-allreduce'
    for elements in [4096, 65536, 2097152]:
        offsets=torch.arange(elements,device='cuda',dtype=torch.int32)%7
        original=(offsets+rank+1).to(torch.bfloat16)
        expected=(offsets*2+3).to(torch.bfloat16)
        values=torch.empty_like(original)
        timings=[]
        for sample in range(25):
            assert time.time()<intent['work_deadline']
            values.copy_(original)
            torch.cuda.synchronize()
            start=time.monotonic_ns()
            dist.all_reduce(values)
            torch.cuda.synchronize()
            elapsed=time.monotonic_ns()-start
            if elements==4096 and sample==0:
                trace('first-allreduce-returned')
                stage='later-allreduce'
            assert torch.equal(values,expected)
            if sample>=5:
                timings.append(elapsed)
        rows.append(dict(elements=elements,bytes=elements*2,checked=25,latency_ns=timings))
    assert dist.get_backend()=='nccl'
    dist.barrier(device_ids=[0])
    mapped={line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines()}
    assert set(pins)<=mapped
    assert not any('/usr/lib/' in p and '/libibverbs.so' in p for p in mapped)

except BaseException as exc:
    trace('failure',failed_stage=stage,error_type=type(exc).__name__)
    with (root/'traces'/f'{rank}-mapped-verbs-open.json').open('x') as f:
        json.dump(dict(rank=rank,node=node,step=step,opened=open_from_mapped_verbs()),f,indent=2)
    raise
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
result=dict(node=node,rank=rank,step=step,pci=name,kfd=kfd,rows=rows,backend='nccl',HIP=torch.version.hip,
            Torch=torch.__version__,GPU_name=torch.cuda.get_device_name(0),boot_id=before['boot_id'],mapped_provider_pins=pins)
with (root/'ranks'/f'{rank}.json').open('x') as f:
    json.dump(result,f,indent=2)
