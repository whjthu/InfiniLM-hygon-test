"""Call the pinned GLM InfiniCCL provider on all assigned TP16 devices."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from collective import ROWS
from provider import Provider, UID_BYTES
import sys
from verify_transport import mapped_providers
from bootstrap import PORT, exchange
import time

from assigned_kfd import assigned_kfd

root=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_bytes())
intent=read(root/'intent.json')
binding=read(root/'src/BINDING.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/'src'/name)==pin for name,pin in intent['source_files'].items())
rank=int(os.environ['SLURM_PROCID'])
local=int(os.environ['SLURM_LOCALID'])
node=socket.gethostname().split('.')[0]
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
assert rank==intent['nodes'].index(node)*8+local and 0<=local<8 and int(os.environ['SLURM_NTASKS'])==16
before=read(root/'pre'/f'{node}.json')
assert before['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert before['kfd_holder_count']==0
assert 0<=time.monotonic_ns()-before['monotonic_ns']<=705_000_000_000
assert time.time()<intent['work_deadline']
library=Path(binding['library'])
assert sha(library)==binding['SHA256']
provider=Provider()
cuda,ccl=provider.cuda,provider.ccl
P=C.c_void_p
I=C.c_int
S=C.c_size_t

class UniqueID(C.Structure):
    _fields_=[('internal',C.c_char*UID_BYTES)]

def bind(lib,name,*types):
    fn=getattr(lib,name)
    fn.argtypes=types
    fn.restype=I
    def call(*args):
        status=fn(*args)
        if status:
            raise RuntimeError(f'{name} returned {status}')
    return call

get_count=bind(cuda,'cudaGetDeviceCount',C.POINTER(I))
set_device=bind(cuda,'cudaSetDevice',I)
get_pci=bind(cuda,'cudaDeviceGetPCIBusId',C.POINTER(C.c_char),I,I)
meminfo=bind(cuda,'cudaMemGetInfo',C.POINTER(S),C.POINTER(S))
create_stream=bind(cuda,'cudaStreamCreate',C.POINTER(P))
destroy_stream=bind(cuda,'cudaStreamDestroy',P)
get_id=bind(ccl,'infinicclGetUniqueId',C.POINTER(UniqueID))
init=bind(ccl,'infinicclCommInitRank',C.POINTER(P),I,UniqueID,I)
destroy=bind(ccl,'infinicclCommDestroy',P)
count=I()
get_count(C.byref(count))
assert count.value==8
set_device(local)
pci_buf=C.create_string_buffer(32)
get_pci(pci_buf,32,local)
pci=pci_buf.value.decode().lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
    pci='0000:'+pci
assert pci in before['pci']
kfd=assigned_kfd(pci)
def memory():
    available,total=S(),S()
    meminfo(C.byref(available),C.byref(total))
    assert available.value>=8*1024**3 and total.value==68702699520
    return available.value
minimum=memory()
with (root/sys.argv[1]/f'{rank:02d}-entered.json').open('x') as out:
    json.dump(dict(rank=rank,node=node,pci=pci,kfd_id=kfd['gpu_id'],step=step),out)

arm,mode=sys.argv[1:]
assert re.fullmatch(r'arm-[0-5]',arm) and mode in {'socket','shca'}
assert os.environ['NCCL_NET']=={'socket':'Socket','shca':'IBext_v8'}[mode]
work=root/arm
def mark(stage,**extra):
    print(json.dumps(dict(rank=rank,stage=stage,elapsed_ns=time.monotonic_ns(),**extra)),flush=True)

def make_uid():
    uid=UniqueID()
    mark("get_id.enter")
    get_id(C.byref(uid))
    mark("get_id.return")
    return bytes(uid)

address=socket.gethostbyname(intent['nodes'][0])
seconds=min(35,intent['work_deadline']-time.time())
assert seconds>0
raw=exchange(rank,address,PORT,bytes.fromhex(intent['bootstrap_key']),step,
             time.monotonic()+seconds,make_uid,mark)
uid=UniqueID.from_buffer_copy(raw)
comm,stream=P(),P()
checks=[]

def require_time():
    if time.time() >= intent['work_deadline']:
        raise TimeoutError('original collective deadline elapsed')

try:
    mark('comm_init.enter')
    init(C.byref(comm),16,uid,rank)
    mark('comm_init.return')
    assert comm.value
    create_stream(C.byref(stream))
    assert stream.value
    for rows in ROWS:
        checks.append(provider.run(comm,stream,rank,rows,require_time))
        minimum=min(minimum,memory())
        if rows==ROWS[0]:
            # The bootstrap verifier needs one mark from every rank.
            mark('first_collective.return')
    providers=mapped_providers(Path('/proc/self/maps').read_text(),mode)
finally:
    if stream.value:
        provider.sync_stream(stream)
    if comm.value:
        destroy(comm)
    if stream.value:
        destroy_stream(stream)
minimum=min(minimum,memory())
with (work/f'{rank:02d}.json').open('x') as out:
    json.dump(dict(passed=True,rank=rank,node=node,pci=pci,kfd_id=kfd['gpu_id'],step=step,
                   mode=mode,providers=providers,library_SHA256=binding['SHA256'],uid_SHA256=hashlib.sha256(raw).hexdigest(),
                   checks=checks,min_free_bytes=minimum,comm_destroyed=True),out,indent=2)
