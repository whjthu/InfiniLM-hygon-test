"""Call the pinned GLM InfiniCCL provider on all assigned TP16 devices."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import struct
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
cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12')
ccl=C.CDLL(str(library))
P=C.c_void_p
I=C.c_int
S=C.c_size_t

class UniqueID(C.Structure):
    _fields_=[('internal',C.c_char*136)]

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
alloc=bind(cuda,'cudaMalloc',C.POINTER(P),S)
free=bind(cuda,'cudaFree',P)
copy=bind(cuda,'cudaMemcpy',P,P,S,I)
sync=bind(cuda,'cudaDeviceSynchronize')
get_id=bind(ccl,'infinicclGetUniqueId',C.POINTER(UniqueID))
init=bind(ccl,'infinicclCommInitRank',C.POINTER(P),I,UniqueID,I)
destroy=bind(ccl,'infinicclCommDestroy',P)
reduce=bind(ccl,'infinicclAllReduce',P,P,S,I,I,P,P)
gather=bind(ccl,'infinicclAllGather',P,P,S,I,P,P)
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
comm,src,dst=P(),P(),P()
checks=[]
def bf16(value):
    return struct.pack('<H',struct.unpack('<I',struct.pack('<f',value))[0]>>16)
try:
    mark('comm_init.enter')
    init(C.byref(comm),16,uid,rank)
    mark('comm_init.return')
    assert comm.value
    alloc(C.byref(src),6144*2)
    alloc(C.byref(dst),6144*2*16)
    for elements in (1024,6144):
        payload=bf16(rank+1)*elements
        host=C.create_string_buffer(payload)
        copy(src,host,len(payload),1)
        for operation in ('allreduce','allgather'):
            samples=[]
            expected=bf16(136)*elements if operation=='allreduce' else b''.join(
                bf16(peer+1)*elements for peer in range(16))
            output=C.create_string_buffer(len(expected))
            for iteration in range(25):
                assert time.time()<intent['work_deadline']
                start=time.monotonic_ns()
                if operation=='allreduce':
                    reduce(src,dst,elements,9,0,comm,None)
                else:
                    gather(src,dst,elements,9,comm,None)
                sync()
                elapsed=time.monotonic_ns()-start
                copy(output,dst,len(expected),2)
                assert output.raw==expected,(operation,elements,iteration)
                if iteration==0 and elements==1024 and operation=='allreduce':
                    mark('first_collective.return')
                if iteration>=5:
                    samples.append(elapsed)
            checks.append(dict(operation=operation,elements=elements,checked=25,
                               output_SHA256=hashlib.sha256(expected).hexdigest(),latency_ns=samples))
            minimum=min(minimum,memory())
    providers=mapped_providers(Path('/proc/self/maps').read_text(),mode)
finally:
    if comm.value:destroy(comm)
    if dst.value:free(dst)
    if src.value:free(src)
    sync()
minimum=min(minimum,memory())
with (work/f'{rank:02d}.json').open('x') as out:
    json.dump(dict(passed=True,rank=rank,node=node,pci=pci,kfd_id=kfd['gpu_id'],step=step,
                   mode=mode,providers=providers,library_SHA256=binding['SHA256'],uid_SHA256=hashlib.sha256(raw).hexdigest(),
                   checks=checks,min_free_bytes=minimum,comm_destroyed=True),out,indent=2)
