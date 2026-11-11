"""Measure the actual SDPA dispatch in two identical frozen SDXL replays."""

from collections import Counter
from contextlib import nullcontext
import ctypes as C
import gc
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys

WORK = Path("/t06-run")
MATH = "aten::_scaled_dot_product_attention_math"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def publish(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(torch):
    node = socket.gethostname().split(".")[0]
    require(node == "f11r4n11" and os.environ["SLURM_JOB_ID"] == "1087656", "foreign allocation")
    runtime = C.CDLL("libamdhip64.so")
    count = C.c_int()
    require(runtime.hipGetDeviceCount(C.byref(count)) == 0 and count.value == 1
            and torch.cuda.device_count() == 1 and bool(torch.version.hip),
            "SDXL comparison requires exactly one visible HIP device")
    pci = C.create_string_buffer(32)
    require(runtime.hipDeviceGetPCIBusId(pci, len(pci), 0) == 0, "missing HIP PCI identity")
    name = pci.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", name):
        name = "0000:" + name
    require(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", name),
            "invalid HIP PCI identity")
    return dict(step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                node=node, pci=name, hip=torch.version.hip, visible_cards=count.value)


def main():
 import hashlib,signal,time
 from snapshot import snapshot
 intent=json.loads((WORK/'intent.json').read_bytes());epoch=intent['epoch'];remaining=epoch+660-time.time();require(0<remaining<=660,'native original epoch exhausted')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original SDXL native epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,remaining)
 before=json.loads((WORK/'before.json').read_bytes());gate=snapshot(False);require(gate['kfd_holder_count']==0 and gate['boot_id']==before['boot_id'] and gate['pci']==before['pci'] and 0<=time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,'stale or occupied original physical node')
 import torch
 require(not torch.cuda.is_initialized(),'device context existed before original admission')
 require(torch.cuda.device_count()==1 and bool(torch.version.hip),'expected one visible HIP device')
 sys.path.insert(0,str(WORK/'source/tools'))
 from hygon_sdxl_assigned_kfd import assigned_kfd
 import importlib.metadata as metadata
 reuse=json.loads((WORK/'src/REUSED_CPU_POLICY.json').read_bytes());frozen=json.loads((Path(reuse['original_root'])/'runtime-preflight.json').read_bytes());require({k:metadata.version(k) for k in frozen['versions']}==frozen['versions'],'native runtime versions changed')
 for provider in frozen['providers'].values():
  p=Path(provider['path']);require(p.stat().st_size==provider['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==provider['SHA256'],'native provider bytes changed')
 assigned=identity(torch);require(assigned['pci'] in before['pci'],'foreign actual assigned PCI');assigned['KFD']=assigned_kfd(assigned['pci']);require(not torch.cuda.is_initialized(),'PyTorch initialized before probe policy');assigned['assigned_PCI_pending']=False
 assigned.update(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),boot_id=gate['boot_id'],epoch=epoch,deadline=epoch+660)
 publish(WORK/'assigned.json',assigned)
 sys.path.insert(0,str(WORK/'source/tools'));source=str(WORK/'source/tools/hygon_sdxl_precision_probe.py');sys.argv=[source,'--reference','/reference/reference.pt','--policy','/reference/policy.json','--weights','/model','--output',str(WORK/'candidate')];runpy.run_path(source,run_name='__main__')
 actual=identity(torch);require(actual['pci'] in before['pci'],'foreign actual native PCI');require(actual['pci']==assigned['pci'],'assigned PCI changed during native update');assigned.update(actual,assigned_PCI_pending=False);publish(WORK/'assigned.json',assigned)
if __name__=='__main__':main()
