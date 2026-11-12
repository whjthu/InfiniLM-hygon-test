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
 from snapshot import drm_inventory
 intent=json.loads((WORK/'intent.json').read_bytes());epoch=intent['epoch'];remaining=epoch+660-time.time();require(0<remaining<=660,'native original epoch exhausted')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original SDXL native epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,remaining)
 before=json.loads((WORK/'before.json').read_bytes());gate=json.loads((WORK/'lease-before.json').read_bytes());require(gate['step']==os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'] and gate['node']==socket.gethostname().split('.')[0] and gate['kfd_holder_count']==0 and gate['boot_id']==before['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and gate['pci']==before['pci']==drm_inventory() and 0<=time.monotonic_ns()-gate['monotonic_ns']<=45_000_000_000 and gate['before_SHA256']==hashlib.sha256((WORK/'before.json').read_bytes()).hexdigest(),'stale or foreign original host lease receipt')

 import torch
 require(not torch.cuda.is_initialized(),'device context existed before original admission')
 require(torch.cuda.device_count()==1 and bool(torch.version.hip),'expected one visible HIP device')
 sys.path.insert(0,str(WORK/'source/tools'))
 from hygon_sdxl_assigned_kfd import assigned_kfd
 import importlib.metadata as metadata
 reuse=json.loads((WORK/'src/REUSED_CPU_POLICY.json').read_bytes());frozen=json.loads((Path(reuse['original_root'])/'runtime-preflight.json').read_bytes());require({k:metadata.version(k) for k in frozen['versions']}==frozen['versions'],'native runtime versions changed')
 for provider in frozen['providers'].values():
  p=Path(provider['path']);require(p.stat().st_size==provider['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==provider['SHA256'],'native provider bytes changed')
 reuse_cpu=json.loads((WORK/'src/REUSED_CPU174.json').read_bytes());cp=Path(reuse_cpu['root']);require(hashlib.sha256((cp/'outcome.json').read_bytes()).hexdigest()==reuse_cpu['outcome_SHA256'] and hashlib.sha256((cp/'evidence/CPU-result.json').read_bytes()).hexdigest()==reuse_cpu['result_SHA256'],'helper CPU original evidence changed');require(hashlib.sha256((WORK/'source/generation/src/gencuda/lora_backward_audit.py').read_bytes()).hexdigest()==reuse_cpu['helper_SHA256'] and hashlib.sha256((WORK/'source/tools/hygon_sdxl_backward_audit.py').read_bytes()).hexdigest()==reuse_cpu['wrapper_SHA256'],'helper source differs from CPU174')
 assigned=identity(torch);require(assigned['pci'] in before['pci'],'foreign actual assigned PCI');assigned['KFD']=assigned_kfd(assigned['pci']);require(not torch.cuda.is_initialized(),'PyTorch initialized before probe policy');assigned['assigned_PCI_pending']=False
 assigned.update(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),boot_id=gate['boot_id'],epoch=epoch,deadline=epoch+660)
 policies_before=dict(BF16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32);require(policies_before['BF16_reduction'] is True,'original native BF16 default policy changed');assigned['policies_before']=policies_before
 publish(WORK/'assigned.json',assigned)
 sys.path.insert(0,str(WORK/'source/tools'));source=str(WORK/'source/tools/hygon_sdxl_backward_audit.py');sys.argv=[source,'--reference','/reference/reference.pt','--policy','/reference/policy.json','--weights','/model','--output',str(WORK/'candidate')];runpy.run_path(source,run_name='__main__')
 policies_after=dict(BF16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32);require(policies_after==policies_before,'native policy changed during additive observation');assigned['policies_after']=policies_after
 actual=identity(torch);require(actual['pci'] in before['pci'],'foreign actual native PCI');require(actual['pci']==assigned['pci'],'assigned PCI changed during native update');assigned.update(actual,assigned_PCI_pending=False);publish(WORK/'assigned.json',assigned)
if __name__=='__main__':main()
