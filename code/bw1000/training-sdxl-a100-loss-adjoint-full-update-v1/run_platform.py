"""Execute the observed SDXL replay with the platform's original runtime."""
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import sys
import time



def main(platform, reference, policy, weights):
    root=Path(__file__).resolve().parent
    claim=json.loads((root/'CLAIM.json').read_bytes())
    epochs=json.loads((root/'intent.json').read_bytes())
    assert epochs['request']==claim['request'] and 0<=time.time()-epochs['epoch']<600
    dep=json.loads((root/(platform+'-deps.json')).read_bytes())
    assert {k:importlib.metadata.version(k) for k in dep['versions']}==dep['versions']
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    identity=dict(host=socket.gethostname().split('.')[0],pid=os.getpid(),
                  startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                  platform=platform)
    assert torch.cuda.device_count()==1 and bool(torch.version.hip)==(platform=='hip')
    assert platform=='hip'
    from hygon_sdxl_assigned_kfd import assigned_kfd
    from snapshot import drm_inventory
    gate=json.loads((root/'lease-before.json').read_bytes())
    step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    assert (gate['node']==identity['host']==claim['hygon']['node'] and gate['step']==step
            and gate['kfd_holder_count']==0 and gate['pci']==drm_inventory()
            and 0<=time.monotonic_ns()-gate['monotonic_ns']<45_000_000_000)
    hip=ctypes.CDLL('libamdhip64.so')
    buf=ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0
    pci=buf.value.decode().lower()
    if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
        pci='0000:'+pci
    assert pci in gate['pci']
    identity.update(pci=pci,KFD=assigned_kfd(pci),step=step)
    opts=dict(bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    assert opts==claim[platform+'_policy']
    pkg=root/'gencuda'
    pkg.mkdir(exist_ok=False)
    for existing, target in [('gencuda_init.py','__init__.py'),('gencuda_runtime.py','runtime.py')]:
        with (pkg/target).open('xb') as f:
            f.write((root/existing).read_bytes())
    import gencuda.runtime as runtime
    assert Path(runtime.__file__).resolve()==pkg/'runtime.py'
    import intervention_replay
    output=root/'candidate'
    sys.argv=[str(root/'intervention_replay.py'),'--reference',reference,'--policy',policy,'--weights',weights,
              '--output',str(output),'--accelerator',platform]
    intervention_replay.main()
    assert opts==dict(bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                       tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    evidence=json.loads((output/'intervention.json').read_bytes())
    result=dict(provenance=identity,policy=opts,intervention=evidence,
                native_step_summary=json.loads((output/'replay/summary.json').read_bytes()),raw_boundary_tensor_exports=0)
    with (root/'result.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')



if __name__=='__main__':
    main(*sys.argv[1:])
