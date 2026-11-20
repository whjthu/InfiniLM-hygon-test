"""Fresh PCI/KFD correspondence followed by the qualified three sysfs points."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
from qualified_sampler import collect, node_map, require, PCI

DRM = Path('/sys/class/drm')
KFD = Path('/sys/class/kfd/kfd')


def fresh_inventory():
    renders = {}
    for path in DRM.iterdir():
        if re.fullmatch(r'renderD\d+', path.name):
            pci = (path/'device').resolve(strict=True).name.lower()
            require(PCI.fullmatch(pci) and pci not in renders.values(), 'DRM_PCI_IDENTITY')
            renders[path.name] = pci
    require(len(renders) == 8, 'EIGHT_DRM_DEVICES')
    kfd = {}
    for node in (KFD/'topology/nodes').iterdir():
        require(node.name.isdecimal(), 'KFD_TOPOLOGY_NAME')
        gpu_id = int((node/'gpu_id').read_text())
        if gpu_id == 0:
            continue
        properties = (node/'properties').read_text()
        require(len(properties) <= 32768, 'KFD_METADATA_BOUND')
        props = dict(row.split(maxsplit=1) for row in properties.splitlines())
        render = 'renderD'+str(int(props['drm_render_minor']))
        require(render in renders, 'KFD_RENDER_MAPPING')
        pci = renders[render]
        domain, bus, slot = pci.split(':')
        device, function = slot.split('.')
        location = int(bus,16)<<8 | int(device,16)<<3 | int(function,16)
        require(int(domain,16) == int(props['domain']) and location == int(props['location_id']), 'KFD_PCI_DOMAIN_LOCATION')
        require(pci not in kfd and gpu_id > 0 and gpu_id not in [r['gpu_id'] for r in kfd.values()], 'KFD_UNIQUE')
        kfd[pci] = dict(gpu_id=gpu_id, render=render, topology_node=node.name)
    require(set(kfd) == set(renders.values()) and len(kfd) == 8, 'EIGHT_DRM_KFD_BIJECTION')
    return dict(sorted(kfd.items()))


def main():
    p = argparse.ArgumentParser()
    for key in ('ledger','output','request-id'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--scheduled',type=float,required=True)
    a=p.parse_args()
    deadline=a.epoch+180
    require(0<=time.time()-a.epoch<160 and a.scheduled+5<deadline-20, 'ORIGINAL_EPOCH')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    owners=node_map(json.loads(Path(a.ledger).read_bytes()))
    affinity=sorted(os.sched_getaffinity(0))
    require(bool(affinity),'STEP_CPU_AFFINITY')
    os.sched_setaffinity(0,affinity[:1])
    node=socket.gethostname().split('.')[0]
    parent=os.environ['SLURM_JOB_ID'];step=os.environ['SLURM_STEP_ID']
    require(node in owners and parent==owners[node][0] and step.isdecimal(), 'STEP_NODE_IDENTITY')
    out=Path(a.output)/(node+'.json')
    require(not out.exists(), 'EXCLUSIVE_NODE_RECEIPT')
    ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    row=dict(node=node,step=parent+'.'+step,request_id=a.request_id,epoch=a.epoch,
             scheduled_epoch=a.scheduled,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             PID=os.getpid(),startticks=ticks,uid=os.getuid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
             GPU_work=False,rows=[],error_type=None)
    try:
        pins=fresh_inventory()
        row['PCI_KFD']=pins
        row['rows']=collect(DRM,sorted(pins),a.scheduled)
        require(fresh_inventory()==pins,'POST_DRM_KFD_CHANGED')
    except Exception as exc:
        row['error_type']=type(exc).__name__
        row['error_reason_SHA256']=hashlib.sha256(str(exc).encode()).hexdigest()
    raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=32768,'NODE_RECEIPT_BOUND')
    with out.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return 0 if row['error_type'] is None else 2


if __name__=='__main__':
    raise SystemExit(main())
