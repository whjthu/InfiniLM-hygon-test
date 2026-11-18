"""One CPU node: exact metadata stages, deliberately no gpu_busy_percent read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
import traceback

DRM=Path('/sys/class/drm')
KFD=Path('/sys/class/kfd/kfd/topology/nodes')
H=lambda b:hashlib.sha256(b).hexdigest()
ORIGINAL_REASON='56fefa267024b7c348a0d81fe807b6d156a27902b6812bb3fe4f77fb6c91961d'


def main():
    p=argparse.ArgumentParser()
    for name in ('request-id','output'):
        p.add_argument('--'+name,required=True)
    for name in ('epoch','step-origin'):
        p.add_argument('--'+name,type=float,required=True)
    a=p.parse_args();deadline=min(a.epoch+60,time.time()+20)
    assert 0<=time.time()-a.epoch<60 and time.time()<deadline
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    assert socket.gethostname().split('.')[0]=='a01r4n14' and os.environ['SLURM_JOB_ID']=='1087656'
    assert os.environ['SLURM_STEP_ID'].isdecimal()
    affinity=sorted(os.sched_getaffinity(0));assert affinity;os.sched_setaffinity(0,affinity[:1])
    out=Path(a.output);assert not out.exists()
    row=dict(request_id=a.request_id,epoch=a.epoch,node='a01r4n14',step='1087656.'+os.environ['SLURM_STEP_ID'],
             PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
             host=socket.gethostname(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             affinity=sorted(os.sched_getaffinity(0)),stages=[],error=None,metadata_inventory_passed=False,
             GPU_busy_reads=0,SDK_imports=False,original_sampler_passed=False)
    current=None
    def stage(name,path,operation):
        nonlocal current
        path=str(path)
        assert path==str(DRM) or path.startswith(str(DRM)+'/') or path==str(KFD) or path.startswith(str(KFD)+'/')
        assert 'gpu_busy_percent' not in path
        current=dict(stage=name,path=path)
        value=operation()
        row['stages'].append(dict(current,status='OK'))
        return value
    def text(path,prefix,limit):
        stream=stage(prefix+'.open',path,lambda:path.open('r',encoding='utf-8'))
        try:
            value=stage(prefix+'.read',path,lambda:stream.read(limit+1))
            assert len(value)<=limit
            return value
        finally:
            stream.close()
    try:
        entries=stage('DRM.iterdir',DRM,lambda:list(DRM.iterdir()))
        assert len(entries)<=256
        renders={}
        for entry in sorted(entries):
            if not re.fullmatch(r'renderD\d+',entry.name):
                continue
            device=entry/'device'
            stage('DRM.device.lstat',device,device.lstat)
            stage('DRM.device.readlink',device,lambda:os.readlink(device))
            target=stage('DRM.device.resolve',device,lambda:device.resolve(strict=True))
            pci=target.name.lower()
            assert re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci) and pci not in renders.values()
            renders[entry.name]=pci
        current=dict(stage='DRM.inventory.check',path=str(DRM))
        assert len(renders)==8
        nodes=stage('KFD.nodes.iterdir',KFD,lambda:list(KFD.iterdir()))
        assert len(nodes)<=32
        devices={}
        for node in sorted(nodes):
            assert node.name.isdecimal()
            gpu_id=int(text(node/'gpu_id','KFD.gpu_id',128))
            if gpu_id==0:
                continue
            properties=text(node/'properties','KFD.properties',32768)
            props=dict(line.split(maxsplit=1) for line in properties.splitlines())
            current=dict(stage='KFD.mapping.check',path=str(node/'properties'))
            render='renderD'+str(int(props['drm_render_minor']));pci=renders[render]
            domain,bus,slot=pci.split(':');device,function=slot.split('.')
            assert int(domain,16)==int(props['domain']) and int(props['location_id'])==(int(bus,16)<<8|int(device,16)<<3|int(function,16))
            assert pci not in devices and gpu_id>0 and gpu_id not in devices.values()
            devices[pci]=gpu_id
        current=dict(stage='KFD.complete.check',path=str(KFD))
        assert len(devices)==8 and set(devices)==set(renders.values())
        row['metadata_inventory_passed']=True
    except Exception as exc:
        # Only syscall/stage facts; no arbitrary repr/env/argv/source body.
        frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)
                for f in traceback.extract_tb(exc.__traceback__)[-4:]]
        row['error']=dict(current or {},type=type(exc).__name__,errno=getattr(exc,'errno',None),
                          filename=getattr(exc,'filename',None),strerror=getattr(exc,'strerror',None),
                          message_SHA256=H(str(exc).encode()),trace=frames,
                          same_original_error_string=H(str(exc).encode())==ORIGINAL_REASON)
    row.update(diagnostic_complete=True,original_failure_stage='NOT_RETROSPECTIVELY_PROVEN_BY_NEW_OBSERVATION')
    raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode();assert len(raw)<=32768
    with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
