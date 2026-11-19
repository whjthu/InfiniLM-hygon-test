"""Measure AdamW on the frozen post-clip gradients and state of Qwen update 2."""
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import statistics
import sys
import time

import torch
from grouped_fused_adamw import GroupedFusedAdamW

ROOT=Path('/operator')
CAPTURE=Path('/capture')
RANKS=(0,6)
KINDS=('postclip','before-optimizer','after-optimizer')
MOMENTS=('exp_avg','exp_avg_sq')
OPTIONS={'lr','betas','eps','weight_decay','amsgrad','maximize','foreach','capturable',
         'differentiable','fused','initial_lr','decoupled_weight_decay'}


def exact(path,pin):
    start=path.lstat()
    assert stat.S_ISREG(start.st_mode) and (start.st_size==pin['bytes'] if 'bytes' in pin else 0<start.st_size<=65536)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1<<20), b''):
            digest.update(part)
    finish=path.lstat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    assert identity(start)==identity(finish) and digest.hexdigest()==pin['SHA256']


def load_rank(rank,pin):
    folder=CAPTURE/f'rank-{rank:02d}'
    receipt=folder/'receipt.json'
    exact(receipt,pin['receipt'])
    d=json.loads(receipt.read_bytes())
    assert d['rank']==rank and d['world_size']==8 and d['completed_global_step']==2
    assert d['optimizer_updates']==d['clip_calls']==2 and len(d['files'])==25
    lengths={x['name']:dict(bytes=x['bytes'],SHA256=x['SHA256']) for x in d['files']}
    records={}
    for kind in KINDS:
        name='update-02-'+kind+'.pt'
        assert lengths[name]==pin['archives'][name]
        source=folder/name
        exact(source,pin['archives'][name])
        records[kind]=torch.load(source,map_location='cpu',weights_only=True)
    return records


def device_identity(card):
    lease=json.loads((ROOT/'lease-before.json').read_bytes())
    assert socket.gethostname().split('.')[0]=='f13r1n10'
    assert lease['step']==os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    assert lease['kfd_holder_count']==0 and torch.version.hip and torch.cuda.device_count()==8
    hip=ctypes.CDLL('libamdhip64.so')
    hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    buf=ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(buf,len(buf),card)==0
    import importlib
    providers={'torch.optim.adam':'691246e104a54c3ff985922a4f551aea088c06c3358a18861e59c34a920d21e1',
               'torch.optim.adamw':'54299056b7745c162192132bb6028f3387c05ff4203518ff0240058584968312'}
    for module,pin in providers.items():
        path=Path(importlib.import_module(module).__file__)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
    pci=buf.value.decode().lower()
    if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
        pci='0000:'+pci
    assert pci in lease['pci']
    return dict(node=lease['node'],step=lease['step'],PCI=pci,card=card,torch=str(torch.__version__),
                hip=torch.version.hip,device=torch.cuda.get_device_name(card))


def pairs(rank,records,card):
    before=records['before-optimizer']
    after=records['after-optimizer']
    clipped=records['postclip']
    assert before['global_step']==after['global_step_before_increment']==clipped['global_step']==1
    groups=before['optimizer']['param_groups']
    assert groups==after['optimizer']['param_groups']
    p0,p1=before['parameters'],after['parameters']
    g=clipped['gradients']
    st0,st1=(d['optimizer']['state_by_name'] for d in (before,after))
    names=[n for group in groups for n in group['parameter_names']]
    assert len(names)==len(set(names))==504 and set(names)==set(p0)==set(p1)==set(g)==set(st0)==set(st1)
    for group in groups:
        flags={k:v for k,v in group.items() if k!='parameter_names'}
        assert set(flags)<=OPTIONS and flags.get('decoupled_weight_decay',True) is True
        assert not any(flags.get(k) for k in ('amsgrad','maximize','differentiable','capturable'))
        assert type(flags['lr']) in (float,int) and math.isfinite(flags['lr']) and flags['lr']>0
    active=[n for n in names if g[n] is not None]
    for name in names:
        grad=g[name]
        for value in (p0[name],p1[name],grad):
            if value is not None:
                assert value.device.type=='cpu' and value.dtype in (torch.float32,torch.bfloat16) and value.shape==p0[name].shape and bool(torch.isfinite(value).all())
        assert p0[name].dtype==p1[name].dtype and (grad is None or grad.dtype==p0[name].dtype)
        for ordinal,state in ((1,st0[name]),(2,st1[name])):
            if grad is None:
                assert state=={}
            else:
                assert set(state)=={'step',*MOMENTS} and state['step'].numel()==1 and state['step'].item()==ordinal
                for key in MOMENTS:
                    value=state[key]
                    assert value.dtype==p0[name].dtype and value.shape==p0[name].shape and bool(torch.isfinite(value).all())
                assert bool((state['exp_avg_sq']>=0).all())
    assert active
    device=torch.device('cuda',card)
    reference=dict(parameters={n:p0[n].to(device) for n in names},
                   grads={n:g[n].to(device) for n in active},
                   states={n:{k:st0[n][k].to(device) for k in MOMENTS} for n in active})

    def arm(mode):
        parameters={n:torch.nn.Parameter(reference['parameters'][n].clone()) for n in names}
        options=[]
        for raw in groups:
            opts={k:v for k,v in raw.items() if k not in ('parameter_names','decoupled_weight_decay')}
            opts.update(foreach=False,fused=True)
            options.append(dict(opts,params=[parameters[n] for n in raw['parameter_names']]))
        if mode=='fused64':
            optimizer=GroupedFusedAdamW(options,tensor_group_width=64)
        else:
            optimizer=torch.optim.AdamW(options)
        layout=[len(g['params']) for g in optimizer.state_dict()['param_groups']]
        assert layout==[len(g['parameter_names']) for g in groups]
        for n,p in parameters.items():
            if n not in reference['grads']:
                continue
            p.grad=reference['grads'][n].clone()
            optimizer.state[p]={k:v.clone() for k,v in reference['states'][n].items()}
            optimizer.state[p]['step']=st0[n]['step'].clone().to(device)
        return parameters,optimizer

    def restore(params,opt):
        with torch.no_grad():
            for name,p in params.items():
                p.copy_(reference['parameters'][name])
                if name in reference['grads']:
                    state=opt.state[p]
                    state['step'].fill_(1)
                    for key in MOMENTS:state[key].copy_(reference['states'][name][key])

    def compare(params,opt):
        totals={k:[0.0,0.0,0.0] for k in ('parameter_update',*MOMENTS)}
        changed=0
        for name,p in params.items():
            candidate=p.detach().cpu()
            changed+=not torch.equal(candidate,p0[name])
            items=[('parameter_update',candidate,p1[name],p0[name])]
            if name in reference['grads']:
                state=opt.state[p]
                assert state['step'].item()==2 and torch.equal(p.grad.detach().cpu(),g[name])
                items.extend((k,state[k].detach().cpu(),st1[name][k],None) for k in MOMENTS)
            else:
                assert p.grad is None and not opt.state.get(p)
            for kind,out,ref,base in items:
                x=out.double()
                y=ref.double()
                delta=x-y
                summ=totals[kind]
                summ[0]+=float(delta.square().sum())
                summ[1]+=float((y if base is None else y-base.double()).square().sum())
                summ[2]=max(summ[2],float(delta.abs().max()))
        assert changed>0 and all(math.isfinite(x) for s in totals.values() for x in s)
        return {k:dict(relative_L2=(math.sqrt(err/norm) if norm else None),absolute_max=mx,
                       error_L2=math.sqrt(err)) for k,(err,norm,mx) in totals.items()}|{'changed_tensors':changed}

    metrics={}
    torch.cuda.synchronize()
    for mode in ('fused','fused64'):
        params,optimizer=arm(mode)
        optimizer.step()
        torch.cuda.synchronize()
        metrics[mode]=compare(params,optimizer)
        del params,optimizer
    numeric_pass=all(metric['absolute_max']<=5e-5 and
                     (metric['relative_L2'] is not None and metric['relative_L2']<=5e-4
                      or metric['error_L2']==0)
                     for group in metrics.values() for key,metric in group.items() if key!='changed_tensors')
    blocks=[]
    if numeric_pass:
        for mode in ('fused','fused64','fused64','fused'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            previous=torch.cuda.memory_allocated()
            params,optimizer=arm(mode)
            events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            wall=[]
            event=[]
            for sample in range(25):
                restore(params,optimizer)
                torch.cuda.synchronize()
                started=time.perf_counter_ns()
                events[0].record()
                optimizer.step()
                events[1].record()
                events[1].synchronize()
                seconds=(time.perf_counter_ns()-started)/1e9
                if sample>=5:
                    wall.append(seconds)
                    event.append(events[0].elapsed_time(events[1]))
            blocks.append(dict(mode=mode,groups=len(optimizer.param_groups),warmup=5,timed=20,wall_seconds=wall,event_ms=event,
                 event_median_ms=statistics.median(event),wall_median_seconds=statistics.median(wall),
                 incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-previous))
            del params,optimizer
    return dict(rank=rank,tensors=len(names),active_gradients=len(active),
                elements=sum(p.numel() for p in p0.values()),dtype_counts=Counter(str(p.dtype) for p in p0.values()),
                recorded_optimizer_modes=[{k:g.get(k) for k in ('foreach','fused','lr','betas','eps','weight_decay')} for g in groups],
                numeric=metrics,numeric_pass=numeric_pass,blocks=blocks)


def write(name,value):
    with (ROOT/f'device-{sys.argv[1]}'/name).open('x',encoding='utf8') as dest:
        json.dump(value,dest,indent=2,allow_nan=False)


def main():
    card=int(sys.argv[1])
    if not 0<=card<8: raise ValueError('DCU_ORDINAL')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(card)
    (ROOT/f'device-{card}').mkdir(mode=0o700)
    source=json.loads((ROOT/'ARCHIVES.json').read_bytes())
    result=dict(rank_receipts=[],identity=None,passed=False,error=None)
    try:
        result['identity']=device_identity(card)
        write('device.json',result['identity'])
        for rank in (RANKS[card%len(RANKS)],):
            start=time.monotonic()
            records=load_rank(rank,source['ranks'][str(rank)])
            load_seconds=time.monotonic()-start
            detail=pairs(rank,records,card)|dict(archive_verify_and_load_seconds=load_seconds)
            write(f'rank-{rank:02d}-result.json',detail)
            result['rank_receipts'].append(dict(rank=rank,numeric_pass=detail['numeric_pass'],blocks=len(detail['blocks'])))
            del records,detail
        result['passed']=all(row['numeric_pass'] for row in result['rank_receipts'])
    except Exception as exc:
        import traceback
        frames=traceback.extract_tb(exc.__traceback__)
        result['error']={'type':type(exc).__name__,'function':frames[-1].name,'line':frames[-1].lineno}
        raise
    finally:
        write('result.json',result)
    return int(not result['passed'])


if __name__=='__main__':
    raise SystemExit(main())
