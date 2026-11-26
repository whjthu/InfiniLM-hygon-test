"""Backward-only diagnostic for the unchanged, fresh single-DCU T09 recipe."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat


STEPS=(6,7)
MAX_TRACE_BYTES=128<<20
ACTIVITIES={'kernel','gpu_memcpy','gpu_memset'}


def require(flag,code):
    if not flag:
        raise ValueError(code)


def stamp(info):
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def read(path,cap):
    require(path.resolve(strict=True)==path,'CANONICAL_BOUNDED_PROFILER_FILE')
    with path.open('rb') as stream:
        a=os.fstat(stream.fileno())
        require(stat.S_ISREG(a.st_mode) and 0<a.st_size<=cap,'BOUNDED_PROFILER_EXTENT')
        raw=stream.read(cap+1)
        require(len(raw)==a.st_size and stamp(a)==stamp(os.fstat(stream.fileno()))==stamp(path.lstat()),
                'STABLE_PROFILER_FILE')
    return raw


def save(path,obj):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,allow_nan=False,sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def marker(step):
    return f'hunyuanvideo/t09/backward/step{step}'


def interval(event):
    start,elapsed=event['ts'],event['dur']
    require(all(type(x) in (float,int) and math.isfinite(x) and x>=0 for x in (start,elapsed))
            and math.isfinite(start+elapsed),'FINITE_PROFILER_INTERVAL')
    return start,start+elapsed


def union(spans):
    finish=total=0
    for start,end in sorted(spans):
        total+=max(0,end-max(finish,start))
        finish=max(finish,end)
    return total


def external_id(e):
    key=e.get('args',{}).get('External id')
    return key if type(key) is int and key>0 else None


def reduce_trace(payload,step):
    events=payload['traceEvents']
    require(isinstance(events,list) and 0<len(events)<=500_000,'BOUNDED_KINETO_RECORD_COUNT')
    spans=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation'
           and str(e.get('name','')).startswith('hunyuanvideo/t09/backward/')]
    require(len(spans)==1 and spans[0]['name']==marker(step),'ONE_EXPECTED_STEP_BACKWARD_MARKER')
    a,b=interval(spans[0])
    require(b>a and type(spans[0].get('pid')) is int and spans[0]['pid']>0,
            'POSITIVE_BACKWARD_MARKER_INTERVAL_AND_PROCESS')
    cpu=defaultdict(list)
    for e in events:
        if e.get('ph')!='X' or e.get('cat')!='cpu_op' or external_id(e) is None:
            continue
        start,end=interval(e)
        inside=a<=start<=end<=b and e.get('pid')==spans[0].get('pid')
        cpu[external_id(e)].append((inside,e['name']))
    counts=Counter()
    times=defaultdict(list)
    ops=defaultdict(list)
    kernels=[]
    devices=set()
    streams=set()
    for e in events:
        cat=e.get('cat')
        if cat not in ACTIVITIES or e.get('ph')!='X':
            continue
        start,end=interval(e)
        args=e.get('args',{})
        times[cat].append((start,end))
        if cat!='kernel':
            continue
        device,stream=args.get('device'),args.get('stream')
        require(type(device) is int and type(stream) is int and min(device,stream)>=0,'EXACT_KINETO_KERNEL_DEVICE_STREAM')
        devices.add(device)
        streams.add(stream)
        linked=cpu.get(external_id(e),[])
        linked_inside=[name for in_scope,name in linked if in_scope]
        if len(linked)==len(linked_inside)==1:
            name,=linked_inside
            counts['unique_backward_CPU_external_id']+=1
            ops[name].append(end-start)
            kernels.append((start,end))
        else:
            counts['unlinked_or_ambiguous']+=1
    all_kernels=times['kernel']
    if len(devices)>1:
        raise ValueError('MULTIPLE_PHYSICAL_ACTIVITY_DEVICE_IDS')
    rank=[]
    for name,durations in ops.items():
        safe=name if re.fullmatch(r'(?:aten|autograd)::[A-Za-z0-9_:]{1,100}',name) else None
        rank.append(dict(known_op=safe,op_SHA256=sha256(name.encode()).hexdigest(),
                         calls=len(durations),summed_device_us=math.fsum(durations)))
    return dict(step=step,trace_events=len(events),cpu_backward_marker_us=b-a,
        observed_activity_counts={key:len(value) for key,value in sorted(times.items())},
        kernel_correlation_counts=dict(counts),profiler_device_ids=sorted(devices),stream_count=len(streams),
        kernel_sum_device_us=math.fsum(end-start for start,end in all_kernels),
        kernel_union_device_us=union(all_kernels),linked_kernel_union_device_us=union(kernels),
        top_linked_operators=sorted(rank,key=lambda r:(-r['summed_device_us'],r['op_SHA256']))[:20],
        hardware_attribution='OBSERVED_UNIQUE_BACKWARD_EXTERNAL_IDS' if union(kernels)>0 and len(devices)==1 else 'UNKNOWN',
        recomputation_kernel_placement='UNKNOWN_NO_DISTINCT_RECOMPUTATION_SCOPE',formal_performance_eligible=False)


def provider(torch):
    require(torch.__version__=='2.10.0' and torch.version.hip=='6.3.26113' and torch.cuda.device_count()==1,
            'ORIGINAL_3640_HIP_TORCH_SINGLE_DEVICE')
    p=torch.profiler
    ran=[p.ProfilerActivity.CPU,p.ProfilerActivity.CUDA]
    supported=p.supported_activities()
    require(set(ran)<=supported,'INSTALLED_KINETO_CPU_AND_DEVICE_ACTIVITY_REQUIRED')
    root=Path(torch.__file__).resolve(strict=True).parent
    sources={}
    for symbol in ('profile','record_function','supported_activities'):
        path=Path(inspect.getsourcefile(getattr(p,symbol))).resolve(strict=True)
        rel=path.relative_to(root)
        raw=read(path,1<<20)
        sources[symbol]=dict(relative_path=rel.as_posix(),bytes=len(raw),SHA256=sha256(raw).hexdigest())
    return p,ran,dict(torch='2.10.0',hip='6.3.26113',providers=sources,
                      supported_activities=sorted(a.name for a in supported))


class Backward67:
    def __init__(self,torch,output):
        self.torch=torch
        self.root=Path(output)/'t09-backward67'
        self.root.mkdir(mode=0o700)
        save(self.root/'intent.json',dict(planned_profiled_global_steps=list(STEPS),
             planned_warmup_updates=5,planned_total_updates=25,performance_eligible=False))
        self.seen=[]

    @contextmanager
    def at(self, step):
        if step not in STEPS:
            yield
            return
        require(len(self.seen)<len(STEPS) and STEPS[len(self.seen)]==step,'PROFILE_ONLY_EXACT_ORDERED_STEPS_6_7')
        self.seen.append(step)
        folder=self.root/f'step-{step}'
        folder.mkdir(mode=0o700)
        save(folder/'start.json',dict(step=step,marker=marker(step),before_gradient_clip=True))
        try:
            p,activities,binding=provider(self.torch)
            save(folder/'provider.json',binding)
            with p.profile(activities=activities,record_shapes=False,profile_memory=False,
                           with_stack=False,with_flops=False) as trace:
                with p.record_function(marker(step)):
                    yield
            raw=folder/'backward.trace.json'
            trace.export_chrome_trace(str(raw))
            data=read(raw,MAX_TRACE_BYTES)
            report=reduce_trace(json.loads(data),step)
            save(folder/'summary.json',dict(report,private_trace_bytes=len(data),private_trace_SHA256=sha256(data).hexdigest()))
        except Exception as exc:
            save(folder/'failure.json',dict(step=step,type=type(exc).__name__,
                 error_SHA256=sha256(str(exc).encode('utf8','replace')).hexdigest()))
            raise


def fresh(torch,output,*,warmup,measured,restored_step,recovery,gradient_checkpointing):
    require((warmup,measured,restored_step)==(5,20,0) and not recovery and gradient_checkpointing is True,
            'EXACT_FRESH_5_20_NONREENTRANT_RECIPE')
    return Backward67(torch,output)
