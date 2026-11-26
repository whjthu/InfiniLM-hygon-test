"""Backward-only diagnostic for the unchanged, fresh single-DCU T09 recipe."""

from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
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


def reduce_original_trace(payload,step):
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



RECOMPUTE_CLASSES=('checkpoint_recompute','bmm_autograd_backward','unresolved')
RECOMPUTE_PREFIX='hunyuanvideo/t09/checkpoint-recompute/'


def recompute_marker(step):
    return f'{RECOMPUTE_PREFIX}step{step}'


class CheckpointRanges:
    """One owner, one forward graph, and one active diagnostic backward at a time."""
    def __init__(self):
        self.forward_step=None
        self.active=None

    @contextmanager
    def forward(self,step):
        require(self.forward_step is None and self.active is None,'ONE_ORIGINAL_FORWARD_AT_A_TIME')
        self.forward_step=step
        try:
            yield
        finally:
            self.forward_step=None

    def contexts(self):
        step=self.forward_step
        return nullcontext(),self.recomputation(step) if step in STEPS else nullcontext()

    @contextmanager
    def recomputation(self,step):
        active=self.active
        require(active is not None and active[0]==step,'CHECKPOINT_RECOMPUTE_MUST_USE_OWN_BACKWARD_WINDOW')
        with active[1](recompute_marker(step)):
            yield

    @contextmanager
    def backward(self,step,record_function):
        require(step in STEPS and self.forward_step is None and self.active is None,'ONE_EXACT_PROFILED_BACKWARD_CONTEXT')
        self.active=(step,record_function)
        try:
            yield
        finally:
            self.active=None


def complete(event,cat,name):
    return event.get('ph')=='X' and event.get('cat')==cat and event.get('name')==name


def contains(parent,child):
    return parent[0]<=child[0] and child[1]<=parent[1] and parent!=child


def intersects(a,b):
    return max(a[0],b[0])<min(a[1],b[1])


def spans_summary(spans):
    return dict(kernel_count=len(spans),sum_device_us=math.fsum(end-start for start,end in spans),
                union_device_us=union(spans))


def classify_bmm(op,checkpoint,engine,autograd):
    if not checkpoint:
        return 'unresolved'
    tid=op.get('tid')
    if type(tid) is not int or tid<=0:
        return 'unresolved'
    span=interval(op)
    if span[0]==span[1]:
        return 'unresolved'
    candidates=[r for r in checkpoint.get(tid,()) if intersects(r,span)]
    if candidates:
        if len(candidates)==1 and contains(candidates[0],span):
            return 'checkpoint_recompute'
        return 'unresolved'
    # A missing same-lane marker cannot prove non-recomputation. Markers on
    # other lanes do not establish a CPU ancestry or external-id relation.
    if tid not in checkpoint:
        return 'unresolved'
    nodes=[r for r in autograd.get(tid,()) if intersects(r,span)]
    parents=[r for r in engine.get(tid,()) if intersects(r,span)]
    if len(nodes)==len(parents)==1 and contains(parents[0],nodes[0]) and contains(nodes[0],span):
        return 'bmm_autograd_backward'
    return 'unresolved'


def classify_bmm_trace(payload,step,old):
    events=payload['traceEvents']
    roi,=(e for e in events if complete(e,'user_annotation',marker(step)))
    window=interval(roi)
    pid=roi['pid']
    ranges=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation'
            and isinstance(e.get('name'),str) and e['name'].startswith(RECOMPUTE_PREFIX)]
    checkpoint=defaultdict(list)
    invalid_ranges=0
    for event in ranges:
        span=interval(event)
        tid=event.get('tid')
        if event['name']!=recompute_marker(step) or event.get('pid')!=pid or type(tid) is not int or tid<=0 \
                or span[0]==span[1] or not contains(window,span):
            invalid_ranges+=1
        else:
            checkpoint[tid].append(span)
    engine,autograd,cpu=(defaultdict(list) for _ in range(3))
    for event in events:
        if event.get('ph')!='X' or event.get('cat')!='cpu_op':
            continue
        ext=external_id(event)
        if ext is not None:
            cpu[ext].append(event)
        field=event.get('tid')
        if event.get('pid')!=pid or type(field) is not int or not contains(window,interval(event)):
            continue
        if event.get('name')=='autograd::engine::evaluate_function: BmmBackward0':
            engine[field].append(interval(event))
        elif event.get('name')=='BmmBackward0':
            autograd[field].append(interval(event))
    classes={key:[] for key in RECOMPUTE_CLASSES}
    baseline_linked,bmm_candidates=[],[]
    other=0
    for event in events:
        if event.get('ph')!='X' or event.get('cat')!='kernel':
            continue
        linked=cpu.get(external_id(event),[])
        if not any(item.get('name')=='aten::bmm' for item in linked):
            other+=1
            continue
        span=interval(event)
        bmm_candidates.append(span)
        code='unresolved'
        if len(linked)==1:
            op=linked[0]
            cpu_start,cpu_end=interval(op)
            if op.get('pid')==pid and window[0]<=cpu_start<=cpu_end<=window[1]:
                baseline_linked.append(span)
                if not invalid_ranges:
                    code=classify_bmm(op,checkpoint,engine,autograd)
        classes[code].append(span)
    baseline=spans_summary(baseline_linked)
    total=spans_summary(bmm_candidates)
    projected={key:spans_summary(spans) for key,spans in classes.items()}
    count=sum(value['kernel_count'] for value in projected.values())
    require(count==total['kernel_count'] and count+other==old['observed_activity_counts'].get('kernel',0),
            'BMM_AND_ALL_KERNEL_COUNTS_CONSERVED')
    require(math.isclose(math.fsum(value['sum_device_us'] for value in projected.values()),total['sum_device_us'],
                         rel_tol=1e-12,abs_tol=1e-6),'BMM_CLASS_SUMS_CONSERVED')
    return dict(classes=projected,all_bmm_candidates=total,original_unique_backward_bmm=baseline,
                kernels_without_bmm_external_id_evidence=other,
                original_unlinked_bmm_candidates=count-baseline['kernel_count'],
                valid_recompute_ranges=sum(map(len,checkpoint.values())),invalid_recompute_ranges=invalid_ranges,
                attribution_complete=bool(checkpoint) and not invalid_ranges and count>0 and
                    projected['unresolved']['kernel_count']==0)


def reduce_trace(payload,step):
    report=reduce_original_trace(payload,step)
    report["bmm_recomputation"]=classify_bmm_trace(payload,step,report)
    return report


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
    def __init__(self,torch,output,checkpoint_ranges):
        self.torch=torch
        self.checkpoint_ranges=checkpoint_ranges
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
                with p.record_function(marker(step)), self.checkpoint_ranges.backward(step,p.record_function):
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


def fresh(torch,output,*,warmup,measured,restored_step,recovery,gradient_checkpointing,checkpoint_ranges):
    require((warmup,measured,restored_step)==(5,20,0) and not recovery and gradient_checkpointing is True,
            'EXACT_FRESH_5_20_NONREENTRANT_RECIPE')
    return Backward67(torch,output,checkpoint_ranges)
