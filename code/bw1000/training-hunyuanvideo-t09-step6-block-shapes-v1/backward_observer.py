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


STEPS=(6,)
MAX_TRACE_BYTES=256<<20
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


NATIVE_SOURCE_PINS=[('__init__.py', 1108, 'b0dbc51df5b633f1da1d459d44703b39841dc119f63a2eb937d61b4ed07706ad'), ('__init__.py', 1109, '334f3ae27552d1d6b7b68b648d9ae7deebed0ae966c585b33877c820ec481b6b'), ('__init__.py', 8290, '718fd95612d1e1b6ba390aa37d5e0a4ae7c36dd4c4c98d8079a9177e4bc6b916'), ('__init__.py', 8998, '6e6e1d7564d1ae167b0f272349671f7190ecc57f792abc0854e46f89964ac513'), ('__init__.py', 21369, '3f288a7a36e15a9d69cfd8626401f6bf0e2cca08ade49c4df290c98e81625846'), ('activation_layers.py', 1699, 'b794926c85569a40b30c1f96451c1e57d8f3b4f9726b0b956459e805b872a24d'), ('attention.py', 12909, '73db640a3170f0fba2c95def7ffa1252ddf31df6a32f65c52b8f88c8f55386c9'), ('embed_layers.py', 7227, '7e025d178c2e67d58cd3c58436ce795c35a5d2aed2a14fb8d7acac97c4d46fd9'), ('format_prompt.py', 3181, 'ab0b035abc39094040aaa43a246bd5ee43a927d5f5abaf2d434ad8c4977b9958'), ('hunyuanvideo_15_vae.py', 38331, '4078879df38616dd95b44f0f4660ffcfdf60c348fa0da33574a21d8eedad8a5b'), ('hunyuanvideo_1_5_transformer.py', 39361, 'd1a702891a92e04c80cdc072bc06cffe6e631fb962dae702d3c9244d347302d0'), ('mlp_layers.py', 5452, 'a796db1b7d89b5b4b837e938e52585b27c00e9fe9c38a9f0f440aaa884467b31'), ('modulate_layers.py', 3100, '1c5929c39db7093390a4761af56330585b4f90026f14a36f912a9e65fa6067b7'), ('norm_layers.py', 3227, '8e63dcbb54ff43d2eefd672c265590f5f90f6654098ee6774d2ea6e8ffee67c3'), ('posemb_layers.py', 14084, 'beb4e0f9f84ad1d60d06db0f982710de6ae07e3ee7013a5078e752f7ec25a8ec'), ('ssta_attention.py', 30949, '2d29dc7078de4b226ad909a44f6068ffefc6e002cc18865cef5dd756eeda3490'), ('token_refiner.py', 10490, '02675467e7505b13e8a2f085a9bf5460a64b43af8355cee86592458eb5effb9b'), ('upsample.py', 5993, 'b88908e9694bab4c1147b1091df0213780d964f5a41350bd0968db922fbe0735')]

BLOCK_PREFIX='hunyuanvideo/t09/block6/'
SHAPE_KEYS=('Input Dims','Input Strides','Input type')


def bind_blocks(transformer):
    entries={id(module):path for path,module in transformer.named_modules(remove_duplicate=False)
             if path and path.split('.')[0] in ('double_blocks','single_blocks') and path.count('.')==1}
    records={}
    for family in ('double_blocks','single_blocks'):
        blocks=getattr(transformer,family)
        require(len(blocks)>0,'EXACT_NONEMPTY_NATIVE_BLOCK_FAMILIES')
        for index,block in enumerate(blocks):
            path=f'{family}.{index}'
            require(entries.get(id(block))==path and id(block) not in records,
                    'UNIQUE_NATIVE_BLOCK_OBJECT_AND_NAMED_MODULE_PATH')
            target=inspect.unwrap(type(block).forward)
            src=Path(inspect.getsourcefile(target)).resolve(strict=True)
            lines,begin=inspect.getsourcelines(target)
            content=read(src,4<<20)
            require((src.name,len(content),sha256(content).hexdigest()) in NATIVE_SOURCE_PINS,
                    'BLOCK_FORWARD_FILE_MUST_BELONG_TO_FROZEN_NATIVE_SOURCE')
            number=len(records)
            records[id(block)]=dict(ordinal=number,path=path,index=index,family=family,
                cls=f'{type(block).__module__}.{type(block).__qualname__}',
                forward_line=begin,forward_SHA256=sha256(''.join(lines).encode()).hexdigest(),
                source_basename=src.name,source_bytes=len(content),source_SHA256=sha256(content).hexdigest())
    require(len(records)<=128,'BOUNDED_ACTUAL_BLOCK_INVENTORY')
    return records


def block_mark(phase,ordinal):
    return f'{BLOCK_PREFIX}{phase}/{ordinal}'


def capacity(torch):
    api=torch.cuda
    free,total=api.mem_get_info(0)
    props=api.get_device_properties(0)
    used,reserved=api.memory_allocated(0),api.memory_reserved(0)
    peak,peak_reserved=api.max_memory_allocated(0),api.max_memory_reserved(0)
    require(all(type(v) is int and v>=0 for v in (free,total,used,reserved,peak,peak_reserved))
            and 0<=free<=total and 0<total==props.total_memory and used<=reserved<=total
            and used<=peak and reserved<=peak_reserved,'ACTUAL_SINGLE_DEVICE_CAPACITY_AND_ALLOCATOR_SNAPSHOT')
    return dict(device=0,total_bytes=total,free_bytes=free,allocated_bytes=used,reserved_bytes=reserved,
                cumulative_allocated_peak_bytes=peak,cumulative_reserved_peak_bytes=peak_reserved)


def actual_operand_fields(args):
    result={}
    for key in SHAPE_KEYS:
        val=args.get(key)
        if key=='Input type':
            supported=isinstance(val,list) and 0<len(val)<=8 and all(isinstance(t,str) and len(t)<=48
                      and re.fullmatch(r'[A-Za-z0-9_ .<>:]+',t) for t in val)
        else:
            supported=isinstance(val,list) and 0<len(val)<=8 and all(isinstance(d,list) and len(d)<=16 and
                       all(type(n) is int and -(1<<40)<=n<(1<<40) for n in d) for d in val)
        result[key]=val if supported else 'UNKNOWN'
    return result


def block_trace(payload,step,blocks):
    """Separate direct marker attribution from sequence-bridged autograd evidence."""
    require(step==6 and 0<len(blocks)<=128,'EXACT_STEP6_BLOCK_TABLE')
    ordered=[b['ordinal'] for b in blocks]
    require(ordered==list(range(len(ordered))) and len({b['path'] for b in blocks})==len(blocks),
            'EXACT_UNIQUE_BLOCK_TABLE')
    events=payload['traceEvents']
    backward=[e for e in events if complete(e,'user_annotation',marker(step))]
    forward=[e for e in events if complete(e,'user_annotation','hunyuanvideo/t09/step6-forward')]
    require(len(backward)==len(forward)==1,'ONE_FORWARD_AND_BACKWARD_RANGE')
    rois={'forward':interval(forward[0]),'recompute':interval(backward[0])}
    pid=forward[0]['pid']
    require(pid==backward[0]['pid'] and rois['forward'][1]<=rois['recompute'][0],'FORWARD_PRECEDES_BACKWARD')
    markers,cpu,engines,original_by_sequence,kernels=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    for e in events:
        if e.get('ph')!='X':
            continue
        name=e.get('name','')
        if e.get('cat')=='user_annotation' and name.startswith(BLOCK_PREFIX):
            kind,index=name.removeprefix(BLOCK_PREFIX).split('/')
            require(kind in rois and index.isdecimal() and int(index) in ordered and e.get('pid')==pid
                    and type(e.get('tid')) is int and contains(rois[kind],interval(e)),
                    'VALID_REGISTERED_BLOCK_RANGE_AND_PHASE')
            markers[e['tid']].append((kind,int(index),interval(e)))
        elif e.get('cat')=='cpu_op':
            if (key:=external_id(e)) is not None:
                cpu[key].append(e)
            if e.get('pid')==pid and name in ('BmmBackward0','autograd::engine::evaluate_function: BmmBackward0'):
                engines[e.get('tid')].append(e)
        elif e.get('cat')=='kernel':
            kernels[external_id(e)].append(e)
    def direct(op):
        if op.get('pid')!=pid:
            return None
        span=interval(op)
        overlaps=[(kind,idx,r) for kind,idx,r in markers[op.get('tid')] if intersects(r,span)]
        if len(overlaps)==1 and contains(overlaps[0][2],span):
            return overlaps[0][:2]
        return None
    candidates=[op for ops in cpu.values() for op in ops if op.get('name')=='aten::bmm' and len(ops)==1]
    for op in candidates:
        found=direct(op)
        args=op.get('args',{})
        seq=args.get('Sequence number')
        logical_thread=args.get('Fwd thread id')
        if found and found[0]=='forward' and type(seq) is type(logical_thread) is int and seq>=0 and logical_thread>0:
            # Kineto tid and PyTorch forward thread IDs need not identify the
            # same namespace. Use the exact provider field in both records.
            original_by_sequence[(logical_thread,seq)].append(found[1])
    totals=Counter()
    groups=defaultdict(list)
    fields=Counter()
    for ext,device_events in kernels.items():
        linked=cpu.get(ext,[])
        if not any(op.get('name')=='aten::bmm' for op in linked):
            continue
        op=linked[0] if len(linked)==1 else None
        found=direct(op) if op else None
        sequence_status='UNKNOWN'
        if (not found and op and op.get('pid')==pid and contains(rois['recompute'],interval(op))
                and not any(intersects(r,interval(op)) for _,_,r in markers[op.get('tid')])):
            parent=[e for e in engines[op.get('tid')] if contains(interval(e),interval(op))]
            node=[e for e in parent if e['name']=='BmmBackward0']
            engine=[e for e in parent if e['name'].startswith('autograd::')]
            if len(node)==len(engine)==1 and contains(interval(engine[0]),interval(node[0])):
                args=node[0].get('args',{})
                seq,tid=args.get('Sequence number'),args.get('Fwd thread id')
                match=original_by_sequence.get((tid,seq),[]) if type(tid) is type(seq) is int else []
                if len(match)==1:
                    found=('autograd_sequence',match[0])
                    sequence_status='UNIQUE_THREAD_SEQUENCE_AND_NATIVE_PARENT'
        kind,index=found if found else ('unknown',None)
        operand=actual_operand_fields(op.get('args',{}) if op else {})
        metadata=json.dumps(operand,sort_keys=True)
        for key,val in operand.items():
            fields[key]+=len(device_events)*(val!='UNKNOWN')
        for e in device_events:
            device,stream=e.get('args',{}).get('device'),e.get('args',{}).get('stream')
            require((type(device),type(stream))==(int,int) and min(device,stream)>=0,'NATIVE_GPU_DEVICE_STREAM_REQUIRED')
            totals[kind]+=1
            groups[kind,index,sequence_status,metadata,device,stream].append(interval(e))
    require(sum(totals.values())==sum(len(kernels[k]) for k,ops in cpu.items()
        if any(op.get('name')=='aten::bmm' for op in ops)),'BMM_KERNEL_GROUP_COUNT_CONSERVED')
    rows=[dict(phase=phase,ordinal=idx,sequence_evidence=seq,operand_metadata=json.loads(meta),device=dev,stream=stream,
               **spans_summary(spans)) for (phase,idx,seq,meta,dev,stream),spans in groups.items()]
    rows.sort(key=lambda row:(-row['sum_device_us'],json.dumps(row,sort_keys=True)))
    return dict(step=6,actual_block_count=len(blocks),marker_count=sum(map(len,markers.values())),
        actual_bmm_kernel_count=sum(totals.values()),classes={k:totals[k] for k in ('forward','recompute','autograd_sequence','unknown')},
        operand_fields_observed={k:fields[k] for k in SHAPE_KEYS},groups=len(rows),
        groups_SHA256=sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        bounded_group_preview=rows[:12],shape_inferred=False)


class CheckpointRanges:
    """One owner, one forward graph, and one active diagnostic backward at a time."""
    def __init__(self):
        self.forward_step=None
        self.active=None
        self.blocks={}
        self.profiler=None

    @contextmanager
    def forward(self,step):
        require(self.forward_step is None and self.active is None,'ONE_ORIGINAL_FORWARD_AT_A_TIME')
        self.forward_step=step
        try:
            yield
        finally:
            self.forward_step=None

    def contexts(self,ordinal):
        step=self.forward_step
        if step not in STEPS:
            return nullcontext(),nullcontext()
        require(self.profiler is not None,"EXACT_STEP6_OBSERVER_ACTIVE_BEFORE_FORWARD")
        return self.profiler(block_mark("forward",ordinal)),self.recomputation(step,ordinal)

    @contextmanager
    def recomputation(self,step,ordinal):
        active=self.active
        require(active is not None and active[0]==step,'CHECKPOINT_RECOMPUTE_MUST_USE_OWN_BACKWARD_WINDOW')
        with active[1](recompute_marker(step)),active[1](block_mark("recompute",ordinal)):
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
    def capture(self, step):
        if step not in STEPS:
            yield
            return
        require(len(self.seen)<len(STEPS) and STEPS[len(self.seen)]==step,'PROFILE_ONLY_EXACT_STEP_6')
        self.seen.append(step)
        folder=self.root/f'step-{step}'
        folder.mkdir(mode=0o700)
        save(folder/'start.json',dict(step=step,marker=marker(step),before_gradient_clip=True))
        try:
            p,activities,binding=provider(self.torch)
            save(folder/'provider.json',binding)
            blocks=list(self.checkpoint_ranges.blocks.values())
            save(folder/"blocks.json",blocks)
            before=capacity(self.torch)
            with p.profile(activities=activities,record_shapes=True,profile_memory=False,
                           with_stack=False,with_flops=False) as trace:
                self.checkpoint_ranges.profiler=p.record_function
                try:
                    yield
                finally:
                    self.checkpoint_ranges.profiler=None
            save(folder/"capacity.json",dict(before=before,after=capacity(self.torch),peak_reset=False))
            raw=folder/'backward.trace.json'
            trace.export_chrome_trace(str(raw))
            data=read(raw,MAX_TRACE_BYTES)
            payload=json.loads(data)
            report=reduce_trace(payload,step)
            report["step6_blocks"]=block_trace(payload,step,blocks)
            save(folder/'summary.json',dict(report,private_trace_bytes=len(data),private_trace_SHA256=sha256(data).hexdigest()))
        except Exception as exc:
            save(folder/'failure.json',dict(step=step,type=type(exc).__name__,
                 error_SHA256=sha256(str(exc).encode('utf8','replace')).hexdigest()))
            raise


    def original_forward(self,step):
        if step not in STEPS:
            return nullcontext()
        return self.checkpoint_ranges.profiler('hunyuanvideo/t09/step6-forward')

    @contextmanager
    def at(self,step):
        if step not in STEPS:
            yield
            return
        marker_provider=self.checkpoint_ranges.profiler
        with marker_provider(marker(step)), self.checkpoint_ranges.backward(step,marker_provider):
            yield


def fresh(torch,output,*,warmup,measured,restored_step,recovery,gradient_checkpointing,checkpoint_ranges):
    require((warmup,measured,restored_step)==(5,20,0) and not recovery and gradient_checkpointing is True,
            'EXACT_FRESH_5_20_NONREENTRANT_RECIPE')
    return Backward67(torch,output,checkpoint_ranges)
