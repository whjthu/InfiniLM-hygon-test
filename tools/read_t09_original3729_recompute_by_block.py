"""Read only the original .3729 trace and reduce proved block recomputation."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

REQUEST='16da9da61f4b40eab4f76be7c6ed044c'
SCOPE='fd68a293d7dd4b95bcd9b05be3f32c80'
STEP='1103606.3729'
LABEL='t09-step6-topology-fd68a293d7dd'
RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/LABEL
FOLDER=RUN/'outputs/hunyuanvideo-1.5/train'/LABEL/'t09-backward67/step-6'
TRACE=(105595897,'14840be27b10a9b1dc722f408a7be432f55c7f6ab0a13df76a8897d2e9e72dc6')
OBSERVER=(26379,'0043b4319d2a682e2d0a427e36c966b6ef5a7402419bab828000887d5c6c66f3')
SOURCE=(5486,'9704e167946aabf5bd4458c6bfef06cd79ddc1a8fc7fda0353d78133cd3f8351')
TERMINAL=(22431,'70799bd690a9404f3103f4faacd3f1a37e9b93ba670aa28324d5e19064d36cb3')
OUTCOME_SHA='303565344dc8d8c070b4010a680ac02834bc1fb811ffc7021090063f95584e2d'
BLOCK_TABLE='f260e3bbf8ad7923d31476fcd72ab1e9a22f8a4702f0980f0e6d386efef37c5f'
BMM=dict(forward=108,recompute=108,autograd_sequence=0,unknown=215)


def need(ok,code):
    if not ok:raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path,limit,pin=None):
    need(path.resolve(strict=True)==path,'EXACT_ORIGINAL_CANONICAL_FILE')
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(fd)
        def state(s):
            return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid
        need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()
             and 0<before.st_size<=limit and (not pin or before.st_size==pin[0]),
             'EXACT_ORIGINAL_OWNED_EXTENT')
        raw=stream.read(before.st_size+1)
        need(len(raw)==before.st_size and state(before)==state(os.fstat(fd))==state(path.lstat())
             and (not pin or digest(raw)==pin[1]),'EXACT_ORIGINAL_SAME_FD_CONTENT')
    return raw


def typed(event,category):
    return event.get('ph')=='X' and event.get('cat')==category


def identity(event):
    pid,tid=event.get('pid'),event.get('tid')
    return (pid,tid) if type(pid) is type(tid) is int and min(pid,tid)>0 else None


def measure(events,observer):
    i=observer['interval']
    spans=[i(e) for e in events]
    return dict(kernels=len(spans),sum_device_us=math.fsum(b-a for a,b in spans),
                union_device_us=observer['union'](spans))


def excess_union(all_events,retained,observer):
    """One-device footprint removed from the observed union if only proved events vanish."""
    interval=observer['interval']
    spans_all=[interval(e) for e in all_events]
    spans_other=[interval(e) for e in retained]
    value=observer['union'](spans_all)-observer['union'](spans_other)
    need(value>=-1e-6,'NONNEGATIVE_EXCLUSIVE_TIMELINE')
    return max(0.,value)


def block_exclusion(block_lists,retained,observer):
    cuts=defaultdict(list)
    sources=list(enumerate(block_lists))+[(-1,retained)]
    for owner,events in sources:
        for event in events:
            a,b=observer['interval'](event)
            if b>a:
                cuts[a].append((owner,1))
                cuts[b].append((owner,-1))
    live=Counter()
    unique=defaultdict(list)
    globally=[]
    previous=None
    for t,changes in sorted(cuts.items()):
        if previous is not None and live and -1 not in live:
            globally.append(t-previous)
            if len(live)==1:
                unique[next(iter(live))].append(t-previous)
        for owner,delta in changes:
            live[owner]+=delta
            if live[owner]==0:del live[owner]
        previous=t
    need(not live,'ALL_ORIGINAL_DEVICE_INTERVALS_CLOSED')
    return math.fsum(globally),{owner:math.fsum(spans) for owner,spans in unique.items()}


def safe_op(name):
    if re.fullmatch(r'aten::[A-Za-z0-9_:.]{1,76}',name):
        return name
    return 'sha256:'+digest(name.encode())


def reduce_trace(payload,observer,blocks,original=None):
    n=len(blocks)
    need(n>0 and [b['ordinal'] for b in blocks]==list(range(n)) and
         len({b['path'] for b in blocks})==n,'ORIGINAL_ORDERED_BLOCK_TABLE')
    old=observer['reduce_trace'](payload,6)
    details=observer['block_trace'](payload,6,blocks)
    events=payload['traceEvents']
    interval,contains,intersects,ext=(observer[x] for x in ('interval','contains','intersects','external_id'))
    marks={}
    for key,label in (('backward',observer['marker'](6)),('forward','hunyuanvideo/t09/step6-forward')):
        one,=(r for r in events if typed(r,'user_annotation') and r.get('name')==label)
        marks[key]=one
    main=interval(marks['backward'])
    before=interval(marks['forward'])
    process=marks['backward']['pid']
    need(process==marks['forward']['pid'] and before[1]<=main[0],'ORIGINAL_SEPARATE_FORWARD_BACKWARD_ROIS')
    by_lane=defaultdict(list)
    cpu_map=defaultdict(list)
    for row in events:
        if typed(row,'cpu_op'):
            if (key:=ext(row)) is not None:cpu_map[key].append(row)
        elif typed(row,'user_annotation') and row.get('name')==observer['recompute_marker'](6):
            lane=identity(row)
            if lane and lane[0]==process:
                by_lane[lane].append(interval(row))
    recompute={}
    for row in events:
        if not typed(row,'user_annotation') or not isinstance(row.get('name'),str):
            continue
        name=row['name']
        if not name.startswith(observer['BLOCK_PREFIX']+'recompute/'):
            continue
        ordinal=name.rsplit('/',1)[1]
        need(ordinal.isdecimal() and int(ordinal)<n and int(ordinal) not in recompute,'ONE_RECOMPUTE_MARKER_PER_EXACT_BLOCK')
        span=interval(row)
        lane=identity(row)
        require=[g for g in by_lane.get(lane,()) if contains(g,span) or g==span]
        need(lane is not None and lane[0]==process and contains(main,span) and len(require)==1 and span[1]>span[0],
             'SAME_LANE_ORIGINAL_GENERIC_AND_BLOCK_RECOMPUTE')
        recompute[int(ordinal)]=(lane,span)
    need(set(recompute)==set(range(n)),'ALL_ORIGINAL_REGISTERED_BLOCKS_MARKED')
    bins=defaultdict(list)
    for ordinal,(lane,span) in recompute.items():bins[lane].append((ordinal,span))
    all_k,proved_backward,proved_forward,backward_envelope=[],[],[],[]
    unknown,other_backward=[],[]
    groups=[defaultdict(list) for _ in blocks]
    block_kernel_lists=[[] for _ in blocks]
    known=[]
    bmm_count=0
    counters=Counter()
    for kernel in events:
        if not typed(kernel,'kernel'):continue
        all_k.append(kernel)
        linked=cpu_map.get(ext(kernel),[])
        op=linked[0] if len(linked)==1 else None
        lane=identity(op) if op else None
        loc=interval(op) if lane is not None else None
        own=lane is not None and lane[0]==process
        backward=own and main[0]<=loc[0]<=loc[1]<=main[1]
        forward=own and before[0]<=loc[0]<=loc[1]<=before[1]
        if forward:
            proved_forward.append(kernel)
            continue
        # Anything not proved forward stays in the envelope. Unknown work
        # cannot be removed and can cover candidate device time.
        backward_envelope.append(kernel)
        if backward:proved_backward.append(kernel)
        if not backward:
            counters['unbound_or_outside_phase']+=1
            unknown.append(kernel)
            continue
        possible=[(i,span) for i,span in bins.get(lane,()) if intersects(span,loc)]
        if len(possible)==1 and contains(possible[0][1],loc) and loc[1]>loc[0]:
            i=possible[0][0]
            known.append(kernel)
            block_kernel_lists[i].append(kernel)
            groups[i][safe_op(op['name'])].append(kernel)
            bmm_count+=int(op['name']=='aten::bmm')
        elif possible or lane not in bins:
            counters['crossing_or_unbound_recompute_lane']+=1
            unknown.append(kernel)
        else:
            other_backward.append(kernel)
    need(len(all_k)==len(proved_forward)+len(backward_envelope)
         and len(backward_envelope)==len(known)+len(unknown)+len(other_backward)
         and len(known)==sum(map(len,block_kernel_lists)),'KERNEL_DENOMINATORS_CONSERVED')
    if original is not None:
        need(len(blocks)==54 and details['classes']==BMM and details['actual_bmm_kernel_count']==431 and bmm_count==108 and
             old['kernel_union_device_us']==original['kernel_union_device_us'] and
             old['linked_kernel_union_device_us']==original['linked_kernel_union_device_us'] and
             details==original['step6_blocks'],'SAME_ORIGINAL_BMM_BOUNDARY_WITH_215_UNRESOLVED')
    retained=unknown+other_backward
    exclusive=excess_union(backward_envelope,retained,observer)
    checked,exclusive_blocks=block_exclusion(block_kernel_lists,retained,observer)
    need(math.isclose(checked,exclusive,rel_tol=1e-12,abs_tol=1e-6),'INDEPENDENT_DEVICE_UNION_SWEEP_EQUIVALENCE')
    by_block=[]
    for idx,block in enumerate(blocks):
        items=block_kernel_lists[idx]
        rank=[dict(operator=op,**measure(matches,observer)) for op,matches in groups[idx].items()]
        rank.sort(key=lambda x:(-x['union_device_us'],-x['sum_device_us'],x['operator']))
        by_block.append(dict(block=block['path'],ordinal=idx,**measure(items,observer),
            exclusive_removal_upper_bound_us=exclusive_blocks.get(idx,0.),top3=rank[:3],operator_count=len(rank)))
    return dict(passed=True,step=6,blocks=by_block,block_count=n,bmm_original_detail=details['classes'],
        known_recompute_BMM=bmm_count,unknown_categories=dict(counters),
        qualified_device_recompute=measure(known,observer),
        phase_linked_backward=measure(proved_backward,observer),phase_linked_forward=measure(proved_forward,observer),
        conservative_backward_envelope=measure(backward_envelope,observer),
        observed_capture_all_kernel=measure(all_k,observer),retained_proven_other_backward=measure(other_backward,observer),
        retained_unknown=measure(unknown,observer),
        non_kernel_device_activity={k:v for k,v in old['observed_activity_counts'].items() if k!='kernel'},
        exclusive_removal_upper_bound_us=exclusive,
        exclusive_all_capture_upper_bound_us=excess_union(all_k,proved_forward+retained,observer),
        removal_model='ALL_PROVED_RECOMPUTE_KERNELS_REMOVED_ON_UNCHANGED_OBSERVED_DEVICE_TIMELINE',
        host_time_or_wall_speedup_verified=False,formal_performance=False)


def main():
    need(os.getuid()==154459 and json.load(sys.stdin)=={'request':REQUEST,'scope':SCOPE,'original_step':STEP},
         'ONE_FIXED_READONLY_ORIGINAL_SCOPE')
    output=dict(request=REQUEST,scope=SCOPE,original_step=STEP,actual_GPU=0,actual_Slurm=0,read_only=True,
                planned_blocks=54,raw_trace_output=False,passed=False)
    try:
        mf=json.loads(read(RUN/'src/SOURCE_FILES.json',16384,SOURCE))
        need(mf['request']==SCOPE and mf['files']['backward_observer.py']==dict(bytes=OBSERVER[0],SHA256=OBSERVER[1]),
             'ORIGINAL_3729_SOURCE_OBSERVER')
        terminal=json.loads(read(RUN/'independent-terminal.json',32768,TERMINAL))
        need(terminal['passed'] is terminal['independent_release'] is True and terminal['request_id']==SCOPE and
             terminal['outcome']['SHA256']==OUTCOME_SHA and terminal['numbered_accounting']['gpu'][:4]==
             [STEP,LABEL+'-gpu','COMPLETED','0:0'] and (terminal['planned_updates'],terminal['completed_updates'])==(6,6),
             'EXACT_INDEPENDENT_3729_CPU_READBACK_AND_RELEASE')
        prior,=terminal['native']['profiled_steps']
        need((prior['raw_trace_bytes'],prior['raw_trace_SHA256'])==TRACE,'TRACE_TIED_TO_ORIGINAL_INDEPENDENT_READBACK')
        observers={'__name__':'frozen_3729_backward_observer'}
        exec(compile(read(RUN/'src/backward_observer.py',65536,OBSERVER),'exact_3729_observer','exec'),observers)
        blocks=json.loads(read(FOLDER/'blocks.json',65536))
        need(digest(json.dumps(blocks,sort_keys=True).encode())==BLOCK_TABLE and
             [r['path'] for r in blocks]==[f'double_blocks.{i}' for i in range(54)],'EXACT_3721_AND_3729_BLOCK_TABLE')
        projection=reduce_trace(json.loads(read(FOLDER/'backward.trace.json',128<<20,TRACE)),observers,blocks,prior)
        output.update(projection,trace_bytes=TRACE[0],trace_SHA256=TRACE[1],original_observer_SHA256=OBSERVER[1])
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        output.update(error_type=type(error).__name__,error_code=(str(error) if
           re.fullmatch(r'[A-Z0-9_]{1,96}',str(error)) else None),error_SHA256=digest(str(error).encode()))
    message=json.dumps(output,sort_keys=True,allow_nan=False,separators=(',',':'))
    need(len(message.encode())<=48128,'BOUNDED_54_BLOCK_TOP_THREE_RESULT')
    print(message,flush=True)


if __name__=='__main__':
    main()
