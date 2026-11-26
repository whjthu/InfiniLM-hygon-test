"""Read only the original .3701 private traces and print a bounded attribution projection."""
import json
import math
import os
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import re
import stat
import sys

REQUEST='a1db362638904b67897d16121d252f17'
RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/t09-bwd67-recompute-29086988dd03')
TRACE=RUN/'outputs/hunyuanvideo-1.5/train/t09-bwd67-recompute-29086988dd03/t09-backward67'
OBSERVER=(14414,'aac6a8c1fee329e3280d2ae2cb62f3c10f22a37264261471b0c7a7c472907e4a')
READBACK=(10258,'da76ced3872c769e8af52f3d05a6fb0ed45c5244a2cad942212b8f6521992d17')
PINS={6:(53825878,'abc5264f9e480b4600c95d11575ace2818019dd2cb260692fde70114a9f0ee7c'),
      7:(53992468,'11f44363cd3fbcf0e0cb28ffbae0ab639f9d534eb1939dd9846962229f104321')}
CLASSES=('checkpoint_recompute','bmm_autograd_backward','unresolved')
FIELDS=('Input Dims','Input Strides','Input type','Concrete Inputs','Call stack','Fwd thread id','Sequence number')
LAUNCH_FIELDS=('device','stream','grid','block','registers per thread','shared memory','queued')


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def h(raw):
    return sha256(raw).hexdigest()


def stable_read(path,pin):
    size,digest=pin
    require(path.resolve(strict=True)==path,'CANONICAL_ORIGINAL_ARTIFACT')
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        def stamp(s):
            return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid
        st=os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size==size,
                'ORIGINAL_OWNER_REGULAR_AND_EXACT_EXTENT')
        data=stream.read(size+1)
        require(stamp(st)==stamp(os.fstat(fd))==stamp(path.lstat()) and len(data)==size and h(data)==digest,
                'ORIGINAL_ARTIFACT_CHANGED_OR_DIGEST_DIFFERS')
    return data


def metric(spans,observer):
    return dict(count=len(spans),sum_device_us=math.fsum(b-a for a,b in spans),union_device_us=observer['union'](spans))


def lane(event):
    pid,tid=event.get('pid'),event.get('tid')
    return (pid,tid) if type(pid) is type(tid) is int and min(pid,tid)>0 else None


def eligible_complete(event,category):
    return event.get('ph')=='X' and event.get('cat')==category


def ancestors(nodes,span,observer):
    contains,intersects=observer['contains'],observer['intersects']
    possibles=[(other,name,ext) for other,name,ext in nodes if intersects(other,span)]
    # Ordinary descendants inside BMM are irrelevant. Crossing and equal
    # ranges cannot establish an unambiguous parent.
    if any(not contains(other,span) and not contains(span,other) for other,_,_ in possibles):
        return None
    enclosing=sorted((entry for entry in possibles if contains(entry[0],span)),
                     key=lambda row:row[0][1]-row[0][0])
    if any(not contains(outer[0],inner[0]) for inner,outer in zip(enclosing,enclosing[1:])):
        return None
    return enclosing


def summarize(payload,step,observer,expected):
    old=observer['reduce_trace'](payload,step)
    bmm=old['bmm_recomputation']
    if expected is not None:
        require(all(old[k]==expected[k] for k in (
            'bmm_recomputation','kernel_correlation_counts','kernel_union_device_us','observed_activity_counts')),
            'ORIGINAL_INDEPENDENT_REDUCTION_DIFFERS')
        require((bmm['valid_recompute_ranges'],bmm['invalid_recompute_ranges'],
                 *(bmm['classes'][key]['kernel_count'] for key in CLASSES))==(54,0,108,215,0),
                'EXACT_54_RANGES_108_RECOMPUTE_215_AUTOGRAD_REQUIRED')
    events=payload['traceEvents']
    interval=observer['interval']
    extid=observer['external_id']
    contains,intersects=observer['contains'],observer['intersects']
    roi,=(e for e in events if eligible_complete(e,'user_annotation') and e.get('name')==observer['marker'](step))
    window=interval(roi)
    source_pid=roi['pid']
    ranges,by_ext,nodes,engine,autograd=(defaultdict(list) for _ in range(5))
    for e in events:
        on=lane(e)
        if eligible_complete(e,'user_annotation') and e.get('name')==observer['recompute_marker'](step):
            if on is not None and on[0]==source_pid and contains(window,interval(e)):
                ranges[on].append(interval(e))
        elif eligible_complete(e,'cpu_op'):
            if (ext:=extid(e)) is not None:
                by_ext[ext].append(e)
            if on is None or on[0]!=source_pid or not contains(window,interval(e)):
                continue
            name=e['name']
            span=interval(e)
            if name=='autograd::engine::evaluate_function: BmmBackward0':
                engine[on[1]].append(span)
            elif name=='BmmBackward0':
                autograd[on[1]].append(span)
            if name!='aten::bmm':
                nodes[on].append((span,name,ext))
    coverage={name:[] for name in ('attributed_recompute','outside_proven_cpu_ranges','unknown')}
    groups=defaultdict(list)
    grouped_bmm=Counter()
    parent_status=Counter()
    observed_fields=Counter()
    device_name={}
    for props in payload.get('deviceProperties',[]):
        if isinstance(props,dict) and type(props.get('id')) is int and isinstance(props.get('name'),str):
            name=props['name']
            require(len(name)<=256,'BOUNDED_DEVICE_NAME')
            if props['id'] in device_name:
                require(device_name[props['id']]==name,'ONE_DEVICE_NAME_PER_TRACE_DEVICE')
            device_name[props['id']]=name
    for e in events:
        if not eligible_complete(e,'kernel'):
            continue
        gpu=interval(e)
        linked=by_ext.get(extid(e),[])
        op=linked[0] if len(linked)==1 else None
        on=lane(op) if op else None
        span=interval(op) if op else None
        possible=[r for r in ranges.get(on,()) if intersects(r,span)] if on else []
        inside=on in ranges and len(possible)==1 and contains(possible[0],span)
        key='attributed_recompute' if inside else (
            'outside_proven_cpu_ranges' if on in ranges and contains(window,span) and not possible else 'unknown')
        coverage[key].append(gpu)
        if not any(item.get('name')=='aten::bmm' for item in linked):
            continue
        kind=observer['classify_bmm'](op,{key[1]:value for key,value in ranges.items()},engine,autograd) \
            if on and on[0]==source_pid and contains(window,span) and not bmm['invalid_recompute_ranges'] else 'unresolved'
        grouped_bmm[kind]+=1
        chain=ancestors(nodes.get(on,()),span,observer) if on else None
        status='strict_nested' if chain else 'no_proven_parent'
        parent_status[status]+=1
        parent=tuple(h(name.encode()) for _,name,_ in chain[:4]) if chain else ()
        op_fields=tuple(key for key in FIELDS if op and key in op.get('args',{}))
        k_fields=tuple(key for key in LAUNCH_FIELDS if key in e.get('args',{}))
        for key in op_fields:
            observed_fields[key]+=1
        args=e.get('args',{})
        name=e.get('name')
        require(isinstance(name,str) and 0<len(name)<=8192,'BOUNDED_ACTUAL_KERNEL_NAME')
        groups[kind,status,parent,args['device'],h(name.encode()),op_fields,k_fields].append(gpu)
    require(all(grouped_bmm[key]==bmm['classes'][key]['kernel_count'] for key in CLASSES),
            'THREE_ORIGINAL_BMM_COUNTS_CONSERVED')
    require(sum(map(len,coverage.values()))==old['observed_activity_counts'].get('kernel',0),
            'ALL_KERNELS_ACCOUNTED')
    by_class={key:dict(bmm['classes'][key]) for key in CLASSES}
    rows=[dict(classification=kind,parent_chain_status=status,parent_name_SHA256=parents,device=device,
               kernel_name_SHA256=name,observed_operator_fields=op_fields,observed_kernel_fields=k_fields,
               **metric(spans,observer))
          for (kind,status,parents,device,name,op_fields,k_fields),spans in groups.items()]
    for key in CLASSES:
        require(math.isclose(math.fsum(r['sum_device_us'] for r in rows if r['classification']==key),
            by_class[key]['sum_device_us'],rel_tol=1e-12,abs_tol=1e-6),'BMM_GROUP_SUMS_CONSERVED')
    ranks=sorted(rows,key=lambda r:(-r['sum_device_us'],r['kernel_name_SHA256'],r['classification'],
                                   r['parent_name_SHA256'],r['observed_operator_fields'],r['observed_kernel_fields']))
    return dict(step=step,valid_recompute_ranges=bmm['valid_recompute_ranges'],
                all_kernel_count=old['observed_activity_counts'].get('kernel',0),
                non_kernel_device_activity={k:v for k,v in old['observed_activity_counts'].items() if k!='kernel'},
                all_operator_kernel_partition={k:metric(v,observer) for k,v in coverage.items()},
                total_recomputation_device_time='UNKNOWN_UNATTRIBUTED_KERNELS_AND_HOST_OR_OTHER_ACTIVITY',
                bmm_classes=by_class,bmm_total=bmm['all_bmm_candidates'],bmm_parent_evidence=dict(parent_status),
                bmm_operator_field_presence={key:observed_fields[key] for key in FIELDS},
                bmm_group_count=len(rows),bmm_group_inventory_SHA256=h(json.dumps(ranks,sort_keys=True).encode()),
                bmm_groups=ranks[:24],bmm_other_groups=max(0,len(rows)-24),
                device_labels={str(i):(device_name[i] if re.fullmatch(r'[A-Za-z0-9 ._()+/-]{1,96}',device_name[i])
                    else h(device_name[i].encode())) for i in old['profiler_device_ids'] if i in device_name},
                record_shapes_enabled=False,with_stack_enabled=False,shape_inferred=False)


def main():
    require(json.load(sys.stdin)=={'request':REQUEST,'scope':'29086988dd03477fbd24c337e7712905','original_step':'1103606.3701'},
            'ONE_EXACT_ORIGINAL_SCOPE')
    ns={'__name__':'sealed_observer_3701'}
    exec(compile(stable_read(RUN/'src/backward_observer.py',OBSERVER),'sealed_observer_3701','exec'),ns)
    prior=json.loads(stable_read(RUN/'independent-terminal.json',READBACK))
    require(prior['independent_release'] is True and prior['passed'] is True and prior['numbered_accounting']['gpu'][0]=='1103606.3701',
            'EXACT_3701_TERMINAL_AND_RELEASE')
    expected={e['step']:e for e in prior['native']['profiled_steps']}
    out=dict(request=REQUEST,original_step='1103606.3701',profiles=[],planned_profiled_steps=[6,7],
             passed=False,actual_new_GPU=0,actual_new_Slurm=0,formal_performance=False,raw_trace_transferred=False)
    for step,pin in PINS.items():
        try:
            spec=expected[step]
            require(pin==(spec['raw_trace_bytes'],spec['raw_trace_SHA256']),'ORIGINAL_TRACE_PIN_DISAGREES')
            raw=stable_read(TRACE/f'step-{step}'/'backward.trace.json',pin)
            out['profiles'].append(dict(summarize(json.loads(raw),step,ns,spec),trace_bytes=pin[0],trace_SHA256=pin[1]))
        except (OSError,ValueError,KeyError,TypeError) as error:
            out['profiles'].append(dict(step=step,error_type=type(error).__name__,error_SHA256=h(str(error).encode())))
    out['passed']=all('trace_SHA256' in row for row in out['profiles'])
    line=json.dumps(out,sort_keys=True,allow_nan=False)
    require(len(line.encode())<=49152,'BOUNDED_SAFE_PROJECTION')
    print(line,flush=True)


if __name__=='__main__':
    main()
