"""Explicit loaded-Trial caller; importing this file never loads Native or weights."""
import ctypes as C
import json
import math
import struct
import time
from causal import H,inputs,schedule

DTYPE_BYTES={0:1,3:8,4:1,9:2,10:4}

def copy(trial,name,row=0):
    m=trial.read('tensor',name.encode(),row)
    if not isinstance(m,dict) or m.get('contiguous') is not True:
        raise ValueError('missing actual contiguous tensor metadata')
    n=1
    for d in m['shape']:
        if type(d)!=int or d<0:raise ValueError('invalid tensor shape')
        n*=d
    if m['dtype'] not in DTYPE_BYTES or n*DTYPE_BYTES[m['dtype']]!=m['bytes'] or m['bytes']>16*1024**2:
        raise ValueError('exact tensor bytes/dtype/bounded copy required')
    b=C.create_string_buffer(m['bytes']);trial.call('copy',name.encode(),row,b,len(b))
    return m,b.raw

def snapshots(trial):
    result={}
    for name,shape,dtype in [('state',[8,512],9),('raw_cache',[128,8448],4)]:
        m,b=copy(trial,name)
        if m['shape']!=shape or m['dtype']!=dtype:raise ValueError('actual fullcapacity state/cache geometry differs')
        result[name]=dict(bytes=len(b),SHA256=H(b),all_zero=not any(b))
    return result

def observe(trial,past,pending):
    j=trial.read('observe')
    for k,v in dict(past=past,pending=pending,ready=True,local_heads=64,
                    compressed_capacity=8192,full_context=32768).items():
        if j.get(k)!=v:raise ValueError('actual owner state differs: '+k)
    if j['driver_free']<16*1024**3:raise ValueError('strict16GiB floor')
    return j

def selected_outputs(trial,width,sink,label):
    result={}
    for name in ('scores','indices'):
        m,b=copy(trial,name)
        if name=='scores':
            if m['shape']!=[1,width] or m['dtype']!=10 or m['bytes']!=4*width:raise ValueError('actual T1 score geometry')
            values=struct.unpack('<'+'f'*width,b)
            if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite T1 score')
            scores=values
        else:
            n=min(512,width)
            if m['shape']!=[1,n] or m['dtype']!=3 or m['bytes']!=8*n:raise ValueError('actual T1 TopK geometry')
            ids=struct.unpack('<'+'q'*n,b)
            if len(set(ids))!=n or any(x<0 or x>=width for x in ids):raise ValueError('invalid actual T1 indices')
            # Native requests descending scores. Equal scores can appear in either ID order.
            if [scores[x] for x in ids] != sorted(scores, reverse=True)[:n]:
                raise ValueError('TopK scores or descending order differ')
        sink(label,name,m,b) # Caller stores original bytes privately on R/L, never stdout/Git.
        result[name]=dict(metadata=m,SHA256=H(b))
    return result

def _run(trial,pinned,deadline,sink,progress,numeric_audit,cache_capture_factory,input_audit):
    """Requires already verified real checkpoint load. No create/load/close or job submission."""
    start=time.monotonic()
    plan=schedule()
    if [dict((k,v) for k,v in x.items() if k!='inputs') for x in pinned['steps']]!=plan:
        raise ValueError('frozen fullcausal schedule differs')
    if pinned['full_context']!=32768 or pinned['floor_bytes']!=16*1024**3 or pinned['compressed_capacity']!=8192 or pinned['topk']!=512:
        raise ValueError('context/floor changed')
    parameters=trial.read('parameters')
    if len(parameters)!=7 or any(x.get('loaded') is not True for x in parameters):
        raise ValueError('seven real checkpoint parameters must already be loaded')
    initial=observe(trial,0,False);records=[];numeric_records=[];cache_records=[];past=0;counts=dict(prepares=0,commits=0,aborts=0,resets=0)
    if any(initial.get(k)!=0 for k in counts):raise ValueError('fresh loaded component required')
    def check_audit(observed, *, prepared_payloads=None):
        if input_audit is not None:
            return input_audit.check(observed, prepared_payloads=prepared_payloads)
        return None
    initial_input_audit = check_audit(initial)
    if input_audit is not None:
        progress(dict(event='input_audit_postload', observed=initial_input_audit))
    def remember(record):
        progress(dict(event='step_complete',record=record))
        records.append(record)
    for i,(step,pin) in enumerate(zip(plan,pinned['steps'])):
        deadline('causal.step.'+str(i));begin=time.monotonic()
        if step['action']=='reset':
            before=snapshots(trial);trial.call('reset');after=snapshots(trial)
            if not after['state']['all_zero'] or before['raw_cache']['SHA256']!=after['raw_cache']['SHA256']:
                raise ValueError('reset state/cache contract')
            past=0;counts['resets']+=1;check_audit(observe(trial,past,False))
            remember(dict(step=i,label=step['label'],before=before,after=after,whole_wall_s=time.monotonic()-begin))
            continue
        if step['past']!=past:raise ValueError('noncausal plan')
        original=inputs(past,step['tokens'],step['variant'])
        for name,v in zip(('hidden_BF16','q_lora_BF16','positions_I64'),original):
            if dict(bytes=len(v),SHA256=H(v))!=pin['inputs'][name]:raise ValueError('input pin differs')
        before=snapshots(trial) if not step['commit'] else None
        cache_capture = (cache_capture_factory(trial, pin, copy, deadline)
                         if cache_capture_factory is not None else None)
        buffers=[C.create_string_buffer(v,len(v)) for v in original]
        pos=C.cast(buffers[2],C.POINTER(C.c_int64));seconds=C.c_double()
        call_start=time.monotonic()
        trial.call('prepare',past,step['tokens'],buffers[0],len(original[0]),buffers[1],len(original[1]),
                   pos,step['tokens'],C.byref(seconds))
        api_wall=time.monotonic()-call_start;counts['prepares']+=1
        deadline('causal.after_prepare.'+str(i))
        if seconds.value<0 or not math.isfinite(seconds.value):raise ValueError('invalid Native prepare clock')
        # Host payload unchanged check; actual GPU input redzones remain a separate unmet prerequisite.
        if any(b.raw!=v for b,v in zip(buffers,original)):raise ValueError('host input mutation')
        prepare_input_audit = check_audit(observe(trial,past,True), prepared_payloads=original)
        outputs={}
        if step['label'] in ('warm_abort','primary','full_boundary'):
            outputs=selected_outputs(trial,max(1,step['compressed_after']),sink,step['label'])
        if cache_capture is not None:
            cache_capture.pending()
        if numeric_audit is not None:
            report = numeric_audit(trial, pin, original[0], copy_tensor=copy, deadline=deadline)
            if report is not None:
                numeric_records.append(report)
                progress(dict(event='numeric_audit', step=i, report=report))
        native_prepare=seconds.value
        if step['commit']:
            trial.call('commit',C.byref(seconds));counts['commits']+=1;past+=step['tokens']
        else:
            trial.call('abort');counts['aborts']+=1
            after=snapshots(trial)
            if any(before[k]['SHA256']!=after[k]['SHA256'] for k in before):raise ValueError('abort mutated committed state/cache')
        check_audit(observe(trial,past,False))
        if cache_capture is not None:
            cache_report = cache_capture.finish()
            cache_records.append(dict(step=i, report=cache_report))
            progress(dict(event='cache_audit', step=i, report=cache_report))
        remember(dict(step=i,label=step['label'],past_before=step['past'],tokens=step['tokens'],
                            input_pins=pin['inputs'],outputs=outputs,Native_prepare_enclosure_s=native_prepare,
                            prepare_API_wall_s=api_wall,whole_step_wall_s=time.monotonic()-begin,
                            **({'prepare_input_audit': prepare_input_audit} if input_audit is not None else {})))
    if counts!=dict(prepares=69,commits=68,aborts=1,resets=2) or past!=512:raise ValueError('incomplete full schedule')
    final=observe(trial,past,False)
    if any(final.get(k)!=v for k,v in counts.items()):raise ValueError('actual Native counter mismatch')
    final_input_audit = check_audit(final)
    audit_report = {'input_audit': {'postload': initial_input_audit, 'final': final_input_audit}} if input_audit is not None else {}
    return dict(counts=counts,final_past=past,records=records,whole_schedule_wall_s=time.monotonic()-start,
                timing='Native prepare enclosure excludes uploads/output audit; API and wholewall retain other costs; none is exclusiveGPUbusy',
                numeric_audit=numeric_records, cache_audit=cache_records,
                performance_eligible=numeric_audit is None and cache_capture_factory is None,
                activation_origin=pinned['activation_origin'],GPU_input_redzones_qualified=False,
                independent_full_C4_numeric_reference=False,Model_GO=False,**audit_report)


def run(trial,pinned,deadline,sink,progress,*,numeric_audit=None,cache_capture_factory=None,input_audit=None):
    """Progress must durably save private records; exceptions never erase planned denominators."""
    progress(dict(event='planned',steps=schedule(),planned_prepares=69,planned_commits=68,
                  planned_aborts=1,planned_resets=2,complete=False))
    try:
        result=_run(trial,pinned,deadline,sink,progress,numeric_audit,cache_capture_factory,input_audit)
    except BaseException as e:
        progress(dict(event='failed',exception_type=type(e).__name__,message=str(e),complete=False))
        raise
    complete = dict(event='complete',counts=result['counts'],final_past=result['final_past'],complete=True)
    if input_audit is not None:
        complete['input_audit'] = result['input_audit']
    progress(complete)
    return result
