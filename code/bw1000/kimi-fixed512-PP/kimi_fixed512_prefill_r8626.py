"""Bounded reuse of native PP defer for eight independent M512 prefills.

No threads, native changes or service transport are created here. Each existing
PP owner runs this function on its main thread. Decode stays outside, unchanged.
"""
import copy,time,threading
from kimi_matched_trace_r8467 import digest

def prefill_phase(adapter,jobs,*,stage,control,label,collect_each,timeout_s=300,
                  now_ns=time.monotonic_ns,on_actual=lambda record:None):
    assert threading.current_thread()is threading.main_thread()or getattr(adapter,'CPU_fixture',False)
    assert stage in range(4)and type(collect_each)is bool and 1<=timeout_s<=300
    frozen=copy.deepcopy(jobs);assert len(frozen)==8 and [j['seq']for j in frozen]==list(range(8))
    assert [j['rows']for j in frozen]==[[i+1]for i in range(8)]
    assert all(j['kind']=='prefill'and j['keys']==[i]and j['pasts']==[0]and len(j['inputs'])==1 and len(j['inputs'][0])==512 for i,j in enumerate(frozen))
    hashes=[digest(j)for j in frozen];plan_hash=digest(hashes);actual=[];feedback=[]
    # Existing socket timeout bounds control waits. A blocking native call remains
    # under the runner's original Slurm hard limit/independent supervisor; no
    # Python interruption/replay is safe after unknown native submission.
    control.set_timeout(timeout_s)
    packets=control.collect(label+':plan',dict(stage=stage,job_hashes=hashes,plan_hash=plan_hash))
    assert len(packets)==4 and [p['stage']for p in packets]==list(range(4))
    assert all(p['job_hashes']==hashes and p['plan_hash']==plan_hash for p in packets)
    phase_start=now_ns();deadline=phase_start+int(timeout_s*10**9)
    for job,expected_hash in zip(frozen,hashes):
        if now_ns()>=deadline:raise TimeoutError('prefill phase deadline before native; no replay')
        assert digest(job)==expected_hash
        begin=now_ns();ids=adapter.forward(job);ready=now_ns()
        assert digest(job)==expected_hash,'native adapter mutated immutable job'
        if stage==3:
            assert isinstance(ids,list)and len(ids)==1 and type(ids[0])is int and 0<=ids[0]<163840
            ids=list(ids) # Own token bytes; never retain borrowed native output.
        else:assert ids is None
        row=dict(seq=job['seq'],job_SHA256=expected_hash,local_begin_ns=begin,
                 local_ready_ns=ready,ids=ids)
        actual.append(row);on_actual(copy.deepcopy(row)) # Actuals before classification.
        if ready>=deadline:raise TimeoutError('prefill phase deadline after native; original handle only')
        if collect_each:
            packet=control.collect(label+':job'+str(job['seq']),dict(stage=stage,record=row))
            assert [p['stage']for p in packet]==list(range(4))
            assert all(p['record']['seq']==job['seq']and p['record']['job_SHA256']==expected_hash for p in packet)
            assert packet[3]['record']['ids']and len(packet[3]['record']['ids'])==1
            if stage==0:feedback.append(dict(seq=job['seq'],visible_ns=now_ns(),ids=list(packet[3]['record']['ids'])))
    packet=control.collect(label+':phaseend',dict(stage=stage,plan_hash=plan_hash,records=actual))
    visible=now_ns()
    assert [p['stage']for p in packet]==list(range(4))and all(p['plan_hash']==plan_hash for p in packet)
    assert all([v['job_SHA256']for v in p['records']]==hashes for p in packet)
    tail=packet[3]['records'];assert len(tail)==8 and all(v['ids']and len(v['ids'])==1 for v in tail)
    if stage==0 and not collect_each:feedback=[dict(seq=v['seq'],visible_ns=visible,ids=list(v['ids']))for v in tail]
    # No cross-host subtraction: only this owner's timestamps are reduced.
    own=dict(stage=stage,phase_start_ns=phase_start,phaseend_visible_ns=visible,
             local_phase_s=(visible-phase_start)*1e-9)
    if stage==3:
        own.update(tail_ready_s=[(v['local_ready_ns']-phase_start)*1e-9 for v in actual],
                   last_tail_ready_s=(actual[-1]['local_ready_ns']-phase_start)*1e-9)
    if stage==0:own['coordinator_visible_s']=[(v['visible_ns']-phase_start)*1e-9 for v in feedback]
    control.set_timeout(900)
    return dict(ids=[list(v['ids'])for v in tail],records=actual,clock=own,
                job_hashes=hashes,plan_hash=plan_hash,collect_each=collect_each,
                feedback=feedback,scope='New fixedM512 configuration qualification/harness batching of existing deferred PP, not faster than existing best service. Tailready != coordinator visible != client. No HTTP/client; puredecode M4 is not changed.')
