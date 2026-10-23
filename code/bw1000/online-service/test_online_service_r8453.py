"""Synthetic CPU fixtures only; no GPU execution or throughput evidence."""
import json
from online_service_core_r8453 import Scheduler
from online_service_metrics_r8453 import evaluate


def reject(fn, error=AssertionError):
    try:
        fn()
    except error:
        return
    raise AssertionError('invalid operation accepted')


clock = 0
def now():
    global clock
    clock += .01
    return clock


# Thousands of epochs: retained control state stays bounded; persistent sink
# here stores counts only, never a hidden in-memory event history.
s=Scheduler(stages=2,micro=4,max_users=8,max_inflight=2,rows=8,max_queue=4,max_events=64,eos=[2])
persisted=0
last_event=-1
def sink(events):
    global persisted,last_event
    for e in events:
        assert e['event_id']==last_event+1
        last_event=e['event_id']
    persisted+=len(events)

for epoch in range(1000):
    for i in range(4):
        assert s.arrive(epoch*5+i,[11,12],1,now())
    assert not s.arrive(epoch*5+4,[11,12],1,now())
    assert len(s.requests)==4 and len(s.queue)==4
    assert s.events[-1]['status']=='rejected'
    j=s.dispatch(now())
    # Tail completion and logical retirement can be observed before upstream
    # host ack arrives. Keep both completed request/job payloads until that ack.
    assert s.take(0,now())==j
    assert s.take(1,now())==j
    s.finish(1,j.seq,now(),ids=[20]*4)
    assert len(s.jobs)==1 and len(s.requests)==4 and len(s.acks[j.seq])==1
    assert not s.can_reset_epoch()
    s.finish(0,j.seq,now())
    assert not s.jobs and not s.requests and not s.acks and not s.published
    assert s.can_reset_epoch()
    reject(lambda:s.reset_epoch_after_native_world_drain(now(),owners_reset_verified=False))
    s.reset_epoch_after_native_world_drain(now(),owners_reset_verified=True)
    s.drain_events(sink)
    assert not s.events
assert s.rejected==1000 and persisted>20000
reject(lambda:s.arrive(0,[11],1,now())) # replay after retirement still rejected.

# Packed prefills: dynamic M=16 with long prompts must split at8192, while
# remaining actual requests stay queued. No dummy/pad rows.
p=Scheduler(stages=2,micro=16,max_users=16,max_inflight=2,rows=16,max_queue=16)
for k in range(16):
    p.arrive(k,[11]*1024,1,now())
a=p.dispatch(now());b=p.dispatch(now())
assert len(a.keys)==len(b.keys)==8
assert sum(map(len,a.inputs))==8192 and sum(map(len,b.inputs))==8192
assert not set(a.rows)&set(b.rows)
assert p.dispatch(now()) is None
reject(lambda:p.arrive(16,[11]*1025,1,now()))

# Event overflow blocks atomically; no request/highwater/cursor mutation until
# caller persists events. Persistence errors retain events for idempotent retry.
e=Scheduler(stages=1,micro=1,max_users=1,max_inflight=1,rows=1,max_queue=1,max_events=5)
e.arrive(0,[11],1,now());job=e.dispatch(now());e.take(0,now())
before=(e.busy[:],e.cursor[:],len(e.events))
reject(lambda:e.finish(0,job.seq,now(),ids=[20]),BufferError)
assert (e.busy,e.cursor,len(e.events))==before
def bad_sink(events):
    raise OSError('synthetic journal unavailable')
reject(lambda:e.drain_events(bad_sink),OSError)
assert len(e.events)==4
e.drain_events(lambda batch:None)
e.finish(0,job.seq,now(),ids=[20])
assert not e.jobs and not e.requests
reject(lambda:e.finish(0,job.seq,now(),ids=[20]))

# Genuine changing cohorts and early EOS, not only one-token completion.
d=Scheduler(stages=2,micro=4,max_users=8,max_inflight=2,rows=8,max_queue=8,eos=[2])
for k in range(4):d.arrive(k,[11,12],3,now())
j0=d.dispatch(now())
for k in range(4,8):d.arrive(k,[13]*3,2,now())
j1=d.dispatch(now())
for j,ids in [(j0,[2,21,21,21]),(j1,[22]*4)]:
    for stage in range(2):
        assert d.take(stage,now())==j
        d.finish(stage,j.seq,now(),ids=ids if stage==1 else None)
j2=d.dispatch(now())
assert j2.rows==(2,3,4,5) and j2.pasts==(2,2,2,3)
assert len(j2.inputs)==4 and all(len(v)==1 for v in j2.inputs)

# Completed request metrics and censored-prefix diagnostics must be disjoint.
events=[]
def event(t,kind,key,**kw):events.append(dict(t=t,kind=kind,request=key,**kw))
for k in range(4):event(0,'arrival',k)
for k in range(3):event(.1,'admit',k)
event(.2,'terminal',3,status='rejected')
for k in range(3):event(1,'token',k,index=0,eos=False,boundary='emitter')
for k in range(3):event(2,'token',k,index=1,eos=False,boundary='emitter')
event(2,'terminal',1,status='cancelled')
event(3,'token',0,index=2,eos=True,boundary='emitter')
event(3,'terminal',0,status='completed')
r=evaluate(dict(events=events,window=[0,3],cards=16,boundary='emitter',target_tok_s=.4))
assert r['statuses']==dict(completed=1,unfinished=1,cancelled=1,rejected=1,failed=0)
assert r['completed_request_metrics']['user_decode_tok_s']['count']==1
assert r['completed_request_metrics']['user_decode_tok_s']['mean']==.5
assert r['censored_prefix_metrics']['observed_decode_tok_s']['count']==2
assert not r['censored_prefix_metrics']['SLO_qualification']
assert r['goodput_decode_tok_s_per_card']==1/3/16
assert r['aggregate_decode_tok_s']==1
assert r['requests'][1]['user_decode_tok_s'] is None
assert r['requests'][1]['observed_prefix_decode_tok_s']==1
assert r['requests'][2]['goodput_qualified'] is False
assert r['client_delivery']=='NOT_MEASURED'
print(json.dumps(dict(CPU_contract_passed=True,synthetic_only=True,
 GPU_and_Model='NOT_RUN',epochs=1000,checks=[
 'bounded queue and explicit rejection; constant-space replay watermark',
 'all-owner ack before job/request/ack publication retirement',
 '1000 epochs bounded retained state and explicit event sink',
 'event backpressure atomic; failed sink retains trace',
 'prefill dynamic M splits8192 bound; no padding',
 'changing cohort with early EOS preserves row/past',
 'complete distributions separated from unfinished/cancelled prefixes',
 'SLO goodput excludes censored prefixes; aggregate retains observed tokens'])))
