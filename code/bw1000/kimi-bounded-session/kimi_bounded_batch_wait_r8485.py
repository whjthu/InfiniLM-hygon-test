"""CPU-reviewed candidate: bounded prefill coalescing only, no native math change.
O,W,W,O execute on one already-loaded Engine via old outer runner/reset/finally.
No GPU authorization/launcher is supplied. CPU fixture results are not speedups.
"""
import dataclasses,json,os,threading,time,hashlib,io
from online_broker_r8455 import Client,Journal
from online_service_core_r8455 import Scheduler
from kimi_online_protocol_r8457 import export
from kimi_actual_arrival_r8472 import RecordingBroker,LiveBuffer
from kimi_matched_static_r8468 import BorrowedProvider
from kimi_matched_trace_r8467 import digest

class BoundedWaitScheduler(Scheduler):
    def __init__(self,*,wait_s,**kwargs):
        assert type(wait_s) in (int,float) and 0<=wait_s<=.01
        self.wait_s=float(wait_s)
        super().__init__(**kwargs)
    def dispatch(self,now):
        # O is exactly inherited policy. Wait only if inherited policy would
        # choose an underfull prefill, never alter native job shape or rows.
        if self.wait_s and len(self.jobs)<self.max_inflight and self.queue:
            live=sum(r.state in ('ready','running')for r in self.requests.values())
            capacity=min(self.micro,self.max_users-live,self.rows-self.next_row+1)
            n=min(capacity,len(self.queue))
            chooses_decode=bool(self.ready) and (n<=0 or self.decode_burst<self.max_decode_burst)
            if not chooses_decode and 0<n<capacity:
                oldest=self.requests[self.queue[0]].arrival
                if now<oldest+self.wait_s:return None
        return super().dispatch(now)

class TickRecordingBroker(RecordingBroker):
    def handle(self,req,arrival,conn):
        # Existing owner 2ms wait polling provides bounded wakeup after a partial
        # prefill deadline. Sole coordinator thread still owns every dispatch.
        if req['role']=='owner' and req['op']=='next':self.dispatch_ready()
        return super().handle(req,arrival,conn)


def emit_fixed_arrivals(c,inputs,arrival_plan,records,*,clock=time.perf_counter,sleep=time.sleep):
    """Actual producer consumer. CPU injectedclock validates policy only."""
    offsets=arrival_plan['offsets_s'];base=clock()
    assert len(offsets)==len(inputs)==8 and offsets[0]==0
    for i,prompt in enumerate(inputs):
        assert hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()==arrival_plan['prompt_SHA256'][i]
        target=base+offsets[i]
        while True:
            remain=target-clock()
            if remain<=0:break
            sleep(min(remain,.001))
        actual_send=clock()
        got=c.call(dict(op='arrive',prompt=prompt,limit=128))
        records.append(dict(request=i,target_t=target,send_t=actual_send,lateness_s=actual_send-target,actual_ingress_t=got['arrival']))
        assert got['accepted'] and got['request']==i

def bounded_epoch(g,inputs,eos,reset,*,full_reference,label,port_delta,wait_s,arrival_plan):
    lib,ctx,r,node,tag=g['lib'],g['ctx'],g['r'],g['node'],g['tag']
    report,control,save=g['report'],g['control'],g['save']
    reset(label);start=time.perf_counter();report['phase']=label;save()
    address=(g['head'],g['port']+port_delta);nonce=os.environ['KIMI_ENGINE_NONCE']
    def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
    a=BorrowedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full_reference)
    broker=journal=capture=producer=None;errors=[];producer_schedule=[]
    event_path=r/'results'/f'Kimi-bounded-arrival-r8485-{tag}-{label}-events.jsonl'
    capture_path=r/'results'/f'Kimi-bounded-arrival-r8485-{tag}-{label}-jobs.jsonl'
    if node==0:
        journal=Journal(event_path) if full_reference else LiveBuffer(event_path)
        capture=capture_path.open('x') if full_reference else LiveBuffer(capture_path)
        broker=TickRecordingBroker(capture=capture,full_reference=full_reference,nonce=nonce,journal=journal,
                               scheduler=BoundedWaitScheduler(stages=4,micro=4,max_users=8,max_inflight=4,rows=32,max_queue=8,eos=eos,wait_s=wait_s),
                               host=address[0],port=address[1])
    control.collect(label+':broker-ready',{})
    def produce():
        c=None
        try:
            c=Client(address,nonce,'ingress');deadline=time.monotonic()+120
            while len(broker.owner_connections)<4:
                if time.monotonic()>deadline:raise TimeoutError('owner attach')
                time.sleep(.01)
            emit_fixed_arrivals(c,inputs,arrival_plan,producer_schedule)
            c.call(dict(op='seal'))
        except BaseException as exc:errors.append(repr(exc));broker.poison('producer: '+repr(exc))
        finally:
            if c:c.close()
    if node==0:producer=threading.Thread(target=produce,daemon=True);producer.start()
    c=None
    try:
        c=Client(address,nonce,'owner',node)
        while True:
            value=c.call(dict(op='next'))
            if value.get('stop'):break
            if value['wait']:time.sleep(.002);continue
            job=value['job'];ids=a.forward(job)
            c.call(dict(op='finish',seq=job['seq'],ids=ids,
                        oracle=a.reference_observation if full_reference and node==3 else None))
        c.call(dict(op='bye'))
    finally:
        a.close()
        if c:c.close()
    control.collect(label+':all-owner-drained',{})
    shared=None
    if node==0:
        try:
            producer.join(5);assert not producer.is_alive() and not errors
            assert not broker.failure and broker.core.can_reset_epoch()
            broker.flush()
            captured=list(broker.records)
        finally:
            broker.close();capture.close();journal.close()
        events=[json.loads(v)for v in event_path.read_text().splitlines()]
        metric,merged=export(events,32)
        outputs=[[e['token']for e in events if e['kind']=='token' and e.get('request')==i]for i in range(8)]
        report.setdefault('actual_arrival_epochs',[]).append(dict(label=label,full_logit_reference=full_reference,
             actual_outputs=outputs,metric=metric,event_path=str(event_path),capture_path=str(capture_path),
             wait_s=wait_s,producer_schedule=producer_schedule,arrival_plan_SHA256=digest(arrival_plan),
             event_SHA256=hashlib.sha256(event_path.read_bytes()).hexdigest(),capture_SHA256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
             phase_s=time.perf_counter()-start,qualified=False,
             scope='Real finite coordinator-observed epoch; LIVE71 has boundedmemory journal persisted afterdrain, no per-token diskfsync; reference/replay excluded fromservice timing; notclientdelivery/no unpairedgain/no infinite service claim'))
        save() # complete IDs/EOS/metric paths and hashes BEFORE any numerical/status assertion.
        assert metric['statuses']['completed']==8 and not any(metric['statuses'][k] for k in ('unfinished','rejected','failed','cancelled'))
        shared=dict(captured=captured,outputs=outputs,event_path=str(event_path),capture_path=str(capture_path))
    # All nodes read same shared immutable file, avoiding multi-MiB controls.
    path=r/'results'/f'Kimi-bounded-arrival-r8485-{tag}-{label}-shared.json'
    if node==0:
        with path.open('x')as f:json.dump(shared,f)
    signature=hashlib.sha256(path.read_bytes()).hexdigest() if node==0 else None
    signatures=control.collect(label+':capture-file-ready',dict(SHA256=signature))
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==signatures[0]['SHA256']
    return json.loads(raw)



def consume_plan(path,inputs):
    import pathlib,math
    p=pathlib.Path(path);assert p.name=='Kimi-fixed-arrival-plan-r8485.json'
    raw=p.read_bytes();plan=json.loads(raw);assert plan['revision']==8485
    assert plan['order']==['O1','W1','W2','O2'] and plan['wait_s']==[0,.01,.01,0]
    assert (plan['TP'],plan['PP'],plan['cards'],plan['max_users'],plan['micro'],plan['output_limit'])==(8,4,32,8,4,128)
    offsets=plan['offsets_s'];assert len(offsets)==len(inputs)==8 and offsets[0]==0
    assert all(type(t)in(int,float) and math.isfinite(t) and 0<=t<=1 for t in offsets) and offsets==sorted(offsets)
    assert [hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).hexdigest()for p in inputs]==plan['prompt_SHA256']
    return plan,hashlib.sha256(raw).hexdigest()

def run_loaded_epochs(g,inputs,eos,reset,identity,arrival_plan):
    """Call only from reviewed existing loaded-Engine runner main thread.

    All4owners execute same loops; old outer finally owns normal native destroy.
    No Model load/retry/resume is done here. Unknown execution is failstop.
    Every actual live trace has independent full-logit CPUargmax qualification.
    """
    assert arrival_plan['order']==['O1','W1','W2','O2']
    assert arrival_plan['wait_s']==[0,.01,.01,0]
    assert (g['tp_nodes'],g['pp'])==(1,4)
    report,control,node=g['report'],g['control'],g['node']
    assert identity['DSO_SHA256']=='9001bf2bfa7e0821d52d9bec61ed5d9bf028c93380673fa880af3ed52e9c7388'
    assert os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=='1'
    report['bounded_wait_control_scope']='New O/W both use TickRecordingBroker; old588 is DIAGNOSTIC_ONLY, not paired control. Every actual send/ingress lateness recorded.'
    report['bounded_wait_native_identity']=identity
    report['bounded_wait_arrival_plan']=arrival_plan;g['save']()
    for index,(label,wait_s) in enumerate(zip(arrival_plan['order'],arrival_plan['wait_s'])):
        live=bounded_epoch(g,inputs,eos,reset,full_reference=False,label='bounded-'+label,
                           port_delta=12000+index*50,wait_s=wait_s,arrival_plan=arrival_plan)
        reset('bounded-'+label+'-independent-full-logit-oracle')
        def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
        a=BorrowedProvider(g['lib'],g['ctx'],stage=node,stages=4,rows=32,check=g['check'],
                           memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=True)
        checked=[];oracle_start=time.perf_counter()
        try:
            for v in live['captured']:
                job=v['job'];native=a.forward(job)
                parts=control.collect('bounded-'+label+'-oracle-'+str(job['seq']),dict(ids=native,oracle=a.reference_observation if node==3 else None,job_SHA256=digest(job)))
                assert all(x['job_SHA256']==v['job_SHA256']for x in parts)
                oracle=parts[3]['oracle'];expected=oracle['cpu_argmax_ids']
                row=dict(job=job,job_SHA256=v['job_SHA256'],live_actual_ids=v['actual_ids'],reference_oracle=oracle,
                         first_difference_row_index=next((i for i,(x,y)in enumerate(zip(v['actual_ids'],expected or []))if x!=y),None))
                checked.append(row);report.setdefault('bounded_wait_oracle_checks',{})[label]=checked;g['save']()
                assert oracle['finite'] and oracle['native_equals_CPU'] and v['actual_ids']==expected,'bounded epoch firstdiff: diagnostic only'
        finally:a.close()
        report.setdefault('bounded_wait_qualification',{})[label]=dict(qualified=True,oracle_phase_s=time.perf_counter()-oracle_start,scope='MATCHED_SHAPE_JOBTRACE_ONLY',oracle_jobs=len(checked),service_boundary='coordinator ingress to emitter feedback; not client')
        g['save']()
    report['bounded_wait_four_epochs_qualified']=True;g['save']()
