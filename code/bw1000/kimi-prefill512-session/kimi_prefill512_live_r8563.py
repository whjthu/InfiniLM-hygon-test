"""CPU-reviewed candidate: bounded prefill coalescing only, no native math change.
O,W,W,O execute on one already-loaded Engine via old outer runner/reset/finally.
No GPU authorization/launcher is supplied. CPU fixture results are not speedups.
"""
import dataclasses,json,os,threading,time,hashlib,io
from online_broker_r8455 import Client,Journal
from online_service_core_r8455 import Scheduler
from kimi_online_protocol_r8457 import export
from kimi_actual_arrival_r8472 import RecordingBroker,LiveBuffer
from kimi_prefill512_scope_r8563 import ScopedProvider as BorrowedProvider
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
    """Independent actual ingress streams; no previous-request ACK dependency."""
    address,nonce=c
    clients=[Client(address,nonce,'ingress')for _ in inputs]
    base=clock();errors=[];lock=threading.Lock()
    def one(i):
        target=base+arrival_plan['offsets_s'][i]
        try:
            assert digest(inputs[i])==arrival_plan['prompt_SHA256'][i]
            while True:
                remain=target-clock()
                if remain<=0:break
                sleep(min(remain,.001))
            sent=clock();got=clients[i].call(dict(op='arrive',prompt=inputs[i],limit=128))
            with lock:records.append(dict(producer_index=i,request=got['request'],target_t=target,send_t=sent,lateness_s=sent-target,actual_ingress_t=got['arrival']))
            assert got['accepted']
        except BaseException as e:
            with lock:errors.append(repr(e))
        finally:clients[i].close()
    threads=[threading.Thread(target=one,args=(i,),daemon=True)for i in range(len(inputs))]
    for t in threads:t.start()
    for t in threads:t.join(20)
    assert not any(t.is_alive()for t in threads)and not errors,errors
    assert sorted(v['request']for v in records)==list(range(len(inputs)))
    records.sort(key=lambda v:v['producer_index'])


def bounded_epoch(g,inputs,eos,reset,*,full_reference,label,port_delta,wait_s,arrival_plan):
    lib,ctx,r,node,tag=g['lib'],g['ctx'],g['r'],g['node'],g['tag']
    report,control,save=g['report'],g['control'],g['save']
    reset(label);start=time.perf_counter();report['phase']=label;save()
    address=(g['head'],g['port']+port_delta);nonce=os.environ['KIMI_ENGINE_NONCE']
    def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
    a=BorrowedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full_reference)
    broker=journal=capture=producer=None;errors=[];producer_schedule=[]
    event_path=r/'results'/f'Kimi-bounded-arrival-r8563-{tag}-{label}-events.jsonl'
    capture_path=r/'results'/f'Kimi-bounded-arrival-r8563-{tag}-{label}-jobs.jsonl'
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
            emit_fixed_arrivals((address,nonce),inputs,arrival_plan,producer_schedule)
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
        shared=dict(captured=captured,outputs=outputs,producer_schedule=producer_schedule,metric=metric,event_path=str(event_path),capture_path=str(capture_path))
    # All nodes read same shared immutable file, avoiding multi-MiB controls.
    path=r/'results'/f'Kimi-bounded-arrival-r8563-{tag}-{label}-shared.json'
    if node==0:
        with path.open('x')as f:json.dump(shared,f)
    signature=hashlib.sha256(path.read_bytes()).hexdigest() if node==0 else None
    signatures=control.collect(label+':capture-file-ready',dict(SHA256=signature))
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==signatures[0]['SHA256']
    return json.loads(raw)


