"""Reuse existing finite TCP coordinator with real Engine reference/candidate.

No new scheduler. Reference emits CPU-argmax IDs+immutable actual job trace;
candidate emits live actuals, never a baseline. Boundedmemory logging stays in live coordinator timing; diskfsync followsdrain. Transport/client scope is explicitly NOT_MEASURED.
"""
import dataclasses,json,os,threading,time,hashlib,io
from online_broker_r8455 import Client,Journal
from online_service_core_r8455 import Scheduler
from kimi_online_protocol_r8457 import FiniteBroker,export
from kimi_matched_static_r8468 import BorrowedProvider
from kimi_matched_trace_r8467 import digest,seal_reference,read_reference,replay


class RecordingBroker(FiniteBroker):
    def __init__(self,*,capture,full_reference,**kwargs):
        self.capture,self.full_reference=capture,full_reference
        self.records=[]
        super().__init__(**kwargs)

    def handle(self,req,arrival,conn):
        if req['role']=='owner' and req['op']=='finish':
            stage=next(k for k,v in self.owner_connections.items()if v is conn)
            if stage==3:
                seq=req['seq'];assert seq==len(self.records)==self.core.cursor[stage]
                assert self.core.busy[stage]==seq
                job=json.loads(json.dumps(dataclasses.asdict(self.core.jobs[seq])))
                record=dict(job=job,job_SHA256=digest(job),actual_ids=req['ids'],
                            oracle=req.get('oracle'),coordinator_feedback_t=arrival)
                # BEFORE nativeCPU validation or state commit, retain actuals.
                assert len(self.records)<4096
                self.capture.write(json.dumps(record)+'\n');self.capture.flush()
                if self.full_reference:os.fsync(self.capture.fileno())
                self.records.append(record)
                if self.full_reference:
                    o=record['oracle'];assert o and o['finite'] and o['native_equals_CPU']
                    assert req['ids']==o['cpu_argmax_ids']==o['native_ids']
        return super().handle(req,arrival,conn)


class LiveBuffer:
    """Finite live-epoch journal, bounded16MiB. Persist AFTER actual drain.

    Reference/replay remain independently durable. User clocks never subtract
    elapsedIO; live steady token path has no diskflush/fsync by construction.
    """
    def __init__(self,path):self.path=path;self.buffer=io.StringIO();self.bytes=0;self.lock=threading.Lock()
    def write(self,text):
        with self.lock:
            n=len(text.encode());assert self.bytes+n<=16*1024**2,'live journal limit'
            self.buffer.write(text);self.bytes+=n
    def __call__(self,events):
        for event in events:self.write(json.dumps(event)+'\n')
    def flush(self):pass
    def close(self):
        with self.path.open('x')as f:
            f.write(self.buffer.getvalue());f.flush();os.fsync(f.fileno())
        self.buffer.close()


def actual_arrival_epoch(g,inputs,eos,reset,*,full_reference,label,port_delta):
    lib,ctx,r,node,tag=g['lib'],g['ctx'],g['r'],g['node'],g['tag']
    report,control,save=g['report'],g['control'],g['save']
    reset(label);start=time.perf_counter();report['phase']=label;save()
    address=(g['head'],g['port']+port_delta);nonce=os.environ['KIMI_ENGINE_NONCE']
    def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
    a=BorrowedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full_reference)
    broker=journal=capture=producer=None;errors=[]
    event_path=r/'results'/f'Kimi-arrival-r8472-{tag}-{label}-events.jsonl'
    capture_path=r/'results'/f'Kimi-arrival-r8472-{tag}-{label}-jobs.jsonl'
    if node==0:
        journal=Journal(event_path) if full_reference else LiveBuffer(event_path)
        capture=capture_path.open('x') if full_reference else LiveBuffer(capture_path)
        broker=RecordingBroker(capture=capture,full_reference=full_reference,nonce=nonce,journal=journal,
                               scheduler=Scheduler(stages=4,micro=4,max_users=8,max_inflight=4,rows=32,max_queue=8,eos=eos),
                               host=address[0],port=address[1])
    control.collect(label+':broker-ready',{})
    def produce():
        c=None
        try:
            c=Client(address,nonce,'ingress');deadline=time.monotonic()+120
            while len(broker.owner_connections)<4:
                if time.monotonic()>deadline:raise TimeoutError('owner attach')
                time.sleep(.01)
            # Real ingress frames. Independent producer thread retains arrival
            # while native owners run. Single finite burst with second half
            # after first dispatch; no simulated GPU work or replay timestamps.
            for i,prompt in enumerate(inputs):
                if i==4:
                    deadline=time.monotonic()+120
                    while broker.core.next_job==0:
                        if time.monotonic()>deadline:raise TimeoutError('first dispatch')
                        time.sleep(.01)
                got=c.call(dict(op='arrive',prompt=prompt,limit=128))
                assert got['accepted'] and got['request']==i
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
        assert metric['statuses']['completed']==8 and not metric['statuses']['unfinished'] and not metric['statuses']['rejected']
        report.setdefault('actual_arrival_epochs',[]).append(dict(label=label,full_logit_reference=full_reference,
             actual_outputs=outputs,metric=metric,event_path=str(event_path),capture_path=str(capture_path),
             event_SHA256=hashlib.sha256(event_path.read_bytes()).hexdigest(),capture_SHA256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
             phase_s=time.perf_counter()-start,qualified=False,
             scope='Real finite coordinator-observed epoch; LIVE71 has boundedmemory journal persisted afterdrain, no per-token diskfsync; reference/replay excluded fromservice timing; notclientdelivery/noABBA/no infinite service claim'))
        save()
        shared=dict(captured=captured,outputs=outputs,event_path=str(event_path),capture_path=str(capture_path))
    # All nodes read same shared immutable file, avoiding multi-MiB controls.
    path=r/'results'/f'Kimi-arrival-r8472-{tag}-{label}-shared.json'
    if node==0:
        with path.open('x')as f:json.dump(shared,f)
    signature=hashlib.sha256(path.read_bytes()).hexdigest() if node==0 else None
    signatures=control.collect(label+':capture-file-ready',dict(SHA256=signature))
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==signatures[0]['SHA256']
    return json.loads(raw)


def matched_arrival_then_live(g,inputs,eos,reset,identity,*,include_live=True):
    """After staticmatched pass, reference actualarrival -> exactreplay ->live.

    A changed live jobtrace gets a separate full-logit CPU oracle replay BEFORE
    its token-only measurements can be numerically qualified. The CPUoracle,
    not candidate actual IDs, defines the comparison. Stop atfirstdifference.
    """
    report,control,r,node,tag=g['report'],g['control'],g['r'],g['node'],g['tag']
    info=actual_arrival_epoch(g,inputs,eos,reset,full_reference=True,label='arrival-reference',port_delta=11000)
    header=dict(scope='MATCHED_SHAPE_JOBTRACE_ONLY',identity=identity,rows=32,reference_flags=67,candidate_flags=71,eos_token_ids=sorted(eos),output_limits={str(k):128 for k in range(8)})
    records=[dict(job=v['job'],job_SHA256=v['job_SHA256'],reference_ids=v['oracle']['cpu_argmax_ids'],oracle=v['oracle']) for v in info['captured']]
    p=r/'results'/f'Kimi-arrival-r8472-{tag}-reference.json'
    h=seal_reference(p,header,records) if node==0 else None
    h=control.collect('arrival-reference-sealed',dict(SHA256=h))[0]['SHA256']
    reference=read_reference(p,h,identity)
    def provider(full):
        def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
        return BorrowedProvider(g['lib'],g['ctx'],stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full)
    reset('arrival-reference-candidate-replay');a=provider(False)
    try:
        def persist(value):report['arrival_reference_replay']=value;g['save']()
        outcome=replay(reference,a,control.collect,persist)
    finally:a.close()
    report['arrival_reference_qualification']=outcome;g['save']()
    if not include_live:return
    live=actual_arrival_epoch(g,inputs,eos,reset,full_reference=False,label='live-token-only',port_delta=11050)
    reset('live-independent-full-logit-oracle-check');a=provider(True);checked=[]
    try:
        for v in live['captured']:
            job=v['job'];native=a.forward(job)
            parts=control.collect('live-oracle-'+str(job['seq']),dict(ids=native,oracle=a.reference_observation if node==3 else None))
            oracle=parts[3]['oracle'];expected=oracle['cpu_argmax_ids']
            row=dict(seq=job['seq'],rows=job['rows'],pasts=job['pasts'],live_actual_ids=v['actual_ids'],reference_oracle=oracle,
                     first_difference_row_index=next((i for i,(x,y)in enumerate(zip(v['actual_ids'],expected or []))if x!=y),None))
            checked.append(row);report['live_oracle_checks']=checked;g['save']() # BEFOREassert.
            assert oracle['finite'] and oracle['native_equals_CPU'] and v['actual_ids']==expected,'live trace mismatch: diagnostic only afterfirstdiff'
    finally:a.close()
    report['live_trace_matched_CPU_oracle_qualified']=True
    report['live_metrics_scope']='Finite realarrival token-only epoch, independently checked samejobtrace CPUoracle; coordinatorfeedback boundary; notclientdelivery/noABBA gain/infinite service'
    g['save']()
