"""One already-loaded real Engine per PP stage, TP8 each; finite single epoch.

Reuse r8235's verified loader/owner destroy path. No math compilation or HTTP.
Actual source/dependency and GPU qualification are still required before launch.
"""
import ctypes,hashlib,json,pathlib,threading,time
from online_broker_r8455 import Broker,Client,Journal
from online_service_core_r8455 import Scheduler
from online_service_metrics_r8455 import evaluate
from online_kimi_adapter_r8455 import KimiLoadedAdapter


class FiniteBroker(Broker):
    def handle(self,req,arrival,conn):
        if req['role']=='ingress' and req['op']=='seal':
            assert self.accepting and self.next_request==8
            self.accepting=False
            return dict(sealed=True)
        if req['role']=='owner' and req['op']=='next':
            if not self.accepting and self.core.can_reset_epoch() and not self.core.queue:
                return dict(wait=False,stop=True)
        return super().handle(req,arrival,conn)


class BorrowedAdapter(KimiLoadedAdapter):
    """Engine ownership stays with old r8235 runner's checked finally destroy.

    close here closes this adapter only. No broker receipt proves native destroy;
    outer runner report+actual terminal/postKFD must prove all owner cleanup.
    """
    def close(self):self.closed=True


def export(events,cards):
    events=[dict(e) for e in events]
    merged=[]
    for i,e in enumerate(events):
        if e['kind']=='ingress_rejection':
            key='transport-rejected-'+str(i)
            merged.extend([dict(kind='arrival',t=e['t'],request=key),
                           dict(kind='terminal',t=e['t'],request=key,status='rejected')])
        else:merged.append(e)
    merged.sort(key=lambda e:e['t'])
    arrivals=[e.get('arrival_t',e['t']) for e in merged if e['kind']=='arrival']
    assert arrivals and merged
    result=evaluate(dict(events=merged,window=[min(arrivals),max(e['t'] for e in merged)],cards=cards,boundary='emitter'))
    result['measurement_scope']='Coordinator full-frame ingress to coordinator-observed emitter feedback; includes feedback transport/control/journal wait. Not actual client delivery or aligned GPU timeline.'
    return result,merged


def run(g):
    lib,ctx,control,report=g['lib'],g['ctx'],g['control'],g['report']
    node,r,tag,check=g['node'],g['r'],g['tag'],g['check']
    assert (g['tp_nodes'],g['pp'])==(1,4)
    assert lib.modern_kimi_vocab_head_get_r8341()==0
    import os
    assert os.environ['INFINILM_KIMI_N4_TILE_R8167']=='0'
    frozen_path=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json'
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625'
    frozen=json.loads(frozen_path.read_text())
    chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text())
    known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
    eos=set(chat['eos_token_ids']);assert eos==set(frozen['eos_token_ids'])
    def gold_for(task):
        key='sort' if task.startswith('sort_probe') else task
        return [v['generated_ids'][0] for v in known['cases'] if v['task']==key and v['grouped']]
    early=[(task,seq) for task in chat['cases'] for seq in gold_for(task) if len(seq)<128 and seq[-1] in eos]
    assert early,'No independently frozen earlyEOS QA prompt available; do not fabricate EOS'
    early_task,early_gold=min(early,key=lambda x:len(x[1]))
    history=next(v for v in frozen['geometries'] if (v['B'],v['M'])==(16,4))['fixed_complete_IDs']
    inputs=[frozen['natural_input_ids'][i%2] for i in range(7)]+[chat['cases'][early_task]['input_ids']]
    gold=history[:7]+[early_gold]
    report.update(online_protocol='r8457',GPU_qualification=False,QA=[],early_EOS_task=early_task,
                  scope='Real TP8PP4 finite N8/Mmax4 dynamic arrival; full frozen IDs/EOS required; no HTTP/client claims',
                  online_inputs_SHA256=hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest())
    def memgate():
        assert all(free>512*1024**2 for free,total in g['memory']()),'512MiB forward floor'
    def adapter():
        return BorrowedAdapter(lib,ctx,stage=node,stages=4,rows=32,check=check,memory_gate=memgate,destroy=lambda p:None)
    def reset(label):
        check(lib.modern_kimi_engine_reset_r2475(ctx))
        control.collect('online-reset:'+label,{'all8owners_native_reset_completed':True})
        assert all(free>5*1024**3 for free,total in g['memory']()),'phase5GiBreserve'
    def static(prompts,label,expected,micro=4):
        reset(label);a=adapter();seq=0;outputs=[[] for _ in prompts];pasts=[0]*len(prompts)
        def forward(indices,ins):
            nonlocal seq
            j=dict(seq=seq,kind='prefill' if all(pasts[i]==0 for i in indices) else 'decode',
                   keys=indices,rows=[i+1 for i in indices],pasts=[pasts[i] for i in indices],inputs=ins)
            ids=a.forward(j)
            packet=control.collect('static:'+label+':'+str(seq),{'ids':ids})
            ids=packet[3]['ids'];assert ids and len(ids)==len(indices)
            for i,w,v in zip(indices,map(len,ins),ids):outputs[i].append(v);pasts[i]+=w
            seq+=1
        for offset in range(0,len(prompts),2):
            ii=list(range(offset,min(offset+2,len(prompts))));forward(ii,[prompts[i] for i in ii])
        for step in range(127):
            active=[i for i,v in enumerate(outputs) if len(v)<128 and v[-1] not in eos]
            if not active:break
            for offset in range(0,len(active),micro):
                ii=active[offset:offset+micro];forward(ii,[[outputs[i][-1]] for i in ii])
        assert all((len(v)==128 or v[-1] in eos) and not any(t in eos for t in v[:-1]) for v in outputs)
        if len(prompts)==1:assert outputs[0] in expected,(label,'complete_frozen_QA_ID_mismatch')
        else:assert outputs==expected,(label,'complete_frozen_static_ID_mismatch')
        a.close()
        return outputs
    for task,c in chat['cases'].items():
        static([c['input_ids']],'QA-before-'+task,gold_for(task),1)
        report['QA'].append(dict(epoch='before',task=task,complete_frozen_IDs_EOS=True));g['save']()
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='1'
    control.collect('static-serving-policy1',{})
    static(inputs,'N8-static-history',gold,4)
    report['static_N8_frozen_IDs_EOS']=True;g['save']()
    reset('dynamic-single-epoch')
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='1'
    control.collect('online-serving-policy1',{})
    broker=journal=producer=None;producer_errors=[]
    journal_path=r/'results'/f'Kimi-online-r8457-{tag}-events.jsonl'
    if node==0:
        journal=Journal(journal_path)
        broker=FiniteBroker(nonce=os.environ['KIMI_ENGINE_NONCE'],journal=journal,
                            scheduler=Scheduler(stages=4,micro=4,max_users=8,max_inflight=4,rows=32,max_queue=8,eos=eos),
                            host=g['head'],port=g['port']+11000)
    control.collect('online-broker-ready',{})
    def produce():
        client=None
        try:
            client=Client((g['head'],g['port']+11000),os.environ['KIMI_ENGINE_NONCE'],'ingress')
            deadline=time.monotonic()+120
            while len(broker.owner_connections)<4:
                if time.monotonic()>deadline:raise TimeoutError('owner attach')
                time.sleep(.01)
            for i,prompt in enumerate(inputs):
                if i==4:
                    deadline=time.monotonic()+120
                    while not broker.core.busy[0] and broker.core.next_job==0:
                        if time.monotonic()>deadline:raise TimeoutError('first actual job')
                        time.sleep(.01)
                response=client.call(dict(op='arrive',prompt=prompt,limit=128))
                assert response['accepted'] and response['request']==i
            client.call(dict(op='seal'))
        except BaseException as exc:
            producer_errors.append(repr(exc));broker.poison('producer: '+repr(exc))
        finally:
            if client:client.close()
    if node==0:
        producer=threading.Thread(target=produce,daemon=True);producer.start()
    client=None;a=adapter()
    try:
        client=Client((g['head'],g['port']+11000),os.environ['KIMI_ENGINE_NONCE'],'owner',node)
        while True:
            response=client.call(dict(op='next'))
            if response.get('stop'):break
            if response['wait']:time.sleep(.002);continue
            job=response['job'];ids=a.forward(job)
            client.call(dict(op='finish',seq=job['seq'],ids=ids))
        client.call(dict(op='bye'))
        report['online_owner_loop_drained']=True
    finally:
        a.close()
        if client:client.close()
    control.collect('online-all-owner-loop-drained',{})
    if node==0:
        try:
            producer.join(5);assert not producer.is_alive() and not producer_errors
            assert not broker.failure and broker.core.can_reset_epoch()
            broker.flush()
        finally:
            broker.close();journal.close()
        events=[json.loads(v) for v in journal_path.read_text().splitlines()]
        metric,merged=export(events,32)
        outputs=[[e['token'] for e in events if e['kind']=='token' and e.get('request')==i] for i in range(8)]
        assert outputs==gold,'dynamic complete frozen IDs mismatch'
        assert outputs[7][-1] in eos and len(outputs[7])<128
        assert metric['statuses']['completed']==8 and not metric['statuses']['rejected'] and not metric['statuses']['unfinished']
        report.update(online_outputs=outputs,online_metric=metric,online_events_SHA256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),online_complete_frozen_IDs_EOS=True)
        g['save']()
    control.collect('online-qualification-recorded',{})
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='0'
    for task,c in chat['cases'].items():
        static([c['input_ids']],'QA-after-'+task,gold_for(task),1)
        report['QA'].append(dict(epoch='after',task=task,complete_frozen_IDs_EOS=True));g['save']()
    report['serving_protocol_complete']=True
    report['GPU_qualification_pending_independent_terminal_audit']=True
    g['save']()
