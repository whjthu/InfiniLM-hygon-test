"""CPU transport/packing contract; does not claim native or Model correctness."""
from pathlib import Path
import sys,json,hashlib,threading,time,concurrent.futures,ast
root=Path(__file__).resolve().parent;private=root/'GLM-chunk128-Model-private-r8872';sys.path.insert(0,str(private))
from glm_chunk128_schedule_r8872 import prompt_calls,pack_prompt,pack_decode,validate_events
from glm_continuous_feedback_r8872 import FeedbackHub,FeedbackClient,CheckedFeedback,safe_decode,tail_ready_metrics
from glm_large_schedule_r3627 import accept
gold=json.loads((private/'GLM477-frozen-gold-r8844.json').read_bytes())
prompts=gold['input_ids'][:16];expected=gold['generated_ids'][:16]
assert all(len(p)==512 for p in prompts)and all(len(g)==128 for g in expected)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
AST={p.name:ast.parse(p.read_text(encoding='utf-8'))for p in private.glob('*.py')}
probe=AST['glm_prefill_micro_probe_r8872.py']
node=next(n for n in ast.walk(probe)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='emit_generation'for t in n.targets))
expr=compile(ast.Expression(node.value),'<actual emit_generation predicate>','eval')
assert eval(expr,{},dict(step=0,chunk_tokens=128,prompt_offset=0,prompt_count=128))is False
assert eval(expr,{},dict(step=0,chunk_tokens=128,prompt_offset=384,prompt_count=128))is True
assert eval(expr,{},dict(step=0,chunk_tokens=512,prompt_offset=0,prompt_count=512))is True
for wrong in (lambda:pack_prompt(prompts,list(range(8)),128,512,0),lambda:pack_decode(prompts,[[]for _ in range(16)],list(range(8)),1,0)):
 try:wrong()
 except AssertionError:pass
 else:raise AssertionError('invalid metadata accepted')

def arm(chunk):
    hub=FeedbackHub('127.0.0.1',0,'cpu-chunk128',participants=16,emitter=8,timeout=20,idle_timeout=60)
    clients=[FeedbackClient(hub.address[0],hub.address[1],'cpu-chunk128',i,timeout=30)for i in range(16)]
    clients[8].request('begin','cpu',cohorts=[list(range(8)),list(range(8,16))])
    barrier=threading.Barrier(16,timeout=30)
    def worker(participant):
        stage=participant//8;generated=[[]for _ in range(16)];done=[False]*16
        times=[[]for _ in range(16)];pasts=[512]*16;native_lengths=[0]*16;trace=[];state={};anchor=time.perf_counter()
        checked=CheckedFeedback(clients[participant],tail_nonemitter=stage==1 and participant!=8)
        def native(desc,flat,pos,kind):
            barrier.wait()
            rows=[]
            for j in range(0,len(desc),3):
                row,past,count=desc[j:j+3];assert native_lengths[row]==past
                assert pos[len(rows)*count:(len(rows)+1)*count]==list(range(past,past+count))
                native_lengths[row]=past+count;rows.append(row)
            assert len(flat)==len(pos)==sum(desc[i]for i in range(2,len(desc),3))
            trace.append((kind,tuple(desc)));barrier.wait()
        def forward(rows,local_step,cohort):
            output_index=local_step+1
            desc,flat,pos=pack_decode(prompts,generated,rows,output_index,stage)
            native(desc,flat,pos,'decode')
            chosen=[expected[r][output_index]for r in rows]
            return(chosen if stage==1 else None,0.0)
        def validate(packet,rows):
            assert packet['ids']==[expected[r][len(generated[r])]for r in rows]
        def decode(**kwargs):
            return safe_decode(feedback=checked,case='cpu',is_emitter=participant==8,
                ids=generated,pasts=pasts,times=times,micro=8,limit=128,eos=set(),
                forward=forward,anchor=anchor,state=state,validate=validate,**kwargs)
        first_tokens=[]
        for rows,past,count in prompt_calls(16,chunk):
            desc,flat,pos=pack_prompt(prompts,rows,past,count,stage);native(desc,flat,pos,'prefill')
            if past+count==512:
                stamp=time.perf_counter()-anchor
                assert all(native_lengths[r]==512 and not generated[r]for r in rows)
                accept(generated,done,rows,[expected[r][0]for r in rows],set(),128)
                for r in rows:times[r].append(stamp)
                first_tokens.append((tuple(rows),len(trace)))
            else:assert all(not generated[r]for r in rows)
            if chunk==128 and rows[0]==8:decode(active_cohorts=[0],max_turns=1)
        assert state.get('next_step',[0,0])==([4,0]if chunk==128 else[0,0])
        prefill_trace_length=len(trace);assert all(native_lengths[r]>=512 for r in range(16))
        decode()
        assert generated==expected and state['next_step']==[127,127]
        assert native_lengths==[639]*16 and all(len(t)==128 for t in times)
        assert not checked.expected and not checked.in_forward
        saved_events=[]
        for kind,descriptor in trace:
            requests=[list(descriptor[i:i+3])for i in range(0,len(descriptor),3)]
            saved_events.append({'requests':requests,'rows':[q[0]for q in requests],
                'flat_tokens':sum(q[2]for q in requests),'round':0 if kind=='prefill'else requests[0][1]-511,
                'prompt_offset':requests[0][1]if kind=='prefill'else None,
                'prompt_count':requests[0][2]if kind=='prefill'else None})
        audited=validate_events(saved_events,generated,chunk)
        return {'participant':participant,'trace':trace,'first_tokens':first_tokens,'prefill_trace_length':prefill_trace_length,
                'generated':generated,'times':times,'state':state,'native_lengths':native_lengths,'schedule_audit':audited}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16)as pool:reports=list(pool.map(worker,range(16)))
        assert all(r['trace']==reports[0]['trace'] for r in reports)
        end=clients[8].request('end','cpu');assert not hub.pending and hub.case is None
        emitter=reports[8];metric=tail_ready_metrics(expected,emitter['times'],set(),max(t[-1]for t in emitter['times'])+0.001,16,128)
        assert metric['useful_output_tokens']==2048 and all(n==128 for n in map(len,emitter['times']))
        return {'chunk':chunk,'owners':len(reports),'identical_native_call_order':True,'prefill_calls':sum(k=='prefill'for k,_ in emitter['trace']),
                'decode_calls':sum(k=='decode'for k,_ in emitter['trace']),'early_decode_calls':sum(k=='decode'for k,_ in emitter['trace'][:emitter['prefill_trace_length']]),
                'first_token_publication':emitter['first_tokens'],'final_row_lengths':emitter['native_lengths'],
                'actual_saved_event_audit':emitter['schedule_audit'],
                'all_full128_transport_ids_match_fixture':True,'transport_end':end,
                'timing_scope':'CPU socket/metadata simulation only; no Model numerical or performance result',
                'trace':emitter['trace']}
    finally:
        for c in clients:c.close()
        hub.close()

out={'CPU_only':True,'GPU':False,'dlopen':False,'native_compile':False,'actual_protocol':'16real TCP clients + existing Hub/CheckedFeedback/safe_decode; actual updated decode_cohorts cursor resumption',
     'arms':[arm(512),arm(128)],'actual_consumer_predicate_cases':3,'metadata_negative_cases':2,
     'source_SHA256':{p.name:sha(p)for p in private.glob('*.py')},
     'remaining':'No native/GPU math exercised; no quality or speed claim. Resource/staging manifest not yet frozen.'}
p=root/'GLM-chunk128-CPUchecks-r8872.json';p.write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps({'path':str(p),'SHA256':sha(p),'arms':[{k:v for k,v in a.items()if k!='trace'}for a in out['arms']]}))
