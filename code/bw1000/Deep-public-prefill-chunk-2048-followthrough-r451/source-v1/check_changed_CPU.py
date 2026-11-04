"""Only changed 1024/2048 public worker/descriptor/client handoff, no device."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy,hashlib,json,os,sys,tempfile,threading,time
import numpy as np
from public_deep_model import worker
from public_deep_model.request_policy import RequestPolicy,CHUNKS,PROMPT_SHA
from public_deep_model.chunk_contract import descriptor_for_index
from public_deep_model.native_binding import binding
from recording_leaves import Context,Request,Engine,Run

D=Path(__file__).resolve().parent

def worker_case(scenario='six-chunks'):
    context=Context(scenario);requests=[Request(i,scenario,context)for i in range(6 if scenario=='six-chunks'else 1)]
    cfg=dict(gpu_start_epoch=time.time(),service_seconds=2400,r451_chunks=list(CHUNKS),r451_output_budget=128,r451_r366_fixed=0,r451_fullpre_fixed=0)
    if scenario=='cancel-before':requests[0].cancel_event.set();requests[0].abort_reason='canceled'
    if scenario=='cancel-midprefill':
        original=context.forward
        def forward(*args):
            result=original(*args);requests[0].cancel_event.set();requests[0].abort_reason='canceled';return result
        context.forward=forward
    report={'requests':[]};reasons=NS(LENGTH='length',EOS_TOKEN='eos_token',CANCELED='canceled')
    def factory(c,config):return RequestPolicy(c,config,provider_validator=lambda lib:dict(actual_DSO=False,recording_CABI=True))
    with patch.object(worker,'RequestPolicy',factory):worker.serve(context,{'engine':Engine(requests),'FinishReason':reasons,'identity':{'family':'Deep','upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0','eos_token_ids':[1]}},cfg,Run(),threading.Event(),report,lambda:None)
    assert context.binding.lib.mode==context.binding.lib.old_mode==0
    if scenario=='six-chunks':
        assert [r['r451_chunk']for r in report['requests']]==list(CHUNKS)and context.calls==54
        for i,(r,req)in enumerate(zip(report['requests'],requests)):
            assert req.generated_token_ids==[9,9,1]and req.finish_reason=='eos_token'
            assert r['normal_reset_before']and r['normal_reset_after']and r['prefill_binding']['arm_index']==i
            t=[t for t in r['Native_continuation']if t['kind']=='prefill'];assert [x['tokens']for x in t]==[CHUNKS[i]]*(8489//CHUNKS[i])+[297]
            assert r['actual_forward_completed']==len(t)+2 and r['r451_actual_T1_coverage']['mode']==0
    else:
        assert not requests[0].generated_token_ids and requests[0].finish_reason=='canceled'
        assert context.calls==(0 if scenario=='cancel-before'else 1)and report['requests'][0]['normal_reset_after']
    return report,requests,dict(scenario=scenario,passed=True,actual_worker_serve=True,actual_current_generate=True,Native_recording_calls=context.calls,actual_Native=False,GPU=False)

def check():
    b,receipt,norm=binding();assert b['library_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and len(norm['link']['full']['objects'])==50
    assert [descriptor_for_index(b,i)['prefill_chunk']for i in range(6)]==list(CHUNKS)
    for bad in [-1,6,True,'1']:
        try:descriptor_for_index(b,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid new descriptor index accepted')
    outcomes=[];report,requests,o=worker_case();outcomes.append(o)
    # Same actual continuation rejects overflow/cap/deadline before any Native leaf.
    from public_deep_model.native_decode import generate
    for kind,kwargs in [('capacity',dict(context=8500,chunk_size=2048,absolute_deadline=time.time()+30)),('cap',dict(context=32768,chunk_size=4097,absolute_deadline=time.time()+30)),('deadline',dict(context=32768,chunk_size=2048,absolute_deadline=time.time()-1))]:
        c=Context(kind);req=Request(0,kind,c)
        try:generate(req,c.forward,c.binding.check,lambda x:None,NS(LENGTH='length',EOS_TOKEN='eos_token',CANCELED='canceled'),vocab=128,**kwargs)
        except (ValueError,TimeoutError):pass
        else:raise AssertionError('Actual continuation accepted '+kind)
        assert c.calls==0
    outcomes.append(dict(scenario='actual_continuation_cap_capacity_absolute_deadline',passed=True,Native_recording_calls=0,internal_GPU_padding_or_memory_qualification=False))
    for s in ['cancel-before','cancel-midprefill']:outcomes.append(worker_case(s)[2])
    import client,fixture,client_common
    card=json.loads((D/'FIXTURE_CARD.json').read_bytes());raw,body,gold,_=fixture.body_and_gold(card,card['plan'][0])
    # Actual completebody/history is read/hashed; generation/report metadata is
    # a recording leaf, not a new tokenizer/Native qualification.
    original_fixture=json.loads(Path(card['fixture_path']).read_bytes());assert original_fixture['actual_prompt_token_ids_SHA256']==PROMPT_SHA
    for r in report['requests']:
        h=r['admission']['history_render_audit'];h.update(qualified=True,prequeue_bound=True,same_prepared_encoder_reproduces_prompt_and_IDs=True,markers_sent_to_Native=False,content_one_contribution_each_and_ordered=True,all_content_contribution_slices_equal=True,output_budget=128,prompt_tokens=8489)
        r['final_metadata']=dict(buffered_P0_text=False)
    class Response:
        status=200;headers={'Content-Type':'text/event-stream'}
        def __init__(self,rows):self.rows=rows
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):return iter(self.rows)
    for broken in [False,True]:
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);runtime=root/'run';runtime.mkdir();identity=dict(run='run',actual_step='CPU451.recording',source_manifest_SHA256='CPU-source',provider_config_SHA256='CPU-config')
            client_common.save(runtime/'MODEL_READY.json',dict(identity=identity,Model_READY=True,Native_internal_owner_count=8,HC_mode=0,TP=8,PP=1));client_common.save(runtime/'FRONTEND.json',dict(identity=identity,PID=os.getpid()))
            pending=copy.deepcopy(report);pending['identity']=identity
            if broken:pending['requests'][0]['prefill_binding']['prefill_chunk']=4096
            client_common.save(runtime/'owner.json',pending);seen=[]
            def urlopen(req,timeout):
                i=len(seen);assert i<6;expected_raw,_,_,_=fixture.body_and_gold(card,card['plan'][i]);assert req.data==expected_raw;seen.append(i)
                message=dict(role='assistant',content=json.dumps(gold));usage=dict(prompt_tokens=8489,completion_tokens=3,total_tokens=8492);ident=report['requests'][i]['request_id']
                events=[dict(id=ident,choices=[dict(index=0,delta=message,finish_reason=None)]),dict(id=ident,choices=[dict(index=0,delta={},finish_reason='stop')]),dict(id=ident,choices=[],usage=usage)]
                return Response([('data: '+json.dumps(e)+'\n\n').encode()for e in events]+[b'data: [DONE]\n\n'])
            with patch.object(client.urllib.request,'urlopen',urlopen),patch.object(client,'wait_listener',lambda pid,port,deadline:None):
                rc=client.main(['--base-url','http://CPU-recording/v1','--runtime-dir',str(root),'--run','run','--expected-step','CPU451.recording','--source-sha','CPU-source','--config-sha','CPU-config','--fixtures',str(D/'FIXTURE_CARD.json'),'--output-dir',str(root/'out'),'--deadline-unix',str(time.time()+180)])
            q=json.loads((root/'out/result.json').read_bytes())
            assert q['completed']and q['planned_HTTP']==6
            assert(rc==0 and q['passed']and q['completed_HTTP']==6 and seen==list(range(6)))if not broken else(rc==1 and not q['passed']and q['attempted_HTTP']==1 and q['completed_HTTP']==0)
            outcomes.append(dict(scenario='actual_client_worker_file_handoff'+('-wrongdescriptor'if broken else''),passed=True,actual_client_main=True,HTTP_transport_and_Native_recording_only=True,actual_wire_parser=True,requests=len(seen),actual_Native=False))
    assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
    return dict(passed=True,cases=outcomes,actual_new_binding50_metadata=True,actual_worker_and_client=True,Native_dlopen=False,GPU=False,old449_945_suites_repeated=False,actual_prompt_body_SHA256=hashlib.sha256(raw).hexdigest(),actual_Native_or_Model=False)

if __name__=='__main__':print(json.dumps(check()))
