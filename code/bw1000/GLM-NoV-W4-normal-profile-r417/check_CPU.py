"""Actual private CLI, prepared-queue and window consumers, recorded device leaves."""
from pathlib import Path
import argparse,ast,asyncio,copy,hashlib,importlib.util,json,os,subprocess,sys,tempfile
from types import SimpleNamespace
D=Path(__file__).resolve().parent;P=D.parent/'GLM-NoV-16card-W4-pair-r414/owner';sys.path.insert(0,str(P));sys.path.insert(0,str(D))
from service_policy import bind_prequeue,select,encode_profile_wire,decode_profile_wire,configure,stop_requested,verify_sources,ProfileHeaderMiddleware,_REQUEST_PROFILE
from longcontext_wire_r204 import BEGIN,WAIT,STOP,encode_wire,prefill_chunks
from glm_prefill_execute_r236 import execute_plan
from service_queue_bridge_r268 import ServiceQueueBridge
from prepared_history import make_prepared_history_prequeue
from prepared_family_entry_r268 import check_sampling
from profile_entry import loader_tree
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rejected(fn):
    try:fn()
    except(ValueError,AssertionError,TypeError):return
    raise AssertionError('Invalid input accepted')
def run():
    verify_sources(D,P);cases=[]
    assert select({})==('glm-nov16-w4',4)and select({'native_profile':'glm-nov16-w2'})==('glm-nov16-w2',2)
    for value in [None,True,2,4,2.,{},'W4','glm-nov16-w8']:rejected(lambda:select({'native_profile':value}))
    e={};configure('glm-nov16-w4',e);assert e['GLM_V_ON_DEMAND_R403']=='1'and e['GLM_PP_SIZE_R3603']=='2';rejected(lambda:configure('glm-nov16-w2',e));cases.append('Explicit profiles/defaultprivateW4/malformed rejection/environment conflict')
    for profile in ['glm-nov16-w2','glm-nov16-w4']:
        ids=list(range(32767));w=encode_profile_wire(1,BEGIN,ids,1,profile=profile);assert decode_profile_wire(list(map(float,w)),1,154880)==(BEGIN,ids,1,select({'native_profile':profile})[1])
        for pos,value in [(-1,8),(-2,0),(1,2)]:bad=w.copy();bad[pos]=value;rejected(lambda:decode_profile_wire(bad,1,154880))
    rejected(lambda:decode_profile_wire(encode_wire(1,BEGIN,[7],4096),1,154880))
    for op in [WAIT,STOP]:assert decode_profile_wire(encode_profile_wire(4,op),4,154880)==(op,[],0,None)
    cases.append('ActualFP32wire preserves32767completeIDs/budget;invalidoldprofile/seq rejects;WAIT/STOP')
    original=list(range(9741));plans=prefill_chunks(original);handoffs=[]
    for stage in range(2):
        for rank in range(8):
            for profile in ['glm-nov16-w2','glm-nov16-w4']:
                op,ids,budget,window=decode_profile_wire(encode_profile_wire(9,BEGIN,original,4096,profile=profile),9,154880);calls=[];votes=[];past=[0]
                def forward(p,v,f):assert p==past[0]and v==ids[p:p+len(v)];calls.append((p,len(v),f));past[0]+=len(v)
                def vote():votes.append(past[0]);return False
                assert execute_plan(prefill_chunks(ids),0,prefill=True,window=window,forward=forward,vote=vote,sample=lambda:17,discard=lambda:None)==(9741,17,False)
                assert len(calls)==20 and len(votes)==1+(20+window-1)//window
                handoffs.append(dict(stage=stage,rank=rank,profile=profile,window=window,prefill_calls=len(calls),votes=len(votes),complete_IDs=True,Native=False))
    cases.append('32actualwindowhandoffs samecomplete9741 across16owners W2/W4,recordedforwardonly')
    reasons=SimpleNamespace(TIMEOUT='timeout',ERROR='error',CANCELED='canceled');messages=[{'role':'user','content':'Retain complete history.'},{'role':'assistant','content':'Past reasoning.'},{'role':'tool','tool_call_id':'past-tool','content':'{"records":[1,2,3]}'}];expected=copy.deepcopy(messages);encoded=[]
    def encode(msg,data,generation,kwargs):
        encoded.append(copy.deepcopy(msg));return ''.join('<'+m['role']+'>'+m.get('content','')+'</'+m['role']+'>'for m in msg)+('<|assistant|><think>'if generation else''),[11,12,13]
    prefix_resolver=lambda data,kwargs:'<|assistant|><think>'
    history=make_prepared_history_prequeue(encode,prefix_resolver)
    family=ast.parse((P/'prepared_family_entry_r268.py').read_bytes());prepared_prequeue=next(n for n in ast.walk(family)if isinstance(n,ast.FunctionDef)and n.name=='prequeue')
    prepared_ns=dict(check_sampling=check_sampling,history=history,prefix_resolver=prefix_resolver,copy=copy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prepared_prequeue],type_ignores=[])),'actualpreparedprequeue-with-recordedencoder','exec'),prepared_ns);prequeue=prepared_ns['prequeue']
    engine=ServiceQueueBridge(None,32768,reasons,encode_chat=encode,decode_tokens=lambda ids:'CPU-only',eos_token_ids=[154820],default_max_tokens=4096,prequeue_factory=prequeue)
    untouched=ServiceQueueBridge(None,32768,reasons,encode_chat=encode,prequeue_factory=prequeue);old=untouched.prequeue_factory;bind_prequeue(engine)
    assert untouched.prequeue_factory is old and not getattr(untouched,'_r417_bound',False)
    params=SimpleNamespace(max_tokens=4096,temperature=0.,stop=None)
    for value in ['bad',None,4]:
        n=len(encoded);rejected(lambda:engine.validate_data({'native_profile':value,'messages':messages,'max_tokens':4096},params));assert len(encoded)==n and not engine._requests
    rejected(lambda:engine.validate_data({'messages':messages,'max_tokens':4097},SimpleNamespace(max_tokens=4097,temperature=0.,stop=None)))
    async def complete_stream(req):
        seen=[]
        async for token in engine.stream_request(req,timeout=1,request_timeout=2):seen.append(token)
        assert len(seen)==1 and seen[0].finished
    body=json.dumps(dict(messages=copy.deepcopy(messages),max_tokens=4096,user='same-complete-body')).encode();arrivals=[]
    async def endpoint(scope,receive,send):
        message=await receive();raw=message['body'];assert raw==body;data=json.loads(raw);before=copy.deepcopy(data);await asyncio.sleep(0);i=len(arrivals)
        engine.validate_data(data,params);req=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id='cpu-'+str(i),request_data=data)
        assert data==before and messages==expected and req.metadata['history_render_audit']['qualified']and req._admission_metadata_ready;arrivals.append(req.metadata['prefill_control_window'])
        decoded=decode_profile_wire(encode_profile_wire(100+i,BEGIN,req.prompt_token_ids,req.max_new_tokens,profile=req.metadata['native_profile']),100+i,154880)
        assert decoded==(BEGIN,list(req.prompt_token_ids),4096,req.metadata['prefill_control_window'])
        assert engine.take(0)is req;engine.record_started(req);req.emit_token(17);req.emit_token(154820,visible=False)
        engine.record_finished(req,2,True);req.finish('eos_token');await complete_stream(req);assert req.request_id not in engine._requests
        await send({'type':'http.response.start','status':200,'headers':[]});await send({'type':'http.response.body','body':b'CPU-recorded-leaves'})
    middleware=ProfileHeaderMiddleware(endpoint)
    async def request(headers):
        sent=[]
        async def receive():return {'type':'http.request','body':body,'more_body':False}
        async def send(value):sent.append(value)
        await middleware({'type':'http','headers':headers},receive,send);assert _REQUEST_PROFILE.get()is None
        return sent
    for i in range(12):
        expected_window=2 if i%2 else 4;sent=asyncio.run(request([(b'x-infinilm-native-profile',('glm-nov16-w'+str(expected_window)).encode())]));assert sent[0]['status']==200 and arrivals[-1]==expected_window
    before=len(arrivals)
    for headers in [[(b'x-infinilm-native-profile',b'bad')],[(b'x-infinilm-native-profile',b'glm-nov16-w2'),(b'x-infinilm-native-profile',b'glm-nov16-w4')],[(b'x-infinilm-native-profile',b'\xff')]]:
        sent=asyncio.run(request(headers));assert sent[0]['status']==400 and len(arrivals)==before and not engine._requests
    sent=asyncio.run(request([]));assert sent[0]['status']==200 and arrivals[-1]==4
    async def simultaneous():return await asyncio.gather(request([(b'x-infinilm-native-profile',b'glm-nov16-w2')]),request([(b'x-infinilm-native-profile',b'glm-nov16-w4')]))
    asyncio.run(simultaneous());assert set(arrivals[-2:])=={2,4}and _REQUEST_PROFILE.get()is None and not engine._requests
    async def fails(scope,receive,send):assert _REQUEST_PROFILE.get()=='glm-nov16-w2';raise RuntimeError('Recorded route exception after profile selection')
    async def exception_case():
        try:await ProfileHeaderMiddleware(fails)({'type':'http','headers':[(b'x-infinilm-native-profile',b'glm-nov16-w2')]},lambda:None,lambda m:None)
        except RuntimeError:pass
        else:raise AssertionError('Expected route exception')
        assert _REQUEST_PROFILE.get()is None
    asyncio.run(exception_case())
    cases.append('ActualASGIscope→headercontext→existingpreparedvalidate/prequeue/history→ServiceQueue→metadata→wiredecode;15identicalbodies/invalidheaders/default/concurrentisolation/exceptionreset;encoder/Nativeleavesrecorded')
    req=engine.add_chat_request(messages=messages,sampling_params=params,request_id='canceled',request_data={'messages':messages,'max_tokens':4096});req.mark_abort(reasons.CANCELED);assert engine.take(0)is req and req.cancel_event.is_set();engine.record_finished(req,0,True);req.finish(reasons.CANCELED);asyncio.run(complete_stream(req));engine.stop();assert engine.take(0)is None
    cases.append('Actualqueue canceledrequest retires without Native andstop sentinel')
    with tempfile.TemporaryDirectory(prefix='glm417-')as t:
        p=Path(t)/'stop.json';assert not stop_requested(p,'run','glm-nov16-w4');p.write_text(json.dumps(dict(normal_stop=True,run='run',profile='glm-nov16-w4')));assert stop_requested(p,'run','glm-nov16-w4');rejected(lambda:stop_requested(p,'other','glm-nov16-w4'))
    cases.append('Typedidentity normalstop marker')
    tree,proof=loader_tree(P);original=ast.parse((P/'model_entry_r290.py').read_bytes());original_assign=next(n for n in ast.walk(original)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='benchmark'for t in n.targets))
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom)and n.module=='service_owner':n.module='owner_r290'
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='benchmark'for t in n.targets):n.value=copy.deepcopy(original_assign.value)
    assert ast.dump(tree)==ast.dump(original);cases.append('ActualparentloaderAST exactlyequal afterundo2necessaryowner/payloadchanges')
    cli=[]
    for profile in ['glm-nov16-w2','glm-nov16-w4']:
        env={k:v for k,v in os.environ.items()if not k.startswith(('GLM_','CUDA_','HIP_','ROCR_'))};proc=subprocess.run([sys.executable,'-I','-B',str(D/'profile_entry.py'),'--profile',profile,'--preflight-only'],env=env,text=True,capture_output=True,timeout=20);assert proc.returncode==0,(proc.stdout,proc.stderr);c=json.loads(proc.stdout);assert c['preflight_passed']and c['GPU']==c['Model']==0 and not c['Native_dlopen'];cli.append(c)
    cases.append('ActualisolatedCLI bothprofiles throughsource/config/loaderAST beforedeviceboundary')
    return dict(completed=True,passed=True,cases=cases,window_handoffs=handoffs,actual_CLI=cli,parent_loader_proof=proof,Native=0,GPU=0,Slurm=0,Model=0,checkpoint_IO=False,actual_HTTP_server_executed=False,actual_template_encoder_executed=False,actual_ASGI_to_prepared_prequeue_to_wire=True,actual_prepared_prequeue_source_SHA256=sha(P/'prepared_family_entry_r268.py'),actual_queue_and_execute_plan=True,full_framework_and_mathematical_qualification_reused='Original414/850 only;newnormalowner andprofilewire notdevicequalified',source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()
    try:q=run()
    except BaseException as e:
        import traceback;q=dict(completed=True,passed=False,error=repr(e),traceback=traceback.format_exc(),GPU=0,Slurm=0,Native=0,Model=0)
    out.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(dict(output=str(out),SHA256=sha(out),passed=q['passed'])));raise SystemExit(0 if q['passed']else 1)
