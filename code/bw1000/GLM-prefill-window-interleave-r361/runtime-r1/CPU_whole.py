"""Actual private owner/client/prepared route with recording Native/HTTP/processes.

CPU synthetic logits and semantic-gold HTTP wires are deliberate boundary mocks,
not Model outputs. No Native library, engine constructor, weights, GPU or Slurm.
"""
import ast,asyncio,copy,hashlib,importlib.machinery,inspect,json,os,pathlib,subprocess,sys,threading,time,traceback,types
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent;D=HERE/'owner';R=pathlib.Path('/refactor')
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];tag='GLM361-CPU-'+step
out=R/'results'/('GLM361-whole-'+step);out.mkdir(exist_ok=False)
result=dict(completed=False,passed=False,actual_CPU=step,CPU_only=True,CPU_source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),GPU=0,Native_dlopen=False,weights_opened=False,real_Slurm=False,
    real_HTTP=False,Native_constructor_called=False,Native_numeric_qualified=False,cases=[])
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    U=R/'candidates/latest-family-service-r270/GLM/python/infinilm'
    root=types.ModuleType('infinilm');root.__path__=[str(U)];root.__package__='infinilm';sys.modules['infinilm']=root
    class Forbidden(types.ModuleType):
        def __getattr__(self,n):raise RuntimeError('Native import forbidden '+n)
    core=Forbidden('infinilm._core');core.__file__='FORBIDDEN';core.__path__=[];core.__spec__=importlib.machinery.ModuleSpec(core.__name__,None);root._core=core;sys.modules[core.__name__]=core
    llm=types.ModuleType('infinilm.llm');llm.__path__=[str(U/'llm')];llm.__package__='infinilm.llm';sys.modules[llm.__name__]=llm
    for suffix,cls in [('scheduler','SchedulerOutput'),('static_scheduler','StaticSchedulerOutput')]:
        m=types.ModuleType('infinilm.llm.'+suffix);setattr(m,cls,type(cls,(),{}));sys.modules[m.__name__]=m
    class ForbiddenEngine:
        def __init__(self,*a,**k):raise RuntimeError('Native constructor forbidden')
    llm.AsyncLLMEngine=ForbiddenEngine
    def audit(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weights forbidden')
    sys.addaudithook(audit)
    import torch
    torch.set_num_threads(1)
    def guard(name):
     def unique(*a,**k):raise RuntimeError('GPU forbidden: '+name)
     return unique
    for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,guard(name))
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
    from infinilm.llm.sampling_params import SamplingParams
    from infinilm.llm.request import FinishReason
    llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason
    sys.path.insert(0,str(D))
    from source_pin_r290 import verify
    mf=sha(D/'SOURCE_MANIFEST.json');verify(mf)
    from prepared_family_entry_r268 import prepare_app
    prepared=prepare_app('GLM','/old/models/ZhipuAI-GLM-5.3')
    from infinilm.server.agent_protocol import prepare_agent_request
    import owner_r290 as owner,client_r355 as client
    from workload_r355 import PLAN,load
    fixtures=load(D)
    os.environ.update(GLM_RUN_R5040=tag,GLM_SOURCE_MANIFEST_SHA=mf,GLM_R290_GPUSTART_UNIX=str(time.time()),
        GLM_R361_CANCEL_FILE=str(R/'control'/('GLM-r361-cancel-'+tag+'.json')))
    engine=prepared['engine'];tape=[];cursor=[0];first=[True];errors=[];processes=[];requests_count=[0];request_lock=threading.Lock()
    class Response:
        status=200
        def __init__(self,raw):self.raw=raw
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def read(self):return self.raw
    def urlopen(req,timeout):
        f=next(f for f in fixtures if req.data==f['raw'])
        with request_lock:index=requests_count[0];requests_count[0]+=1
        data,schemas,choice=prepare_agent_request(copy.deepcopy(f['body']),'glm5');data.update(_agent_schemas=schemas,_agent_choice=choice)
        request=engine.add_chat_request(messages=data['messages'],sampling_params=prepared['server']._build_sampling_params(data),
            request_id='CPU361-%d'%index,request_data=data,add_generation_prompt=True,chat_template_kwargs=data.get('chat_template_kwargs')or{})
        assert request.metadata['real_cache_rows']==2
        async def consume():
            events=[]
            async for event in engine.stream_request(request,timeout=30,request_timeout=30):events.append(event)
            return events
        events=asyncio.run(consume());assert len(events)==1 and events[0].finished
        assert request.finish_reason==FinishReason('eos_token')and request.request_id not in engine._requests
        wire=dict(id=request.request_id,choices=[dict(finish_reason='tool_calls',message=dict(tool_calls=[dict(id='CPU-tool-'+str(index),type='function',function=dict(name=f['fixture']['gold']['tool'],arguments=json.dumps(f['gold'])))]))],
            usage=dict(prompt_tokens=len(request.prompt_token_ids),completion_tokens=len(request.generated_token_ids),total_tokens=len(request.prompt_token_ids)+len(request.generated_token_ids)))
        return Response(json.dumps(wire).encode())
    class Process:
        pid=7355001
        def __init__(self,argv,**kwargs):
            assert pathlib.Path(argv[3]).name=='client_r355.py'and kwargs['start_new_session']
            self.returncode=None;self.thread=threading.Thread(target=self.execute,args=(argv[4:],));processes.append(self);self.thread.start()
        def execute(self,args):
            try:
                with patch.object(sys,'argv',['client_r355.py']+args):self.returncode=client.main()
            except BaseException:errors.append(traceback.format_exc());self.returncode=2
        def poll(self):return self.returncode
        def wait(self,timeout=None):
            self.thread.join(timeout)
            if self.thread.is_alive():raise subprocess.TimeoutExpired('CPU client',timeout)
            return self.returncode
    class Server:
        def __init__(self,*a):self.started=False;self.should_exit=False;self.force_exit=False
        def run(self):
            self.started=True
            while not self.should_exit and not self.force_exit:time.sleep(.001)
    uv=types.ModuleType('uvicorn');uv.Config=lambda *a,**k:types.SimpleNamespace(**k);uv.Server=Server
    patches=[]
    for n in ['tensor','ones','arange','full','zeros']:
        fn=getattr(torch,n)
        def make(fn):
            def cpu(*a,**k):k.pop('device',None);return fn(*a,**k)
            return cpu
        patches.append(patch.object(torch,n,make(fn)))
    patches += [patch.object(torch.cuda,'synchronize',lambda *a:None),patch.object(torch.cuda,'mem_get_info',lambda *a:(32*1024**3,64*1024**3)),
        patch.dict(sys.modules,{'uvicorn':uv}),patch.object(subprocess,'Popen',Process),patch.object(os,'getpgid',lambda pid:pid),patch.object(os,'killpg',lambda *a:None),patch.object(client.urllib.request,'urlopen',urlopen)]
    fault={'kind':None}
    class Function:
        def __init__(self,name,lib):self.name=name;self.lib=lib
        def __call__(self,*a):
            l=self.lib;l.counts[self.name]=l.counts.get(self.name,0)+1
            if self.name=='modern_glm_model_transient_report_r5688':return b'{"rows":2,"capacity":32768,"enabled":false,"GPU_logical_bytes":0}'
            if self.name=='modern_glm_model_reset':l.past=[0,0];l.generated=[0,0]
            if self.name=='modern_glm_model_forward':
                row,past,T=tuple(a[3]);assert a[1:3]==(T,1)and past==l.past[row]
                assert a[5].tolist()==list(range(past,past+T));l.past[row]+=T
                if l.past[row]>=len(fixtures[row]['ids']):l.generated[row]+=1
                token=154820 if l.generated[row]>=(9 if row==0 else 11)else 100+row
                a[6].zero_();a[6][0,token]=1;l.trace.append(('F',row,past,T))
                if fault['kind']=='cancel'and len(l.trace)==5:pathlib.Path(os.environ['GLM_R361_CANCEL_FILE']).write_text('{}')
                if fault['kind']=='partial_error'and len(l.trace)==6:return 1
            if self.name=='modern_glm_cross_journal_r5775':return b'{"identities":[]}'
            return 0
    class Library:
        def __init__(self):self.functions={};self.counts={};self.past=[0,0];self.generated=[0,0];self.trace=[]
        def __getattr__(self,n):
            if n not in self.functions:self.functions[n]=Function(n,self)
            return self.functions[n]
    def record_control(x):
        caller=inspect.currentframe().f_back.f_code.co_name
        if first[0]or caller=='barrier':x.mul_(32);first[0]=False
        tape.append(x.clone())
    def replay_control(x):
        q=tape[cursor[0]];cursor[0]+=1;assert x.shape==q.shape;x.copy_(q)
    check=lambda rc:None if rc==0 else (_ for _ in ()).throw(RuntimeError(rc))
    payload={'benchmark':{'generation_config':{'eos_token_id':[154820,154827,154829]}}}
    for p in patches:p.start()
    try:
        emitter={'library_sha256':'038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'};lib=Library()
        owner.run(lib,object(),3,0,payload,record_control,check,lambda x:x,lambda:0,154880,emitter,out/'emitter.json',lambda *a:None,prepared)
        assert not errors and requests_count[0]==12 and all(p.returncode==0 for p in processes)
        assert len(emitter['pairs'])==6 and emitter['normal_HTTP_shutdown']and emitter['normal_reset_close']
        assert emitter['queue_journal']['max_accepted_unfinished']==2
        result['cases'].append(dict(name='actual_owner_client_prepared_pair_admissions',passed=True,requests=12,CPU_semantic_gold_wire_mock=True,synthetic_logits=True))
        for i,pair in enumerate(emitter['pairs']):
            hit=pair['prefill_consumption'];assert hit['qualified'] and pair['decode_defer_fixed_both_arms']
            assert hit['enabled']==PLAN[i]
            assert hit['complete_long_W4_groups']==(5 if PLAN[i]else 0)
            assert hit['short_advances']==(5 if PLAN[i]else 0)
            assert [r['tokens']for r in pair['requests']]==[9,11]
            assert pair['requests'][1]['generated_ids'][0]==101,'Long retained logits overwritten by intervening short forward'
            events=[e for e in pair['feedback_events']if e['phase']=='prefill']
            assert [e['row']for e in events]==([0,0,0,0,0,0,1]if PLAN[i]else[0,1])
        result['cases'].append(dict(name='actual_owner_boundary_hit_past_feedback_and_retained_logits',passed=True,ON_groups=5,ON_short_advances=5,OFF_advances=0,defer_fixed_ON=True,full_timeline_scope=True))

        # Actual private entry factory call and actual complete normal cleanup suffix.
        tree=ast.parse((D/'model_entry_r290.py').read_bytes());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
        ctor=next(n for n in worker.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ctx' for t in n.targets))
        start=next(i for i,n in enumerate(worker.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.slice,ast.Constant)and t.slice.value=='BLAS_policy'for t in n.targets))
        cleanup=compile(ast.Module(body=worker.body[start:],type_ignores=[]),'actual361-worker-normal-cleanup','exec')
        owners=[]
        for stage in range(4):
            for rank in range(8):
                class Factory:
                    def modern_glm_model_create(self,*a):
                        assert a[:4]==(rank,stage,2,32768);assert len(a)==8;return 'CPU_CTX'
                cfg={'num_hidden_layers':78};ns=dict(lib=Factory(),rank=rank,stage=stage,ROWS=2,CAPACITY=32768,json=json,cfg=cfg,buffers=[b'world',b'tp',b'control'])
                exec(compile(ast.Module(body=[ctor],type_ignores=[]),'actual361-create-call','exec'),ns);assert ns['ctx']=='CPU_CTX'
                if(stage,rank)==(3,0):row=emitter;library=lib;path=out/'emitter.json'
                else:
                    cursor[0]=0;row={'library_sha256':emitter['library_sha256']};library=Library();path=out/('stage%d-rank%d.json'%(stage,rank))
                    owner.run(library,object(),stage,rank,payload,replay_control,check,lambda x:x,lambda:0,154880,row,path,lambda *a:None,prepared)
                    assert cursor[0]==len(tape)
                env=dict(report=row,policy=types.SimpleNamespace(modern_blas_policy_r243_json=lambda:b'{"failures":0,"owners":[]}'),os=os,
                    host_stats=lambda:[0]*7,cache_stats809=lambda rank:dict(created=1,destroyed=1,active=0,active_bytes=0),service_stats=lambda:{},
                    lib=library,ctx=object(),check=check,rank=rank,m15_stats129=lambda:[0]*17,report_path=path,json=json,torch=torch)
                exec(cleanup,env)
                assert row['completed']and row['worker_shutdown_completed']and row['post_close_Torch_liveness']
                assert library.counts['modern_glm_model_destroy']==library.counts['modern_glm_model_shutdown']==1
                assert [(r['generated_IDs_SHA256'],r['tokens'])for p in row['pairs']for r in p['requests']]==[(r['generated_IDs_SHA256'],r['tokens'])for p in emitter['pairs']for r in p['requests']]
                owners.append(dict(stage=stage,rank=rank,rows=2,actual_entry_create_AST=True,actual_owner_run=True,actual_normal_cleanup_AST=True))
        result['cases'].append(dict(name='all32_factory_argv_actual_owner_and_cleanup',passed=True,owners=owners))
        for kind,expected_calls in [('cancel',5),('partial_error',6)]:
            prepared=prepare_app('GLM','/old/models/ZhipuAI-GLM-5.3');engine=prepared['engine']
            tag_negative='GLM361-CPU-'+step+'-'+kind
            os.environ.update(GLM_RUN_R5040=tag_negative,GLM_R361_CANCEL_FILE=str(R/'control'/('GLM-r361-cancel-'+tag_negative+'.json')))
            fault['kind']=kind;first[0]=True;tape.clear();cursor[0]=0;errors.clear();processes.clear()
            failure_report={'library_sha256':emitter['library_sha256']};failure_library=Library();failure_path=out/(kind+'.json')
            with patch.object(owner,'PLAN',(True,)),patch.object(client,'PLAN',(True,)):
                try:owner.run(failure_library,object(),3,0,payload,record_control,check,lambda x:x,lambda:0,154880,failure_report,failure_path,lambda *a:None,prepared)
                except (InterruptedError,RuntimeError)as error:failure_type=type(error).__name__
                else:raise AssertionError('Partial/cancel Native failure was silently continued')
            assert len(failure_library.trace)==expected_calls
            assert failure_library.counts['modern_glm_model_reset']==2,'No reset/replay after failed partial work'
            assert not failure_report['normal_shutdown'] and not failure_report['normal_reset_close'] and not failure_report['pairs']
            assert failure_report.get('service_traceback') and all(not p.thread.is_alive()for p in processes)
            if kind=='partial_error':assert failure_library.past[0]==480 and failure_library.generated[0]==2
            result['cases'].append(dict(name='actual_owner_'+kind+'_failstop',passed=True,Native_forward_calls=expected_calls,no_forward_or_reset_after_failure=True,frontend_not_replayed=True,error_type=failure_type,Native_is_recording=True))
        fault['kind']=None
        # Execute the actual new worker catch branch and original guarded exit2.
        owner_try=next(n for n in worker.body if isinstance(n,ast.Try)and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='_preloaded_probe_r3091'for x in ast.walk(n)))
        failure_lib=Library();failure_path=out/'actual-worker-failstop.json';failure_report={}
        def partial(*a):raise RuntimeError('CPU recording Native already advanced; forbidden ledger replay')
        env=dict(_preloaded_probe_r3091=partial,lib=failure_lib,ctx=object(),stage=3,rank=0,payload={},control=lambda *a:None,check=check,ptr=lambda x:x,stream=lambda:0,vocab=154880,report=failure_report,report_path=failure_path,phase_marker=lambda *a:None,_protocol_preweight={'prepared':prepared},traceback=traceback,json=json)
        try:exec(compile(ast.Module(body=[owner_try],type_ignores=[]),'actual361-worker-catch','exec'),env)
        except RuntimeError:pass
        else:raise AssertionError('Actual worker swallowed Native failure')
        assert failure_report['failed_owner_failstop'] and not failure_lib.counts
        guarded=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='guarded')
        class ExpectedExit(BaseException):pass
        def exit2(code):raise ExpectedExit(code)
        env=dict(worker=partial,traceback=traceback,r=R,tag='GLM361-CPU-'+step+'-guarded',stage=3,json=json,os=os)
        exec(compile(ast.Module(body=[guarded],type_ignores=[]),'actual361-guarded-exit','exec'),env)
        with patch.object(os,'_exit',side_effect=exit2):
            try:env['guarded'](0,{})
            except ExpectedExit as error:assert error.args==(2,)
            else:raise AssertionError('Original guarded failstop exit2 missing')
        result['cases'].append(dict(name='actual_worker_partial_error_branch_and_guarded_exit2',passed=True,no_Native_reset_destroy_or_shutdown_on_error=True,original_whole_set_controller_cleanup_required=True))

    finally:
        for p in reversed(patches):p.stop()
    from loader_guard_r290 import guard_loader
    guards=[]
    text="def load(ctx,report):\n with safe_open('CPU_METADATA',framework='pt'):\n  lib.modern_glm_model_load_batch(ctx,'[]')\n lib.modern_glm_model_finish(ctx)\n"
    class File:
        def __enter__(self):return self
        def __exit__(self,*a):return False
    guarded,counts=guard_loader(text);env=dict(lib=Library(),safe_open=lambda *a,**k:File(),__weight_deadline_check_r290__=lambda s:guards.append(s))
    exec(compile(guarded,'actual-original-loader-guard','exec'),env);env['load'](object(),{})
    assert guards==['safe_open','modern_glm_model_load_batch','modern_glm_model_finish']
    result['cases'].append(dict(name='actual_loader_boundary_guard',passed=True,counts=counts,actual_weights_opened=False))
    from deadline_r290 import check as original_check
    with patch.dict(os.environ,{'GLM_R290_GPUSTART_UNIX':str(time.time()-6121)}):
        try:original_check('CPU expired original residency')
        except TimeoutError:pass
        else:raise AssertionError('Original deadline silently reset')
    result['cases'].append(dict(name='original_absolute_deadline',passed=True,no_Native=True))
    assert not torch.cuda.is_initialized()and'GLM-PP4-registry-repair-r184b'not in pathlib.Path('/proc/self/maps').read_text()
    result.update(passed=True,owner_manifest_SHA256=mf,actual_prepared_identity=prepared['identity'],whole_entry_scope='Actual constructor-call AST, actual owner/client/prepared route, original loader guard, actual cleanup suffix; preweight/SDK/loader numerical operation remains reused source contract, not executed Native',complete_worker_body_mocked=False)


try:main()
except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
finally:
    result['completed']=True;result['controller_manifest_SHA256']=sha(HERE/'controller/SOURCE_MANIFEST.json')
    raw=json.dumps(result,indent=2).encode()+bytes([10]);(out/'CPU-result.json').write_bytes(raw)
    if result['passed']:
        candidate=R/'candidates/GLM-prefill-window-interleave-r361/runtime-r1';candidate.mkdir(parents=True,exist_ok=False);(candidate/'CPU-result.json').write_bytes(raw)
    print(json.dumps(result))
raise SystemExit(0 if result['passed']else 1)
