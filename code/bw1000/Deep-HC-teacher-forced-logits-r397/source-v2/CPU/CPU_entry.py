"""Actual bootstrap→diagnostic callback; fake loaded Native leaves, no dlopen/GPU."""
import ast,copy,hashlib,importlib.util,json,os,pathlib,sys,threading,time,traceback,types
import numpy as np
from types import SimpleNamespace as NS
D=pathlib.Path(__file__).resolve().parent;S=D.parent/'owner';R=pathlib.Path('/refactor')
O=R/'candidates/Deep-HC-teacher-forced-logits-r397-CPU-v2'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'passed':False,'GPU':False,'Native_dlopen':False,'weights_opened':False,'new_compile':False,'cases':[]}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def actual_CPU_processor():
    # Run actual public tool normalization, processor method and history encoder
    # without importing the package's Native extension. Only tokenizer creation
    # is a leaf substitution: same actual tokenizer.json via Rust tokenizers.
    from tokenizers import Tokenizer
    class TokenizerCPU:
        @classmethod
        def from_pretrained(cls,path,**kwargs):return cls(path)
        def __init__(self,path):self.backend=Tokenizer.from_file(str(pathlib.Path(path)/'tokenizer.json'))
        def encode(self,prompt,add_special_tokens=False):assert add_special_tokens is False;return self.backend.encode(prompt,add_special_tokens=False).ids
        def __call__(self,prompt,add_special_tokens=False,**kwargs):
            assert add_special_tokens is False and not kwargs
            return {'input_ids':self.encode(prompt,add_special_tokens=False)}
    for name in ['infinilm','infinilm.server','infinilm.processors']:
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
    tr=types.ModuleType('transformers');tr.PreTrainedTokenizerFast=TokenizerCPU;sys.modules['transformers']=tr
    public=R/'candidates/latest-family-service-r270/Deep/python/infinilm'
    basic_path=public/'processors/basic_llm_processor.py';assert sha(basic_path)=='c1578883410b66565628f1405f4d78b315cf1640fe594d838a881b5a8f1b18a3'
    basic_tree=ast.parse(basic_path.read_bytes());call=next(n for n in ast.walk(basic_tree)if isinstance(n,ast.FunctionDef)and n.name=='__call__')
    assert hashlib.sha256(ast.dump(call,include_attributes=False).encode()).hexdigest()=='cb121e81a9c283ac816a737d3d0e29462a6a85c4195e5c4f99bb01faa8c806cc'
    call=copy.deepcopy(call);call.decorator_list=[];module=ast.Module(body=[call],type_ignores=[]);ast.fix_missing_locations(module);actual_basic={};exec(compile(module,str(basic_path),'exec'),actual_basic)
    basic=types.ModuleType('infinilm.processors.basic_llm_processor');basic.BasicLLMProcessor=type('BasicLLMProcessor',(),{'__call__':actual_basic['__call__']});sys.modules[basic.__name__]=basic
    public=R/'candidates/latest-family-service-r270/Deep/python/infinilm'
    agent=load('infinilm.server.agent_protocol',public/'server/agent_protocol.py')
    history=load('infinilm.processors.history_encoding',public/'processors/history_encoding.py')
    processor_path=public/'processors/deepseek_v4_processor.py'
    assert sha(processor_path)=='73cc8acb226b39390b1723c99362ab76a4b6ded841ee2715ba6d3272ceb16e1a'
    processor=load('infinilm.processors.deepseek_v4_processor',processor_path).DeepSeekV4Processor(str(R/'results/latest-three-processors-r260/Deep-model-metadata'))
    from prepared_chat import make_prepared_encode_chat
    encode=make_prepared_encode_chat(processor)
    return {'engine':NS(encode_chat=encode),'identity':{'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0','eos_token_ids':[1]}},dict(agent_SHA256=sha(public/'server/agent_protocol.py'),history_SHA256=sha(public/'processors/history_encoding.py'),processor_SHA256=sha(processor_path),Basic_call_source_SHA256=sha(basic_path),Basic_call_AST_SHA256='cb121e81a9c283ac816a737d3d0e29462a6a85c4195e5c4f99bb01faa8c806cc',Basic_call_actual_body_executed=True,tokenizer_leaf='actual Rust tokenizer.json; no Transformers/Native import')

try:
    O.mkdir(exist_ok=False);sys.path.insert(0,str(S));os.environ['DEEP_SOURCE_MANIFEST_SHA']=sha(S/'SOURCE_MANIFEST.json')
    mf=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in mf['files_SHA256'].items())
    from native_binding_r206b import binding
    binding_receipt=binding();assert binding_receipt[0]['library_SHA256']=='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1' and len(binding_receipt[2]['link']['full']['objects'])==49
    import bootstrap_r206b as boot,preweight_r206b,teacher_forced_r397 as diag
    tree,proof=boot.transform(boot.BASE.read_bytes());assert proof['Native_forward_AST_unchanged'] and proof['original_finally_destroy_retained'];q['actual_loader_AST']=proof
    # Confirm actual preserved forward ABI, not an inferred sampled-token hook.
    forwards=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='forward'];assert len(forwards)==1
    assert any(isinstance(n,ast.Return)and isinstance(n.value,ast.Tuple)and [getattr(x,'id',None)for x in n.value.elts]==['code','logits','emitted']for n in ast.walk(forwards[0]))
    prepared,processor_proof=actual_CPU_processor();payload,fixture=diag.load_private();prompt=diag.prepare_prompt(prepared,payload,fixture)
    assert len(prompt)==8489;q['actual_CPU_prompt']=dict(processor_proof,prompt_IDs_SHA256=diag.ids_sha(prompt),prompt_count=len(prompt),teacher_IDs_SHA256=diag.ids_sha(payload['teacher_token_ids']),teacher_count=83)
    expected=[prompt[k:k+1024]for k in range(0,8489,1024)]+[[v]for v in payload['teacher_token_ids']]
    from CPU_fake_hc import fake_lib
    for scenario in ['success','forward-failure','nonzero-code','nonfinite','counter-failure','reset-failure','expired','cleanup-failure','forward-and-cleanup-failure']:
        tag='CPU397-'+scenario;calls=[];lifecycle=[];stop=threading.Event();threads=[];state={'past':0,'sequence':-1,'resets':0}
        def fake_load_and_exec(actual_tree,namespace):
            assert namespace['__service_r206b__'] is diag.serve
            assert proof['Native_forward_AST_unchanged'] and actual_tree is not None
            lifecycle.append('create-leaf')
            for i in range(8):
                t=threading.Thread(target=stop.wait,name='CPU397-fake-Native-owner-'+str(i));t.start();threads.append(t)
            lifecycle.append('load-leaf')
            def reset(ctx):
                state['resets']+=1
                if scenario=='reset-failure':return 1
                state['past']=0;state['sequence']+=1;return 0
            lib=fake_lib(reset,'HC-hit-failure'if scenario=='counter-failure'else'success')
            cleanup_attempts=[];disarms={'owner':0,'inner':0};original_set=lib.deep_hc_model_epilogue_set_r366
            def set_mode(mode):
                is_cleanup=(scenario=='cleanup-failure'and len(lib.state['set_modes'])==4)or(scenario=='forward-and-cleanup-failure'and len(calls)>0)
                if is_cleanup:cleanup_attempts.append('HC_clear');raise RuntimeError('CPU injected HC clear exception')
                return original_set(mode)
            lib.deep_hc_model_epilogue_set_r366=set_mode
            def disarm(kind,v):
                assert v==0;disarms[kind]+=1
                if disarms[kind]>1:
                    cleanup_attempts.append(kind+'_disarm')
                    if scenario in ['cleanup-failure','forward-and-cleanup-failure']:raise RuntimeError('CPU injected '+kind+' disarm exception')
                return 0
            lib.deep_owner_arm_r8900=lambda v:disarm('owner',v)
            lib.deep_attention_inner_arm_r9193=lambda v:disarm('inner',v)
            def forward(ids,pos,rows,past,total,offsets):
                s=state['sequence'];step=sum(1 for v in calls if v['sequence']==s)
                assert 0<=s<4 and step<92 and ids==expected[step]
                assert rows==[0]and past==[state['past']]and total==[state['past']+len(ids)]and offsets==[0,len(ids)]and pos==list(range(past[0],total[0]))
                calls.append({'sequence':s,'step':step,'past':past[0],'tokens':len(ids),'input_IDs_SHA256':diag.ids_sha(ids)});state['past']=total[0]
                if scenario in ['forward-failure','forward-and-cleanup-failure']:raise RuntimeError('CPU injected Native forward leaf error')
                if scenario=='nonzero-code':return 7,np.zeros((1,129280),dtype=np.uint16),np.array([30722],dtype=np.int64)
                lib.advance(len(ids));logits=np.zeros((1,129280),dtype=np.uint16);logits[0,30722]=0x3f80;logits[0,28302]=0x3f7f
                if scenario=='nonfinite':logits[0,100]=0x7fc0
                return 0,logits,np.array([30722],dtype=np.int64)
            loaded=dict(__latest_prepared_r277__=prepared,tag=tag,np=np,lib=lib,ctx=object(),check=lambda x:(_ for _ in ()).throw(RuntimeError('CPU nonzero CAPI'))if x else None,forward=forward,cfg={'num_hidden_layers':43,'vocab_size':129280},torch=NS(cuda=NS(mem_get_info=lambda d:(29*1024**3,64*1024**3))),__bootstrap_proof_r206b__=proof,__provider_proof_r206b__={'CPU_fake_GPU_provider_leaf':True},__create_proof_r206b__={'CPU_fake_Native_create_load_leaves':True})
            try:namespace['__service_r206b__'](loaded)
            finally:
                lifecycle.append('destroy-leaf');stop.set()
                for t in threads:t.join(2)
                namespace.update(tag=tag,report={'engine_worker_shutdown_completed':True},__owner_exit_observation_r206b__=loaded.get('__owner_exit_observation_r206b__',{}))
                legacy=O/('legacy-'+scenario+'.json');legacy.write_text(json.dumps({'engine_worker_shutdown_completed':True}));namespace['path']=legacy
            assert lib.state['mode']==0 and not any(t.is_alive()for t in threads)
            q.setdefault('mode_calls',{})[scenario]=lib.state['set_modes']
        preweight_r206b.run=lambda:{'proof':{'CPU_fake_preweight_provider_leaf':True},'prepared':prepared}
        boot.execute_loader=fake_load_and_exec
        diag.owner_threads=lambda ns:{'status':'CPU_FAKE_NATIVE_OWNERS','owners':[{'device':i,'tid':t.native_id}for i,t in enumerate(threads)]}
        os.environ['DEEP_R277_GPUSTART_UNIX']='0'if scenario=='expired'else str(time.time())
        error=None
        try:boot.main()
        except BaseException as exc:error=repr(exc)
        assert (error is None)==(scenario=='success'),(scenario,error)
        assert lifecycle==['create-leaf','load-leaf','destroy-leaf']and len(threads)==8 and not any(t.is_alive()for t in threads)
        p=R/'results'/('Deep-HC-teacher-forced-logits-r397-'+tag+'.json');actual=json.loads(p.read_bytes())
        assert actual['HTTP_requests']==0 and actual['quality_PASS']is False and actual['performance_PASS']is False
        if scenario=='success':
            assert actual['completed']and actual['passed']and actual['normal_Native_shutdown']and actual['owner_exit_observation']['all8_unique_threads_exited']
            assert len(calls)==actual['actual_forwards']==368 and state['resets']==actual['actual_resets']==4
            assert [x['HC_mode']for x in actual['sequences']]==[0,1,1,0]
            assert all(x['prefill_calls']==9 and x['decode_calls']==83 and len(x['decisions'])==84 and x['last_total']==8572 for x in actual['sequences'])
            assert q['mode_calls'][scenario]==[0,1,1,0,0]
            assert all(calls[s*92+k]['input_IDs_SHA256']==calls[k]['input_IDs_SHA256']for s in range(4)for k in range(92))
        else:
            assert not actual['passed']and not actual['normal_Native_shutdown']and actual['error']
            assert actual['completed']is True
            if scenario!='cleanup-failure':assert len(actual['sequences'])<4 and actual['actual_forwards']<368
            if scenario in ['forward-failure','forward-and-cleanup-failure','nonzero-code']:assert actual['actual_forward_attempts']==1 and actual['actual_forwards']==0 and actual['sequences'][0]['prefill_calls']==0
            if scenario=='forward-and-cleanup-failure':assert 'Native forward leaf error'in actual['primary_error'] and len(actual['cleanup_errors'])==3
        if scenario in ['cleanup-failure','forward-and-cleanup-failure']:
            assert len(actual['cleanup_errors'])==3 and actual['HC_after_clear']==0 and actual['observer_disarm_rc']==[None,None]
        else:assert actual['HC_clear_rc']==actual['HC_after_clear']==0 and actual['observer_disarm_rc']==[0,0]
        assert actual['actual_step']==os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']and actual['legacy_tag']==tag
        q['cases'].append(dict(case=scenario,passed=True,actual_forward_leaf_calls=len(calls),actual_reset_leaf_calls=state['resets'],actual_error=error,eight_fake_Native_owner_threads_exited=True,actual_bootstrap_to_module_callback=True,original_Native_forward_AST_preserved=True))
    logits=np.zeros((1,129280),dtype=np.uint16);logits[0,30722]=logits[0,28302]=0x3f80
    tie=diag.logits_summary(np,logits,[30722],83,payload['teacher_token_ids'],129280)
    assert tie['top1_margin']==0 and tie['maximum_tie_count']==2 and tie['selected_scores']['native_argmax']['id']==30722 and tie['selected_scores']['independent_argmax']['id']==28302
    q['cases'].append({'case':'BF16_exact_expansion_and_valid_native_argmax_tie','passed':True})
    assert 'torch'not in sys.modules and '_infinilm'not in sys.modules
    q['passed']=True
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
q['completed']=True;O.mkdir(exist_ok=True);p=O/'CPU-entry.json';p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','error']}));assert q['passed']
