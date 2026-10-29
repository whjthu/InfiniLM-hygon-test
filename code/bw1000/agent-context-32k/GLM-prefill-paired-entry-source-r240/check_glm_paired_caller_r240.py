"""Reuse existing source-derived forward/control witnesses; no runtime/CCL/Model."""
import ast,datetime,hashlib,inspect,json,pathlib,sys,textwrap,types
R=pathlib.Path(sys.argv[1]).resolve();D=pathlib.Path(__file__).parent
B=R/'code/bw1000/agent-context-32k/GLM-prefill-executor-source-r236'
W1=R/'code/bw1000/agent-context-32k/GLM-prefill-window2-source-r229/glm_agent_service_baseline_r204.py'
output=pathlib.Path(sys.argv[2]);assert not output.exists()
sys.path.insert(0,str(D));sys.argv=[str(__file__),str(R)]

def load_case(path,new=False):
    raw=path.read_text();raw=raw[:raw.index('\nout=[]')]
    if new:
        raw=raw.replace("source=(D/'glm_agent_service_window2_r236.py').read_text()","source=(D/'glm_agent_service_window12_r240.py').read_text()")
        raw=raw.replace('eos=()):','eos=(),window=2):')
        raw=raw.replace('env=dict(execute_plan=','env=dict(prefill_window=window,execute_plan=')
    m=types.ModuleType('candidate' if new else 'reference');m.__file__=str(D/'check.py' if new else path)
    exec(compile(raw,str(path),'exec'),m.__dict__);return m
old=load_case(B/'check_glm_window2_r236.py');new=load_case(B/'check_glm_window2_r236.py',True)
counts={'actual_W2_normal':0,'actual_W2_cancel_or_timeout':0,'actual_W2_forward_exception':0,'actual_W1_normal_vs_original_W1':0}

def compare(kwargs,category):
    a,x=old.case(**kwargs);b,y=new.case(**kwargs)
    assert a==b and x['generated']==y['generated'] and x['reason']==y['reason']
    assert x['calls']==y['calls'] and x['past']==y['past']
    counts[category]+=1
for n in [1,511,512,513,1024,1025,1200,8192,16384,32760]:
    for stage,rank in [(0,0),(1,0),(3,0),(3,1)]:
        for limit in [1,3]:compare(dict(n=n,stage=stage,rank=rank,limit=limit),'actual_W2_normal')
for stage,rank in [(0,0),(3,0),(3,1)]:
    for at in [0,1,2,3]:compare(dict(n=1200,stage=stage,rank=rank,cancel_at=at),'actual_W2_cancel_or_timeout')
    compare(dict(n=1200,stage=stage,rank=rank,timeout_at=1),'actual_W2_cancel_or_timeout')
    for at in [1,2,3]:compare(dict(n=1200,stage=stage,rank=rank,fail_at=at),'actual_W2_forward_exception')
# Reuse the prior W1 reference method: drop only its absent report-flag check.
raw=(B/'check_glm_window2_r236.py').read_text();start=raw.index('def case(');end=raw.index('\nout=[]')
case_source=raw[start:end].replace("assert error is None and env['report']['normal_reset']is True","assert error is None")
exec(compile(case_source,'<same-prior-W1-reference-case>','exec'),old.__dict__)
source=W1.read_text();start=source.index('            try:\n                for step in range(limit):');end=source.index("            row={'sequence':seq",start)
old.compiled=compile(textwrap.dedent(source[start:end]),'<actual-original-W1>','exec')
for n in [1,511,512,513,1024,1025,1200,8192,16384,32760]:
    for stage,rank in [(0,0),(1,0),(3,0),(3,1)]:
        for limit in [1,3]:
            kwargs=dict(n=n,stage=stage,rank=rank,limit=limit)
            a,x=old.case(**kwargs);b,y=new.case(**kwargs,window=1)
            assert a==b and x['generated']==y['generated'] and x['reason']==y['reason']
            assert x['calls']==y['calls'] and x['past']==y['past']
            counts['actual_W1_normal_vs_original_W1']+=1
paths=[D/'glm_agent_service_window12_r240.py',D/'glm_prefill_execute_r236.py',D/'glm_prefill_paired_client_r240.py',D/'check_glm_paired_caller_r240.py',B/'check_glm_window2_r236.py',W1]
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU source-extracted actual caller traces; no Torch/Native/CCL/model execution','counts':counts,'total':sum(counts.values()),'all_compared_events_calls_outputs_past_equal':True,'source_SHA256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'W1_exception_equivalence_claimed':False,'original_production_W1_modified':False,'prefill_input_case_sizes':[1,511,512,513,1024,1025,1200,8192,16384,32760],'new_GPU':0,'new_remote_jobs':0,'performance_rounds':0,'measured_gain':None}
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'counts':counts,'total':result['total'],'Model_qualification':False}))
