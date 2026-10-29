import inspect,contextlib,datetime,hashlib,importlib.util,io,json,pathlib,sys,textwrap
R=pathlib.Path(sys.argv[1]).resolve();output=pathlib.Path(sys.argv[2]).resolve();D=pathlib.Path(__file__).parent;B=R/'code/bw1000/agent-context-32k/GLM-prefill-window2-source-r229';sys.argv=[str(__file__),str(R)]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s)
 raw=p.read_text();cut=raw.index('\nout=[]');exec(compile(raw[:cut],str(p),'exec'),m.__dict__)
 return m
old=load(B/'check_glm_window2_r229.py','old_check');new=load(D/'check_glm_window2_r236.py','new_check');counts={'W2_normal':0,'W2_cancel_or_timeout':0,'W2_forward_exception':0,'W1_normal':0};scalar=[]
def compare(kwargs,category,fatal=False):
 a,x=old.case(**kwargs);b,y=new.case(**kwargs);assert a==b,(category,kwargs,a,b);assert x['generated']==y['generated']and x['reason']==y['reason'];assert x['calls']==y['calls']
 if fatal:
  if x['past']!=y['past']:scalar.append({'case':kwargs,'old_local_past':x['past'],'new_outer_local_past':y['past'],'scope':'Fatal exception propagates; both skip normal reset/barrier and do not resume. No Native state or record is rolled back.'})
 else:assert x['past']==y['past']
 counts[category]+=1
for n in [1,511,512,513,1024,1025,1200,8192,16384,32760]:
 for stage,rank in [(0,0),(1,0),(3,0),(3,1)]:
  for limit in [1,3]:compare({'n':n,'stage':stage,'rank':rank,'limit':limit},'W2_normal')
for stage,rank in [(0,0),(3,0),(3,1)]:
 for at in [0,1,2,3]:compare({'n':1200,'stage':stage,'rank':rank,'cancel_at':at},'W2_cancel_or_timeout')
 compare({'n':1200,'stage':stage,'rank':rank,'timeout_at':1},'W2_cancel_or_timeout')
 for at in [1,2,3]:compare({'n':1200,'stage':stage,'rank':rank,'fail_at':at},'W2_forward_exception',True)
case_source=inspect.getsource(old.case).replace("assert error is None and env['report']['normal_reset']is True","assert error is None");exec(compile(case_source,'<W1-reference-checker-report-field-only>','exec'),old.__dict__)
baseline=(B/'glm_agent_service_baseline_r204.py').read_text();start=baseline.index('            try:\n                for step in range(limit):');end=baseline.index("            row={'sequence':seq",start);old.compiled=compile(textwrap.dedent(baseline[start:end]),'<actual-W1-baseline>','exec')
source=(D/'glm_agent_service_window2_r236.py').read_text();start=source.index('            request_completed_normally=False');end=source.index("            row={'sequence':seq",start);block=textwrap.dedent(source[start:end]).replace('window=2 if step==0 else 1','window=1');assert block!=textwrap.dedent(source[start:end]);new.compiled=compile(block,'<helper-W1-normal-comparison>','exec')
for n in [1,511,512,513,1024,1025,1200,8192,16384,32760]:
 for stage,rank in [(0,0),(1,0),(3,0),(3,1)]:
  for limit in [1,3]:compare({'n':n,'stage':stage,'rank':rank,'limit':limit},'W1_normal')
r={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Local source-derived fake forward/control trace comparison only; no Torch/Native/CCL/Model execution','counts':counts,'all_trace_events_equal':True,'all_call_records_and_generated_choices_equal':True,'fatal_outer_local_past_differences':scalar,'default_W1_baseline_file_modified':False,'W1_exception_equivalence_claimed':False,'W1_reference_checker_adjustment':'Old W1 has no normal_reset report flag; only that checker assertion was removed. Exact reset/barrier events, errors, all call records, output choices and normal past remain compared.','new_GPU':0,'performance_rounds':0,'reference_import_side_effects':'Only source definitions before built-in case loop are loaded; no reference result file writes.','source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [B/'glm_agent_service_baseline_r204.py',B/'glm_agent_service_window2_r229.py',D/'glm_agent_service_window2_r236.py',D/'glm_prefill_execute_r236.py',D/'check_glm_window2_r236.py',D/'compare_source_traces_r236.py']}}
p=output;assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'counts':counts,'all_trace_events_equal':True,'fatal_outer_local_past_difference_cases':len(scalar),'new_GPU':0}))