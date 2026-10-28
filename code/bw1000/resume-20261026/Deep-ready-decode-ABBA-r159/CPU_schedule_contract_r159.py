"""Execute actual scheduling AST with a fake CPU owner; no model imports."""
import pathlib,ast,json,importlib.util,sys
D=pathlib.Path(__file__).resolve().parent;tree=ast.parse((D/'Deep_ready_decode_r159.py').read_bytes());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['ready_rows_before_next_prefill','per_row_decode_past']];helpers={};exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual-r159-helpers>','exec'),helpers)
generate=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='generate');prefill=next(n for n in generate.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='start');regular=next(n for n in generate.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='step');prefill_code=compile(ast.Module(body=[prefill],type_ignores=[]),'<actual-r159-prefill-schedule>','exec');regular_code=compile(ast.Module(body=[regular],type_ignores=[]),'<actual-r159-regular-schedule>','exec');cases=[]
def scenario(label,candidate,caps,eos_at):
 B=32;prompts=[[10]*int(400+i)for i in range(B)];generated=[[]for _ in range(B)];done=[False]*B;native_lengths=[0]*B;initialized=[False]*B;windows=[];calls=[]
 def forward(rows,seq,past,is_prefill):
  assert rows and len(rows)==len(set(rows))and all(0<=i<B for i in rows)
  if is_prefill:
   assert len(rows)<=4 and all(not initialized[i]for i in rows)
   assert all(p==0 for p in past)
  else:
   assert all(initialized[i]and not done[i]for i in rows),'Unready/EOS/capped row submitted'
   for i,p,x in zip(rows,past,seq):assert p==native_lengths[i]and x==[generated[i][-1]],'Per-row cache past differs'
  for i,x in zip(rows,seq):
   native_lengths[i]+=len(x);initialized[i]=True;generated[i].append(1)
   if len(generated[i])>=caps[i]or(eos_at[i]is not None and len(generated[i])>=eos_at[i]):done[i]=True
  calls.append({'prefill':is_prefill,'rows':rows.copy(),'past':past.copy()});windows.append({'prefill':is_prefill});return True
 ns=dict(helpers,B=B,prompts=prompts,generated=generated,done=done,forward=forward,windows=windows,ready_decode=candidate,report={},label=label,save=lambda:None,caps=caps,EOS=1,compact=True,budget=False)
 exec(prefill_code,ns);early=[x for x in windows if x.get('early_ready_decode')];assert len(early)<=7 and sum(x['prefill']for x in calls)==8
 if label=='candidate_no_early_stop':assert len(early)==7 and [len(x)for x in generated]==sum(([8-j]*4 for j in range(8)),[])
 if not candidate:assert all(len(x)==1 for x in generated)
 exec(regular_code,ns);assert all(done),'Completion lost by varying row past'
 assert all(native_lengths[i]==len(prompts[i])+len(generated[i])-1 for i in range(B))
 cases.append({'name':label,'candidate':candidate,'prefill_calls':8,'early_decode_calls':len(early),'no_unready_or_stopped_row':True,'all_per_row_native_lengths_match':True,'completed_fake_rows':sum(done),'passed':True})
scenario('baseline',False,[12]*32,[None]*32)
scenario('candidate_no_early_stop',True,[12]*32,[None]*32)
scenario('candidate_varied_caps_and_EOS',True,[3+i%5 for i in range(32)],[1 if i%7==0 else 2 if i%9==0 else None for i in range(32)])
scenario('candidate_all_EOS_from_prefill',True,[12]*32,[1]*32)
for step in range(1,12):
 generated=[[1]*step for _ in range(32)];lengths=list(range(400,432));assert helpers['per_row_decode_past'](lengths,generated,list(range(32)))==[p+step-1 for p in lengths]
# Execute actual clock AST, excluding early tokens from post-prefill numerator.
targets={'pf','dw','anchor','common','post_useful','early_calls'};clock_nodes=[n for n in generate.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in targets for t in n.targets)];assert len(clock_nodes)==6
windows=[]
for block in range(8):
 windows.append({'prefill':True,'return_monotonic':10.*(block+1),'active_useful_tokens':4})
 if block<7:windows.append({'prefill':False,'return_monotonic':10.*(block+1)+1,'active_useful_tokens':4*(block+1),'early_ready_decode':True})
windows.append({'prefill':False,'return_monotonic':81.,'active_useful_tokens':32});ns={'windows':windows,'ended':81.};exec(compile(ast.Module(body=clock_nodes,type_ignores=[]),'<actual-r159-clock>','exec'),ns);assert ns['common']==1. and ns['post_useful']==32 and ns['early_calls']==7;assert sum(w['active_useful_tokens']for w in windows if not w['prefill'])==144
spec=importlib.util.spec_from_file_location('bootstrap159CPU',D/'bootstrap_r159.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);old=pathlib.Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/Deep-C4-independent64-private-r9278/modern_Deep_independent64_r9278.py');mod.transform(old.read_bytes())
ids=[x['id']for x in json.loads((D/'dataset64.json').read_bytes())['cases']];assert len(ids)==len(set(ids))==64;assert 'torch'not in sys.modules and 'numpy'not in sys.modules
q=json.loads((D/'CPU-local-contract-r159.json').read_bytes());q.update(local_static_and_actual_schedule_AST_execution_passed=True,CPU_fake_owner_only=True,actual_GPU_executed=False,Native_dlopen=False,Torch_imported=False,scheduling_cases=cases,baseline_past_formula_equivalent=True,clock_contract={'total_early_plus_late_decode_tokens':144,'post_admission_tokens':32,'post_admission_seconds':1,'excluded_early_tokens':112,'passed':True},ABBA_cases_per_arm=128,ABBA_independent_question_ids=64);(D/'CPU-local-contract-r159.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'passed':True,'cases':len(cases),'actual_operator_or_model':False,'clock_correct':True}))
