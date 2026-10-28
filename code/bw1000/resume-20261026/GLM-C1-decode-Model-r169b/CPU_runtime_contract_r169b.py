import pathlib,json,ast,sys,importlib.util,copy
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D));import arrival_runner_r9258 as r
from case_plan_r169b import load
from profile_consumer_r169 import consume,overhead
checks=[];cases=json.loads((D/'corpus-reference.json').read_bytes())['requests'][:4]
for on in [False,True,False]:
 now=[0.];ledger=[0]*4;calls=[]
 def forward(rows,tokens,pasts,kind,prefill):
  assert len(rows)==1 and pasts==[ledger[i]for i in rows]
  for i,p,t in zip(rows,pasts,tokens):ledger[i]=p+len(t)
  now[0]+=.1;calls.append((kind,rows,pasts,list(map(len,tokens))))
  return [999 if ledger[i]>=len(cases[i]['input_ids'])+24 else 17+i for i in rows],.1,now[0]-.1,now[0]
 fixture=[dict(x,max_new_tokens=32)for x in cases]
 result,trace=r.run_episode(fixture,[0.]*4,False,{999},True,lambda cmd,seq:r.decode_command(r.encode_command(cmd),seq),lambda cmd,out:out,forward,clock=lambda:now[0],wait=lambda sec:now.__setitem__(0,now[0]+sec),budget_seconds=120,deferred_feedback=True,max_live=1)
 assert result['all_requests_completed']and result['peak_admitted_live']==1 and not result['row_recycling']and all(x['stop_reason']=='EOS'for x in result['requests'])and all(len(rows)==1 for _,rows,_,_ in calls)
 checks.append({'observer':on,'fake_scheduler_only':True,'naturalEOS_CPU':4,'single_admission':True})
names=['CAPI.owner.begin','PP.incoming.begin','PP.incoming.end','layer.0.full.begin','detail.layer.0.attention.begin','detail.layer.0.attention.end','layer.0.end','PP.outgoing.begin','PP.outgoing.end','CAPI.owner.end.before_original_drain']
events=[{'prefill':False,'native_seconds':.01}for i in range(20)];calls=[{'event_index':i,'op':'DECODE','rows':[0],'pasts':[100+i],'counts':[1]}for i in range(20)]
sample=lambda j:{'selector':'decode_T1','tokens':1,'sequence':j,'eligible_decode1_index':j,'row_past_count':[[0,99+j,1]],'marks':[{'name':n,'owner_host_s':10+i*.001,'gpu_from_first_s':i*.001}for i,n in enumerate(names)]}
obs={'selector_revision':'r169','required':[0,0,0,0,3],'coverage':[0,0,0,0,3],'samples':[sample(j)for j in [17,18,19]]}
assert consume(obs,calls,events)['coverage']=='COMPLETE'
bad=copy.deepcopy(obs);bad['samples'][0]['row_past_count'][0][1]=0
try:consume(bad,calls,events)
except ValueError:pass
else:raise AssertionError('prefix accepted')
bad=copy.deepcopy(obs);bad['coverage']=[0]*5;assert consume(bad,calls,events)['coverage']=='UNKNOWN'
badcalls=copy.deepcopy(calls);badcalls[16]['op']='ANCHOR'
try:consume(obs,badcalls,events)
except ValueError:pass
else:raise AssertionError('anchor accepted')
arms=[{'label':str(i),'native_calls':calls,'observer_enabled':x}for i,x in enumerate([False,True,False])];assert overhead(arms,events)['bounded_overhead_screen_passed']
assert len(load()['episodes'])==3 and [x['observer']for x in load()['episodes']]==[False,True,False]
for p in D.glob('*.py'):ast.parse(p.read_text())
out={'passed':True,'CPU_only':True,'GPU':False,'model':False,'checks':checks,'consumer_actual_source_contract':True,'anchor_prefix_rejected':True,'missing_coverage_UNKNOWN':True,'one_load_4_unique_QA_3_arms':True};print(json.dumps(out))
