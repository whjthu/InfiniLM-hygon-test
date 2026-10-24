from pathlib import Path
import importlib.util,json,hashlib
r=Path.cwd();s=importlib.util.spec_from_file_location('actual',r/'kimi_interleaved_prefill_decode_r8669zq.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=[]
for scenario in ['full128','earlyEOS','wrongPastInjected']:
 gold=[[1000+i*256+j for j in range(128)]for i in range(16)];eos={999999}
 if scenario=='earlyEOS':gold[0]=[gold[0][0],999999];gold[1]=[999999]
 positions=[0]*16;clock=[0.];calls=[]
 def forward(ins,pasts,rows,label,token_only):
  assert rows==sorted(set(rows))
  result=[]
  for x,p,row in zip(ins,pasts,rows):
   i=row-1;n=positions[i];assert p==(0 if n==0 else 512+n-1)
   if n==0:assert len(x)==512 and not token_only
   else:assert x==[gold[i][n-1]]and token_only
   if scenario=='wrongPastInjected'and len(calls)==3:raise ValueError('injected incorrect producer/cache position')
   result.append(gold[i][n]);positions[i]+=1
  clock[0]+=.02 if not token_only else .001
  calls.append(rows);return result,.02 if not token_only else .001,None
 state={}
 try:
  q=m.run_interleaved(inputs=[[i]*512 for i in range(16)],eos=eos,forward=forward,clock=lambda:clock[0],state=state,label='CPU-contract',limit=128)
  assert scenario!='wrongPastInjected';assert q['generated_ids']==gold and q['completed'];assert q['forward_jobs'][0]['rows']==[1,2]and q['forward_jobs'][1]['kind']=='decode';results.append({'scenario':scenario,'passed':True,'full_IDs_EOS_match':True,'maximum_output_length':max(map(len,q['generated_ids'])),'first_decode_N':q['forward_jobs'][1]['N'],'forward_count':len(q['forward_jobs']),'row_position_checks_each_call':True})
 except ValueError:
  assert scenario=='wrongPastInjected'and not state['completed'];results.append({'scenario':scenario,'passed':True,'failure_propagated':True,'partial_actual_outputs_retained':bool(state['generated_ids'][0])})
p=r/'Kimi-interleaved-CPU-contract-r8669zq.json';p.write_text(json.dumps({'CPU_only':True,'GPU':False,'actual_consumer_SHA256':hashlib.sha256((r/'kimi_interleaved_prefill_decode_r8669zq.py').read_bytes()).hexdigest(),'tests':results,'performance_claim':False},indent=2)+'\n',newline='\n');print(json.dumps(results))
