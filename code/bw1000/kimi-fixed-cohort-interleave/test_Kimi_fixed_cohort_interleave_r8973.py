import sys,pathlib,threading,json,time,hashlib
from kimi_fixed_cohort_interleave_r8973 import run,jobs
from kimi_cohort_feedback_idle_r4915 import FeedbackHub,FeedbackClient
r=pathlib.Path.cwd();B=64;inputs=[[i%2+1]*512 for i in range(B)];hub=FeedbackHub('127.0.0.1',0,'cpu-fixed-cohort-r8973',4,3,timeout=8,idle_timeout=20);ready=threading.Barrier(4);end=threading.Barrier(4);outputs={};traces={};errors=[]
def token(row,past):return 1000+row*701+past
# Independent serial fixed-cohort oracle: same first/last inputs and row-local past.
expected=[[token(i,0)]+[token(i,p)for p in range(512,639)]for i in range(B)]
def worker(rank):
 cli=None
 try:
  cli=FeedbackClient(*hub.address,'cpu-fixed-cohort-r8973',rank,timeout=10);ready.wait(10)
  if rank==3:cli.request('begin','test',cohorts=[list(range(a,a+16))for a in range(0,64,16)])
  ready.wait(10);trace=[];state=[0]*64
  def f(kind,rows,ins,pasts,index):
   assert list(jobs()[index][1])==rows and jobs()[index][0]==kind
   assert pasts==[state[i]for i in rows]
   if kind=='prefill':assert len(rows)==2 and all(p==0 for p in pasts)and all(len(x)==512 for x in ins)
   else:
    assert len(rows)==16 and rows==list(range(rows[0],rows[0]+16))and rows[0]%16==0
    assert all(len(x)==1 and x[0]==(token(i,0)if p==512 else token(i,p-1))for i,p,x in zip(rows,pasts,ins))
   trace.append((kind,rows[:],pasts[:]))
   for i,x in zip(rows,ins):state[i]+=len(x)
   return ([token(i,p)for i,p in zip(rows,pasts)]if rank==3 else None),0.
  outputs[rank]=run(inputs=inputs,feedback=cli,case='test',is_emitter=rank==3,forward=f,eos=set());traces[rank]=trace;end.wait(10)
  if rank==3:outputs['feedback']=cli.request('end','test')
  end.wait(10)
 except BaseException as e:errors.append((rank,repr(e)));hub.fail_case(str(e))
 finally:
  if cli:cli.close()
threads=[threading.Thread(target=worker,args=(i,))for i in range(4)];tick=time.perf_counter()
for t in threads:t.start()
for t in threads:t.join(20)
assert not any(t.is_alive()for t in threads), 'CPU_deadlock';assert not errors,errors
assert all(outputs[i]['generated_ids']==expected for i in range(4));assert all(traces[i]==traces[0]for i in range(4));assert outputs['feedback']['published_per_cohort']==[128]*4
hub.close();p=r/'Kimi-fixed-cohort-interleave-CPU-contract-r8973.json';q={'CPU_only':True,'GPU':False,'passed':True,'real_TCP_owners':4,'generated_IDs':64*128,'all128_independent_row_oracle':True,'same_owner_jobseq_rows_pasts_inputs':True,'prefill_calls':32,'decode_M16_calls':508,'no_generic_smallM_decode':True,'final_pasts':[639]*64,'bounded_original_feedback':outputs['feedback'],'seconds':time.perf_counter()-tick,'source_SHA256':hashlib.sha256((r/'kimi_fixed_cohort_interleave_r8973.py').read_bytes()).hexdigest(),'scope':'One meaningful actualTCP replay of proposed scheduler, not GPU timing/numeric qualification; no Model/SDK/Engine init.'};p.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))

