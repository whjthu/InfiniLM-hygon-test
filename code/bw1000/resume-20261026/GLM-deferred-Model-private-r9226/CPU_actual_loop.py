import ast,pathlib,threading,time,queue,weakref,collections,json,hashlib,sys
p=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(p));from decode_wave_r9225 import decode_wave_r9225
sys.path.insert(0,str(p.parent/'GLM-B32-Model-private-r9183'));from schedule_r9132b import microbatches
source=p/'glm_ragged_probe_r9226.py';tree=ast.parse(source.read_text());run=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='run');loop=next(v for v in ast.walk(run)if isinstance(v,ast.While)and isinstance(v.test,ast.UnaryOp)and ast.unparse(v.test)=='not all(done)');code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(source),'exec')
class Logits:
 def __init__(self,rows,values):self.rows=rows;self.values=values
results=[]
for C,M in [(32,16),(16,8),(32,32),(7,16)]:
 for enabled in [False,True]:
  world=threading.Barrier(16,timeout=10);tp=[threading.Barrier(8,timeout=10)for _ in range(2)];mail=[queue.Queue()for _ in range(8)];acks=[queue.Queue()for _ in range(8)];out=[None]*16;errors=[];threads=[]
  def work(participant):
   try:
    stage,rank=divmod(participant,8);owner=threading.get_ident();pasts=[23+(i%4)for i in range(C)];ledger=pasts[:];generated=[[100+i]for i in range(C)];done=[False]*C;trace=[];live=weakref.WeakSet();peak=[0];serial=[0];retired=set();report={};savecalls=[0];rowhist=collections.Counter()
    def forward(rows,tokens,pp,label,on,prefill):
     assert threading.get_ident()==owner and len(rows)==len(set(rows))and all(i not in retired for i in rows)
     assert pp==[ledger[i]for i in rows]and tokens==[[generated[i][-1]]for i in rows]
     key=(serial[0],tuple(rows),tuple(pp));serial[0]+=1;trace.append(('F',tuple(rows),tuple(pp)));tp[stage].wait()
     if stage==0:mail[rank].put(key);assert acks[rank].get(timeout=10)==key
     else:assert mail[rank].get(timeout=10)==key;acks[rank].put(key);time.sleep(.0001)
     values=[0 if len(generated[i])>=2+i%5 else generated[i][-1]+1 for i in rows]
     for i in rows:ledger[i]+=1
     logits=Logits(rows[:],values);live.add(logits);peak[0]=max(peak[0],len(live));t=time.perf_counter();return logits,.001,t,t+.001
    def emit(rows,logits,prefill):
     assert threading.get_ident()==owner and logits.rows==rows;trace.append(('E',tuple(rows)));world.wait()
     for i,token in zip(rows,logits.values):
      generated[i].append(token)
      if token==0:done[i]=True;retired.add(i)
     world.wait();return sum(v!=0 for v in logits.values)
    def budget(n):return True
    def save():savecalls[0]+=1
    env=dict(done=done,C=C,M=M,generated=generated,pasts=pasts,ndec=0.,first_decode=None,last_decode=None,rows_hist=rowhist,useful_decode=0,report=report,step=1,case_budget_stop=False,stopped=False,on=1,label='CPU',forward=forward,emit=emit,budget=budget,save=save,time=time,microbatches=microbatches,decode_wave_r9225=decode_wave_r9225,deferred_feedback_r9225=enabled)
    exec(code,env)
    assert all(done)and ledger==pasts and len(retired)==C
    assert peak[0]<=(2 if enabled else 1),(enabled,peak)
    # Exact frontend loop lifetime releases final output when helper is exhausted.
    assert not list(live)
    out[participant]={'trace':trace,'generated':generated,'pasts':pasts,'peak_live_logits':peak[0],'forward_calls':serial[0],'retired':sorted(retired),'steps':env['step']-1}
   except BaseException as e:errors.append((participant,repr(e)));world.abort();tp[0].abort();tp[1].abort()
  for participant in range(16):threads.append(threading.Thread(target=work,args=(participant,)))
  for t in threads:t.start()
  for t in threads:t.join(15)
  assert not any(t.is_alive()for t in threads)and not errors,errors
  assert all(q['trace']==out[0]['trace']and q['generated']==out[0]['generated']and q['pasts']==out[0]['pasts']for q in out)
  trace=out[0]['trace'];peak=out[0]['peak_live_logits'];result={'C':C,'M':M,'enabled':enabled,'owner_threads':16,'allrank_order_agreed':True,'past_ledger_agreed':True,'EOS_retired_all':True,'max_live_logits':peak,'forward_calls_per_owner':out[0]['forward_calls'],'steps':out[0]['steps'],'first_actions':trace[:4],'output_SHA256':hashlib.sha256(json.dumps(out[0]['generated']).encode()).hexdigest()};results.append(result)
for a,b in zip(results[::2],results[1::2]):assert a['output_SHA256']==b['output_SHA256']and a['forward_calls_per_owner']==b['forward_calls_per_owner']
# Wholebatch fail before callback; no externally visible forward on malformed groups.
for groups,flag in [([[0],[0]],True),([[32]],True),([[]],True),([[0]],1)]:
 calls=[]
 try:list(decode_wave_r9225(groups,lambda rows:calls.append(rows),flag));raise AssertionError('accepted invalid')
 except(ValueError,TypeError):pass
 assert calls==[]
q={'passed':True,'qualification':'CPU_ONLY_ACTUAL_FRONTEND_AST_LOOP_WITH_16_REAL_OWNER_THREADS_P2P_AND_WORLD_BARRIERS_NOT_GPU_CCL','source_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'test_source_SHA256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'case_count':len(results),'cases':results,'no_Runtime_or_GPU_initialized':True,'no_native_math_test_or_performance_claim':True,'malformed_wholebatch_rejected_before_callback':4,'finite_token_identity':'synthetic deterministic mock only; not quality criterion'}
(p/'CPU-actual-loop.json').write_text(json.dumps(q,indent=2),encoding='utf8');print(json.dumps({'passed':True,'cases':len(results),'owners_each':16,'source_SHA256':q['source_SHA256']}))
