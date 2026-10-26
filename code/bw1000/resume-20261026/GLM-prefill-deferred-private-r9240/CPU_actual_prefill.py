import ast,pathlib,json,hashlib,threading,queue,weakref,sys,copy
p=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(p));from decode_wave_r9225 import decode_wave_r9225
from schedule_r9132b import prefix_chunks,microbatches
src=p/'glm_ragged_probe_r9240.py';tree=ast.parse(src.read_text());outer=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='run');prefix=next(v for v in ast.walk(outer)if isinstance(v,ast.FunctionDef)and v.name=='prefix');cb=next(v for v in ast.walk(outer)if isinstance(v,ast.FunctionDef)and v.name=='prefill_group_r9240');loop=next(v for v in ast.walk(outer)if isinstance(v,ast.For)and 'prefill_group_r9240' in ast.unparse(v.iter))
f=ast.parse("def perform(C,prefill_deferred_r9240,forward,emit,group):\n npf=0.;prefill_tokens=0;pasts=[0]*C;label='CPU';on=1\n").body[0];f.body.extend([prefix,cb,loop,ast.parse('return npf,prefill_tokens,pasts').body[0]]);env={'decode_wave_r9225':decode_wave_r9225,'prefix_chunks':prefix_chunks,'microbatches':microbatches};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(src),'exec'),env);perform=env['perform']
class Logits:
 def __init__(self,rows,values):self.rows=rows[:];self.values=values[:]
allresults=[]
for C,lengths in [(32,[80+(i*17)%50 for i in range(32)]),(32,[512+(i*11)%128 for i in range(32)]),(16,[1,2,638,100]*4),(8,[1]*8)]:
 group=[{'input_ids':[10+i*700+j for j in range(n)]}for i,n in enumerate(lengths)];original=copy.deepcopy(group);arms=[]
 for mode in [False,True]:
  barrier=threading.Barrier(16,timeout=10);tp=[threading.Barrier(8,timeout=10)for _ in range(2)];mail=[queue.Queue()for _ in range(8)];ack=[queue.Queue()for _ in range(8)];results=[None]*16;errors=[]
  def worker(pid):
   try:
    stage,rank=divmod(pid,8);owner=threading.get_ident();ledger=[0]*C;outputs={};log=[];live=weakref.WeakSet();peak=[0];calls=[0]
    def forward(rows,tokens,pasts,label,on,prefill):
     assert threading.get_ident()==owner and prefill is True and on==1 and 0<len(rows)<=8 and 0<sum(map(len,tokens))<=4096 and len(rows)==len(set(rows))
     for i,vals,past in zip(rows,tokens,pasts):assert past==ledger[i]and vals==group[i]['input_ids'][past:past+len(vals)]and 0<past+len(vals)<=640
     key=(calls[0],tuple(rows),tuple(pasts),tuple(map(len,tokens)));calls[0]+=1;tp[stage].wait()
     if stage==0:mail[rank].put(key);assert ack[rank].get(timeout=10)==key
     else:assert mail[rank].get(timeout=10)==key;ack[rank].put(key)
     for i,vals in zip(rows,tokens):ledger[i]+=len(vals)
     kind='anchor'if label.endswith('.PFanchor')else'prefix';log.append((kind,tuple(rows),tuple(pasts),tuple(map(len,tokens))));a=Logits(rows,[v[-1]+1 for v in tokens]);live.add(a);peak[0]=max(peak[0],len(live));return a,.001,calls[0],calls[0]+.001
    def emit(rows,out,prefill):
     assert threading.get_ident()==owner and prefill is True and out.rows==rows
     assert all(ledger[i]==lengths[i]and i not in outputs for i in rows),'early or repeated firsttoken'
     barrier.wait();log.append(('emit',tuple(rows)))
     for i,val in zip(rows,out.values):outputs[i]=val
     barrier.wait()
    wall,n,pasts=perform(C,mode,forward,emit,group)
    assert ledger==lengths and pasts==lengths and n==sum(lengths)and len(outputs)==C and all(outputs[i]==group[i]['input_ids'][-1]+1 for i in range(C));assert not list(live)and peak[0]<=(2 if mode else 1);assert original==group
    results[pid]={'trace':log,'tokens':n,'native_calls':calls[0],'peak_logits':peak[0],'final_pasts':pasts,'outputs':outputs}
   except BaseException as e:errors.append((pid,repr(e)));barrier.abort();tp[0].abort();tp[1].abort()
  threads=[threading.Thread(target=worker,args=(i,))for i in range(16)]
  for t in threads:t.start()
  for t in threads:t.join(15)
  assert not any(t.is_alive()for t in threads)and not errors,errors
  assert all(v==results[0]for v in results)
  arms.append(results[0]);allresults.append({'C':C,'lengths':lengths,'mode':mode,'actual_owner_threads':16,'tokens':results[0]['tokens'],'native_calls_per_owner':results[0]['native_calls'],'max_live_logits':results[0]['peak_logits'],'past_readonly_output_mapping_pass':True,'first_operations':results[0]['trace'][:7]})
 assert arms[0]['native_calls']==arms[1]['native_calls']and arms[0]['outputs']==arms[1]['outputs']and arms[0]['final_pasts']==arms[1]['final_pasts']
 # Native payload multiset is identical; only calls/feedback ordering differs.
 assert sorted(x for x in arms[0]['trace']if x[0]!='emit')==sorted(x for x in arms[1]['trace']if x[0]!='emit')
 if C==8:assert arms[0]['trace']==arms[1]['trace']
for bad in [0,1,'1',None]:
 calls=[]
 try:perform(8,bad,lambda *a:calls.append(a),None,[{'input_ids':[1]}]*8);raise AssertionError('nonbool')
 except TypeError:pass
 assert not calls
q={'passed':True,'scope':'CPU_ONLY_ACTUAL_PREFIX_AND_NEW_CALLSITE_AST_WITH_16_REAL_OWNER_THREADS_P2P_WORLD_BARRIERS_NOT_GPU','source_SHA256':hashlib.sha256(src.read_bytes()).hexdigest(),'case_count':len(allresults),'cases':allresults,'unchanged_payload_multiset':True,'same_native_CAPI_call_count':True,'partialprefix_logits_never_emitted':True,'firsttoken_order_and_wholeinput_past_lengths_preserved':True,'max_two_live_logits':True,'input_readonly':True,'one_micro_control_identical_order':True,'no_GPU_Model_quality_or_speed_claim':True};(p/'CPU-actual-prefill-loop.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'passed':True,'cases':len(allresults),'owners_each':16,'source_SHA256':q['source_SHA256']}))
