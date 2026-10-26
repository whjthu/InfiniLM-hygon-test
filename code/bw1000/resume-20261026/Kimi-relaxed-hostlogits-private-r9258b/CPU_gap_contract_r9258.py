import pathlib,sys,json,itertools,math,threading,types,ctypes as C,time
import numpy as np
rt=pathlib.Path(__file__).with_name('runtime');sys.path.insert(0,str(rt))
from host_gap_r9258 import scalar_gaps,special_provenance
from relaxed_gap_policy_r9257 import decide
from batch_policy_r9128 import greedy,ReplayStateModel
from batch_native_r9128 import Native
checks=0
def bits(x):return(np.asarray(x,np.float32).view(np.uint32)>>16).astype(np.uint16)
# Ragged perrequest offsets: candidate j is input[offset+j+1], never next request's anchor.
rows=[[20,1,2,3],[22,7],[23]]
T=sum(map(len,rows));a=np.full((T,32),bits(-5),np.uint16);sel=[4]*T
a[:,4]=bits(1.)
a[0,1]=bits(.953125);a[1,2]=bits(.875);a[2,3]=bits(.9921875);a[4,7]=bits(1.)
p=scalar_gaps(a,sel,rows)
assert p['gaps']==[.046875,.125,.0078125,None,0.,None,None]and p['gathered_BF16_scalars']==8
checks+=1
# BF16 ties use actual selected top1 (including tie-breaking choice), not a CPU recomputed argmax.
a[0,1]=bits(1.);q=scalar_gaps(a,sel,rows);assert q['gaps'][0]==0
a[0,1]=bits(1.0625);q=scalar_gaps(a,sel,rows);assert q['gaps'][0]is None and q['negative_gaps']==1
for x in [np.inf,np.nan,-np.inf]:
 a[0,1]=bits(x);q=scalar_gaps(a,sel,rows);assert q['gaps'][0]is None and q['nonfinite_pairs']==1
checks+=4
# Exhaustive policy truncation/terminal guard and OFF=original decisions.
for K in range(3):
 for props in itertools.product(range(4),repeat=K):
  for top in itertools.product(range(4),repeat=K+1):
   for gs in itertools.product([0.,.09375,.125,None],repeat=K):
    for remaining in range(1,K+2):
     base=greedy(7,props,top,{3},remaining)
     off,m=decide(7,props,top,None,0,{3},{2},remaining);assert off==base
     z,m=decide(7,props,top,gs,.10,{3},{2},remaining)
     assert z.commit_inputs==(7,)+z.emitted[:-1]
     model=ReplayStateModel([8]);model.verify((7,)+props);model.commit(z)
     assert model.shadow is None
     if not z.terminal:assert model.active==(8,)+z.commit_inputs
     assert m['relaxed']<=m['verified_relaxed'] and m['relaxed']<=len(z.emitted)
     for i in m.get('relaxed_positions',[]):
      assert i<len(z.emitted)and props[i]not in{2,3}and top[i]not in{2,3}and 0<=gs[i]<=.10
     checks+=1
# Four genuine threads share real Native.finish control packet before callback; only emitter owns host scores.
class Hub:
 def __init__(self):self.cv=threading.Condition();self.p={}
 def collect(self,node,key,value):
  with self.cv:
   self.p[node]=value;self.cv.notify_all();assert self.cv.wait_for(lambda:len(self.p)==4,10)
   return[self.p[i]for i in range(4)]
hub=Hub();results={};errors=[]
a[:]=bits(-5);a[:,4]=bits(1);a[0,1]=bits(.953125);a[1,2]=bits(.875);a[2,3]=bits(.9921875);a[4,7]=bits(1)
def rank(node):
 try:
  n=Native.__new__(Native);n.tail=node>=2;n.node=node;n.emitter=2;n.vocab=32;n.control=types.SimpleNamespace(collect=lambda key,val:hub.collect(node,key,val));n.serial=0;n.calls=[];n.relaxed_gap_epsilon=.10
  got=[]
  def callback(selected,p):
   assert p is not None;off=0;ds=[]
   for row in rows:
    z,m=decide(row[0],row[1:],selected[off:off+len(row)],p['gaps'][off:off+len(row)-1],.10,{31},{30},8)
    ds.append((z,m));off+=len(row)
   got.append(ds)
  n.finish('verify',[1,3,7],[10,12,14],list(map(len,rows)),False,a.copy(),np.asarray(sel,np.int64),np.asarray([int(node>=2)+2*int(node==2)]),.01,callback,rows)
  results[node]=got
 except BaseException as e:errors.append(repr(e))
ts=[threading.Thread(target=rank,args=(i,))for i in range(4)]
for t in ts:t.start()
for t in ts:t.join(15)
assert not errors and len(results)==4 and all(results[i]==results[0]for i in range(4)),errors
# Exact native commit ABI retains original accepted value even terminal; continuation false performs zero commit.
captured=[]
def nativecommit(ctx,count,ids,accepted,continuation):
 av=np.ctypeslib.as_array(C.cast(accepted,C.POINTER(C.c_size_t)),shape=(count,)).tolist()
 cv=np.ctypeslib.as_array(C.cast(continuation,C.POINTER(C.c_int)),shape=(count,)).tolist()
 captured.append((av,cv));return 0
n=Native.__new__(Native);n.ctx=1;n.lib=types.SimpleNamespace(modern_kimi_batch_commit_r9122=nativecommit);n.ck=lambda code:None
terminal,m=decide(7,[1,2],[1,2,3],[0,0],.10,{3},(),1)
n.commit([1],[terminal]);assert captured==[([terminal.accepted],[0])]and terminal.terminal
class Tok:
 all_special_ids=[30];added_tokens_decoder={29:types.SimpleNamespace(special=True),28:types.SimpleNamespace(special=False)};special_tokens_map={'eos_token':'eos'}
 def __len__(self):return 32
assert special_provenance(Tok(),{31})['protected_ids']==[29,30,31]
out={'passed':True,'GPU':False,'native_dlopen':False,'policy_cases':checks,'real_numpy_BF16_scalar_gather':True,'real4thread_Native_finish_callback_order':True,'real_ctypes_commit_signature':True,'terminal_original_accepted_preserved_continuation0':True,'strict_OFF_original_decision':True,'negative_nonfinite_gap_strictfallback':True,'ragged_causal_alignment':True,'fullV_F32_conversion':False,'quality_or_speed_claim':False}
pathlib.Path(__file__).with_name('CPU-gap-contract-r9258.json').write_bytes(json.dumps(out,indent=2).encode());print(json.dumps(out))
