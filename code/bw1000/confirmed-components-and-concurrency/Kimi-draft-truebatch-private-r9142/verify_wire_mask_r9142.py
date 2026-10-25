import pathlib,json,hashlib,socket,threading,time
import numpy as np
from draft_context_r9128 import ContextStore
from batch_frames_r9142 import send_query,receive_bodies,FEATURE_BYTES
from batch_wire_r9128 import send,recv
checks=[]
for C in [1,2,4,8,16,32]:
 store=ContextStore();store.begin(1);items=[{'request':i+1,'past':0,'rows':1+i%3,'K':1+i%7,'anchor':100+i}for i in range(C)];parts=[bytes([i%251])*q['rows']*FEATURE_BYTES for i,q in enumerate(items)];a,b=socket.socketpair();errors=[];answer=[]
 def worker():
  try:
   m,body=recv(b);assert not body;blobs=receive_bodies(b,m);store.validate_many(1,m['requests']);store.append_many(1,m['requests'],blobs,lambda x,y:x+y);tick=time.perf_counter();time.sleep(.002);send(b,{'seq':m['seq'],'actual_model_batch_C':len(m['requests']),'ONE_mock_forward':True,'wall':time.perf_counter()-tick})
  except BaseException as e:errors.append(repr(e))
 t=threading.Thread(target=worker);t.start();tick=time.perf_counter();send_query(a,{'cmd':'query_batch','seq':0,'epoch':1,'requests':items},parts);h,body=recv(a);elapsed=time.perf_counter()-tick;t.join(3);a.close();b.close();assert not errors and h['actual_model_batch_C']==C and h['ONE_mock_forward']and elapsed>=h['wall'];assert len(store.values)==C
 before=dict(store.values);bad=[dict(q,past=q['rows'],rows=1)for q in items];bad[-1]['past']+=1
 try:store.append_many(1,bad,[b'x']*C,lambda x,y:x+y);raise RuntimeError('mustreject')
 except AssertionError:pass
 assert store.values==before
 if C>=2:
  valid=[dict(q,past=q['rows'],rows=1)for q in items];calls=[0]
  def fail(x,y):
   calls[0]+=1
   if calls[0]==2:raise ValueError('syntheticconcatfailure')
   return x+y
  try:store.append_many(1,valid,[b'x']*C,fail);raise RuntimeError('mustreject')
  except ValueError:pass
  assert store.values==before
 store.retire(1,1)
 try:store.append(1,1,0,1,b'x',lambda x,y:x+y);raise RuntimeError('retiredslotmustreject')
 except AssertionError:pass
 checks.append({'C':C,'one_logical_RPC':True,'feature_frames':C,'caller_wall_includes_mockcompute_and_transport':True,'elapsed_s':elapsed,'bad_raggedpast_atomic_reject':True,'retiredslot_no_reuse':True})
# Independent small SDPA algebra confirms paddingmask and row positions, not trainedmodelmath.
rng=np.random.default_rng(9142);lengths=[1,3,7,11];L=max(lengths);B=len(lengths);Q=rng.normal(size=(B,8,4));K=rng.normal(size=(B,L+8,4));V=rng.normal(size=(B,L+8,4));maxerr=0.
for i,N in enumerate(lengths):
 valid=np.r_[np.arange(L)<N,np.ones(8,dtype=bool)];dp=N+np.arange(8);keypos=np.r_[np.arange(L),dp];mask=valid[None,:]& (np.abs(dp[:,None]-keypos[None,:])<4096);scores=Q[i]@K[i].T/2;scores=np.where(mask,scores,-np.inf);a=np.exp(scores-scores.max(1,keepdims=True));a/=a.sum(1,keepdims=True);bat=a@V[i];ix=np.r_[np.arange(N),np.arange(L,L+8)];one=Q[i]@K[i,ix].T/2;one=np.exp(one-one.max(1,keepdims=True));one/=one.sum(1,keepdims=True);single=one@V[i,ix];maxerr=max(maxerr,float(np.max(np.abs(bat-single))))
assert maxerr<1e-12
q={'CPU_only':True,'GPU':False,'trained_Torch_math_tested':False,'logical_query_batch_C':[1,2,4,8,16,32],'checks':checks,'mask_SDPA_numpy_vs_unpadded_reference_max_abs':maxerr,'scope':'truebinarysocket/framing/cache/rollback/retiredleases/timing; CPU SDPA algebra. Actual pinnedTorch worker GPUfixture stillrequired'};pathlib.Path('CPU-wire-mask-contract-r9142.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
