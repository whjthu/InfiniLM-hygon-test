import pathlib,json,threading,socket,traceback,struct,hashlib,time
import torch
from committed_context_r9202 import CommittedFeatureContexts
from batch_policy_r9128 import greedy
from batch_wire_r9128 import send,recv
from batch_frames_r9142 import send_query,receive_bodies
import batch_draft_worker_r9202
class Linear(torch.nn.Linear):
 def __init__(self):super().__init__(40,8,bias=False);self.calls=[];self.fail=False
 def forward(self,x):
  self.calls.append(tuple(x.shape))
  if self.fail:raise RuntimeError('controlled CPUprojection error')
  return super().forward(x)
class Norm(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(8));self.variance_epsilon=1e-6
 def forward(self,x):d=x.dtype;y=x.float();return self.weight*(y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.variance_epsilon)).to(d)
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.context_proj=Linear();self.context_norm=Norm();self.eval()
def sha(m):return hashlib.sha256(b''.join(p.detach().numpy().tobytes()for p in m.parameters())).hexdigest()
def snapshot(s):return(s.raw.epoch,s.enabled,s.raw.values.copy(),s.projected.values.copy(),s.raw.retired.copy(),s.projected.retired.copy(),s.projected.projection_calls)
def same(s,old):
 assert(s.raw.epoch,s.enabled)==old[:2]
 for data,previous in [(s.raw.values,old[2]),(s.projected.values,old[3])]:assert data.keys()==previous.keys()and all(data[k][0]==v[0]and data[k][1]is v[1]for k,v in previous.items())
 assert(s.raw.retired,s.projected.retired,s.projected.projection_calls)==old[4:]
def reject(s,fn):
 old=snapshot(s)
 try:fn();raise ValueError('invalid transaction accepted')
 except(AssertionError,RuntimeError):pass
 same(s,old)
torch.set_num_threads(1);torch.manual_seed(9202);result={'CPU_only':True,'GPU':False,'trainedweights_loaded':False,'scaled_rowwise_h8_raw40_math':True,'actual_binary_feature_width':35840,'actual_binary_dtype':'BF16','cohorts':[]};errors=[];left,right=socket.socketpair();left.settimeout(30);right.settimeout(30);seq=0
# Execute real binary framing+ContextStore/ProjectedContextStore/atomicwrapper on owner server thread.
def worker():
 try:
  model=Model();weights=sha(model);state=CommittedFeatureContexts(model,weights);expected=0
  with torch.inference_mode():
   while True:
    m,body=recv(right);assert not body and m['seq']==expected;expected+=1;cmd=m['cmd']
    if cmd=='begin_cohort':state.begin(m['epoch'],m.get('projected_cache',False));send(right,{'epoch':m['epoch'],'projected_cache':state.enabled});continue
    if cmd=='retire':
     key=(m['epoch'],m['request']);other={k:id(v[1])for k,v in state.raw.values.items()if k!=key};state.retire(*key);assert other=={k:id(v[1])for k,v in state.raw.values.items()};send(right,{'retired':True});continue
    if cmd=='close_cohort':assert not state.raw.values;state.close();send(right,{'closed':True});continue
    if cmd=='close':state.close();assert sha(model)==weights;send(right,{'closed':True});break
    assert cmd=='query_batch';items=m['requests'];state.validate_many(m['epoch'],items);bodies=receive_bodies(right,m);parts=[]
    for q,b in zip(items,bodies):
     # Wire uses full actual BF16 feature rows. CPU rowwise math narrowed to40 only to avoid trained0.5GB CPU GEMMs.
     
     with torch.inference_mode(False):full=torch.frombuffer(bytearray(b),dtype=torch.bfloat16).reshape(1,q['rows'],35840);parts.append(full[:,:,:40].float().clone())
     assert bool(torch.isfinite(full).all())
    versions=[x._version for x in parts];calls=len(model.context_proj.calls);contexts=state.append_many(m['epoch'],items,parts,lambda:None);assert versions==[x._version for x in parts];delta=len(model.context_proj.calls)-calls;assert delta==int(state.enabled)
    if state.enabled:
     assert model.context_proj.calls[-1]==(sum(q['rows']for q in items),40)
     for q,out in zip(items,contexts):ref=model.context_norm(model.context_proj(state.raw.values[(m['epoch'],q['request'])][1]));assert torch.allclose(out,ref,rtol=1e-5,atol=2e-6)
    # Real wrapper whole-batch invalidpast/projectionfailure/fencefailure never publishes raw or projected dictionaries.
    valid=[dict(q,past=q['past']+q['rows'],rows=1)for q in items];bad=[dict(q)for q in valid];bad[-1]['past']+=1;ones=[torch.zeros(1,1,40)for q in items];reject(state,lambda:state.append_many(m['epoch'],bad,ones,lambda:None))
    def failfence():raise RuntimeError('controlled CPUfence failure')
    reject(state,lambda:state.append_many(m['epoch'],valid,ones,failfence))
    if state.enabled:
     model.context_proj.fail=True;reject(state,lambda:state.append_many(m['epoch'],valid,ones,lambda:None));model.context_proj.fail=False
    old=snapshot(state);reject(state,lambda:state.begin(m['epoch']+1,False));same(state,old)
    send(right,{'requests':[{'request':q['request'],'past':state.raw.values[(m['epoch'],q['request'])][0],'markers':state.raw.values[(m['epoch'],q['request'])][1][0,:,0].tolist()}for q in items],'projected_cache':state.enabled,'packed_projection_calls':delta})
  right.close()
 except BaseException as e:errors.append(traceback.format_exc())
t=threading.Thread(target=worker);t.start()
def rpc(cmd,**kw):
 global seq
 send(left,{'cmd':cmd,'seq':seq,**kw});seq+=1;return recv(left)[0]
def markerbytes(values):
 return b''.join(struct.pack('<H',int(torch.tensor(float(v),dtype=torch.bfloat16).view(torch.uint16)))*35840 for v in values)
try:
 for epoch,(C,on)in enumerate([(16,False),(16,True),(32,False),(32,True)],1):
  ids=list(range(1,33,2))if C==16 else list(range(1,33));W=4 if C==16 else 2;active=ids[:];past={i:0 for i in ids};expected={i:[]for i in ids};pending={i:[i]*4 for i in ids};hello=rpc('begin_cohort',epoch=epoch,projected_cache=on);assert hello['projected_cache']==on;queries=0
  for round in range(6):
   order=active[::-1]if round%2 else active;items=[{'request':i,'past':past[i],'rows':len(pending[i]),'anchor':100,'K':W-1}for i in order];bodies=[markerbytes(pending[i])for i in order];send_query(left,{'cmd':'query_batch','seq':seq,'epoch':epoch,'requests':items},bodies);seq+=1;answer,_=recv(left);assert answer['projected_cache']==on and answer['packed_projection_calls']==int(on)
   for i,row in zip(order,answer['requests']):expected[i]+=pending[i];past[i]+=len(pending[i]);assert row['request']==i and row['past']==past[i]and row['markers']==expected[i]and max(row['markers'])<180
   # Actual target greedy Decision chooses different acceptedprefix perrequest. Only its retained verifyfeature rows go next RPC.
   for i in active:
    proposals=list(range(1,W));a=(i+round)%W;target=proposals[:a]+[99]+[77]*(W-1-a);d=greedy(100,proposals,target,eos=[999],remaining=100);assert len(d.commit_inputs)==a+1
    all_verify_features=markerbytes([20+i+round]*len(d.commit_inputs)+[180+i]*(W-len(d.commit_inputs)));committed=all_verify_features[:len(d.commit_inputs)*35840*2];vals=torch.frombuffer(bytearray(committed),dtype=torch.bfloat16).reshape(-1,35840)[:,0].float().tolist();assert len(vals)==len(d.commit_inputs)and max(vals)<180;pending[i]=vals
   if round==2:
    retired=active[1::2]
    for i in retired:assert rpc('retire',epoch=epoch,request=i)['retired']
    active=active[::2]
   queries+=1
  for i in active:rpc('retire',epoch=epoch,request=i)
  assert rpc('close_cohort')['closed'];result['cohorts'].append({'C':C,'projected_cache':on,'actual_socket_query_batches':queries,'reorder_shrink_retire_epoch':True,'actual_greedy_commit_crop_never_cached_rejected_suffix':True,'onepackedprojection_perONbatch':on,'wholebatch_invalidpast_projection_fence_failure_atomic':True})
 assert rpc('close')['closed'];left.close();t.join(10);assert not t.is_alive()and not errors,errors
finally:
 try:left.close()
 except:pass
model=Model();isolated=CommittedFeatureContexts(model,sha(model));reject(isolated,lambda:isolated.begin(1,1));isolated.begin(1,True);single=[{'request':1,'past':0,'rows':512}];isolated.append_many(1,single,[torch.randn(1,512,40)],lambda:None)
for j in range(64):isolated.append_many(1,[{'request':1,'past':512+j*8,'rows':8}],[torch.randn(1,8,40)],lambda:None)
reject(isolated,lambda:isolated.append_many(1,[{'request':1,'past':1024,'rows':1}],[torch.zeros(1,1,40)],lambda:None));ownererrors=[]
def foreign():
 try:isolated.retire(1,1)
 except AssertionError:ownererrors.append(True)
th=threading.Thread(target=foreign);th.start();th.join();assert ownererrors==[True];isolated.retire(1,1);isolated.close();result['CPU_length1024_strictmodebool_owner_checked']=True
assert not torch.cuda.is_initialized();result['complete']=True;result['source_entry_safe_imported']=True;pathlib.Path(__file__).with_name('CPU-contract-r9202.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
