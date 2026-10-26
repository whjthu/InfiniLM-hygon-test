import torch,pathlib,json,threading,hashlib
from projected_context_r9200 import ProjectedContextStore
class Linear(torch.nn.Linear):
 def __init__(self):super().__init__(40,8,bias=False);self.calls=[]
 def forward(self,x):self.calls.append(tuple(x.shape));return super().forward(x)
class Norm(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(8));self.variance_epsilon=1e-6
 def forward(self,x):dtype=x.dtype;y=x.float();return self.weight*(y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.variance_epsilon)).to(dtype)
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.context_proj=Linear();self.context_norm=Norm();self.eval()
def weight_hash(m):return hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes()for x in m.parameters())).hexdigest()
def snap(c):return(c.epoch,c.values.copy(),c.retired.copy(),c.projection_calls,c.projected_rows)
def unchanged(c,s):assert c.epoch==s[0]and c.values.keys()==s[1].keys()and all(c.values[k][0]==s[1][k][0]and c.values[k][1]is s[1][k][1]for k in c.values)and c.retired==s[2]and(c.projection_calls,c.projected_rows)==s[3:]
def reject(c,fn):
 before=snap(c)
 try:fn();raise RuntimeError('must reject')
 except(AssertionError,ValueError):pass
 unchanged(c,before)
records=[];torch.manual_seed(9200);model=Model();start_hash=weight_hash(model);epoch=0;maxerr=0
with torch.inference_mode():
 for C in [1,8,16,32]:
  ids=([17]if C==1 else(list(range(1,32,4))if C==8 else(list(range(1,33,2))if C==16 else list(range(1,33)))));assert len(ids)==C
  for N in [128,384,512]:
   epoch+=1;c=ProjectedContextStore(model,start_hash);c.begin(epoch);raw={};active=ids[:]
   for round,width in enumerate([N,1,3,7,1,3,7]):
    order=active[::-1]if round%2 else active;items=[{'request':i,'past':0 if i not in raw else raw[i].shape[1],'rows':width}for i in order];parts=[torch.randn(1,width,40)*.1 for i in order];before=len(model.context_proj.calls);values=c.append_many(epoch,items,parts);assert len(model.context_proj.calls)==before+1 and model.context_proj.calls[-1]==(len(order)*width,40);assert c.last_append['projection_calls']==1
    for i,p,out in zip(order,parts,values):
     raw[i]=p if i not in raw else torch.cat([raw[i],p],dim=1);expected=model.context_norm(model.context_proj(raw[i]));err=float((out-expected).abs().max());maxerr=max(maxerr,err);assert torch.allclose(out,expected,rtol=1e-5,atol=1e-6)
    saved=snap(c);bad=[dict(x,past=x['past']+width)for x in items];bad[-1]['past']+=1;reject(c,lambda:c.append_many(epoch,bad,[torch.randn(1,width,40)for x in bad]));unchanged(c,saved)
    if round==3 and C>1:
     for i in active[1::2]:c.retire(epoch,i);raw.pop(i)
     active=active[::2]
   assert c.projection_calls==7;first=active[0];past=raw[first].shape[1];one={'request':first,'past':past,'rows':1};reject(c,lambda:c.append_many(epoch,[one,one],[torch.randn(1,1,40)]*2));reject(c,lambda:c.append_many(epoch+1,[one],[torch.randn(1,1,40)]));reject(c,lambda:c.append_many(epoch,[dict(one,rows=9)],[torch.randn(1,9,40)]));reject(c,lambda:c.append_many(epoch,[one],[torch.randn(1,1,39)]));reject(c,lambda:c.append_many(epoch,[one],[torch.ones(1,1,40,dtype=torch.int32)]))
   fork=c.fork();fork.append_many(epoch,[one],[torch.randn(1,1,40)]);assert c.values[(epoch,first)][0]==past and fork.values[(epoch,first)][0]==past+1
   if len(active)>1:
    its=[{'request':i,'past':raw[i].shape[1],'rows':1}for i in active[:2]];count=[0]
    def broken(a,b):
     count[0]+=1
     if count[0]==2:raise ValueError('syntheticcatfailure')
     return torch.cat([a,b],dim=1)
    reject(c,lambda:c.append_many(epoch,its,[torch.randn(1,1,40)for i in its],concat=broken))
   errors=[]
   def foreign():
    try:c.get_many(epoch,[first])
    except AssertionError:errors.append(True)
   t=threading.Thread(target=foreign);t.start();t.join();assert errors==[True]
   c.retire(epoch,first);reject(c,lambda:c.append_many(epoch,[dict(one,past=0)],[torch.randn(1,1,40)]));c.close();c.begin(epoch+1);c.append_many(epoch+1,[{'request':first,'past':0,'rows':128}],[torch.randn(1,128,40)]);c.close();records.append({'C':C,'prefill':N,'append_rounds':[1,3,7,1,3,7],'onepackedprojection_perappend':True,'sparsekeys_shrink_retire_epoch_owner_atomic_reject':True})
 c=ProjectedContextStore(model,start_hash);c.begin(1);c.append_many(1,[{'request':1,'past':0,'rows':512}],[torch.randn(1,512,40)])
 for j in range(64):c.append_many(1,[{'request':1,'past':512+8*j,'rows':8}],[torch.randn(1,8,40)])
 assert c.values[(1,1)][0]==1024;reject(c,lambda:c.append_many(1,[{'request':1,'past':1024,'rows':1}],[torch.randn(1,1,40)]));c.close()
 with torch.inference_mode(False):badmodel=Model()
 c=ProjectedContextStore(badmodel,'separateweightmutationprobe');c.begin(1)
 with torch.no_grad():badmodel.context_proj.weight.add_(1)
 reject(c,lambda:c.append_many(1,[{'request':1,'past':0,'rows':1}],[torch.randn(1,1,40)]));c.close()
assert weight_hash(model)==start_hash and not torch.cuda.is_initialized();q={'CPU_only':True,'GPU':False,'trained_model_math':False,'actual_rowwise_Linear_RMS':True,'cases':records,'max_abs_fullrecompute_vs_append':maxerr,'weights_SHA256_before_after':start_hash,'owner_epoch_retire_past_atomicfailure_length1024_append1to8_checked':True};pathlib.Path(__file__).with_name('CPU-contract-r9200.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
