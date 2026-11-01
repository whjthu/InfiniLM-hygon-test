"""Real Torch CPU consumer checks with reduced geometry; no checkpoint, target, or CUDA."""
import copy,json,threading,hashlib,pathlib,time
import torch
from long_context_r323 import LongCommittedFeatureContext,plan_query
from committed_context_r9202 import CommittedFeatureContexts

class RowNorm(torch.nn.Module):
 def __init__(self,h):
  super().__init__();self.weight=torch.nn.Parameter(torch.ones(h));self.variance_epsilon=1e-6
 def forward(self,x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.variance_epsilon)).to(x.dtype)*self.weight
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.context_proj=torch.nn.Linear(40,8,bias=False);self.context_norm=RowNorm(8)

def reject(fn):
 try:fn()
 except (AssertionError,RuntimeError):return
 raise AssertionError('Invalid operation was accepted')
def snapshot(s):return (s.epoch,s.request,s.phase,s.length,id(s.raw),id(s.projected),s.projection_calls,copy.deepcopy(s.last_append))
def run():
 torch.set_num_threads(1);torch.manual_seed(323);assert not torch.cuda.is_initialized()
 cases=[];t=time.perf_counter()
 for dtype in [torch.float32,torch.bfloat16]:
  model=Model().to(dtype=dtype).eval()
  reject(lambda:LongCommittedFeatureContext(model,'CPU323'))
  s=LongCommittedFeatureContext(model,'CPU323',enabled=True);s.begin(1,1,1024)
  parts=[torch.randn(1,512,40,dtype=dtype)for _ in range(2)]
  with torch.inference_mode():
   s.append_prompt(1,1,0,parts[0],lambda:None);before=snapshot(s)
   reject(lambda:s.append_prompt(1,1,512,torch.ones(1,513,40,dtype=dtype),lambda:None));assert snapshot(s)==before
   reject(lambda:s.append_prompt(1,1,512,torch.ones(1,512,39,dtype=dtype),lambda:None));assert snapshot(s)==before
   reject(lambda:s.seal_prompt(1,1));assert snapshot(s)==before
   reject(lambda:s.append_committed(1,1,512,parts[1][:,:8],lambda:None));assert snapshot(s)==before
   s.append_prompt(1,1,512,parts[1],lambda:None);s.seal_prompt(1,1)
   # Same per-chunk projection math as legacy projected cache (its second512 is rejected).
   expected=torch.cat([model.context_norm(model.context_proj(p[0])).unsqueeze(0)for p in parts],1)
   torch.testing.assert_close(s.projected,expected,rtol=0,atol=0)
   assert torch.equal(s.raw,torch.cat(parts,1)) and s.projection_calls==2
   for n in [1,8]:
    part=torch.randn(1,n,40,dtype=dtype);past=s.length;s.append_committed(1,1,past,part,lambda:None)
    assert s.length==past+n
   before=snapshot(s);part=torch.randn(1,8,40,dtype=dtype)
   def bad_fence():raise RuntimeError('CPU injected stream fence failure')
   reject(lambda:s.append_committed(1,1,s.length,part,bad_fence));assert snapshot(s)==before
   reject(lambda:s.append_prompt(1,1,s.length,part,lambda:None));assert snapshot(s)==before
   reject(lambda:s.append_committed(1,1,s.length-1,part,lambda:None));assert snapshot(s)==before
   reject(lambda:s.append_committed(1,True,s.length,part,lambda:None));assert snapshot(s)==before
   reject(lambda:s.append_committed(2,1,s.length,part,lambda:None));assert snapshot(s)==before
   reject(lambda:s.append_committed(1,1,s.length,torch.randn(1,9,40,dtype=dtype),lambda:None));assert snapshot(s)==before
   reject(lambda:s.begin(2,1,512));assert snapshot(s)==before
   errors=[]
   def foreign():
    try:s.query_plan(1,1,163584)
    except AssertionError:errors.append('owner')
   thread=threading.Thread(target=foreign);thread.start();thread.join();assert errors==['owner']
   assert s.retire(1,1);reject(lambda:s.query_plan(1,1,163584));reject(lambda:s.retire(1,1));reject(lambda:s.begin(1,1,512))
   s.begin(2,1,32760)
   for past in range(0,32760,512):
    rows=min(512,32760-past);s.append_prompt(2,1,past,torch.ones(1,rows,40,dtype=dtype),lambda:None)
   s.seal_prompt(2,1);assert s.query_plan(2,1,163584)['last_draft_position']==32767
   s.append_committed(2,1,32760,torch.ones(1,1,40,dtype=dtype),lambda:None)
   # Fallback is an actual consumer return before any head/query execution.
   fallback=s.query(2,1,None,163584);assert fallback['mode']=='ar_tail' and fallback['values']is None
   s.append_committed(2,1,32761,torch.ones(1,7,40,dtype=dtype),lambda:None);before=snapshot(s)
   reject(lambda:s.append_committed(2,1,32768,torch.ones(1,1,40,dtype=dtype),lambda:None));assert snapshot(s)==before
   assert s.payload_bytes()=={'raw':32768*40*torch.empty((),dtype=dtype).element_size(),'projected':32768*8*torch.empty((),dtype=dtype).element_size()}
   assert s.close();reject(lambda:s.query_plan(2,1,163584));s.begin(3,1,512)
   before=snapshot(s);model.context_proj.weight.add_(1);reject(lambda:s.append_prompt(3,1,0,parts[0],lambda:None));assert snapshot(s)==before;s.close()
  legacy=CommittedFeatureContexts(model,'CPU323-legacy',max_requests=1);legacy.begin(1,True)
  with torch.inference_mode():
   legacy.append_many(1,[{'request':1,'past':0,'rows':512}],[parts[0]],lambda:None)
   reject(lambda:legacy.append_many(1,[{'request':1,'past':512,'rows':512}],[parts[1]],lambda:None));assert legacy.lengths()=={'1':512};legacy.close()
  cases.append({'dtype':str(dtype),'projection_chunk_numeric_match':True,'continuous512_prompt_to32760':True,'committed1_and8':True,'atomic_fence_failure':True,'owner_epoch_retire_close':True,'capacity32768_and_overflow_rejected':True,'actual_AR_tail_return_no_query':True,'legacy_second512_still_rejected':True})
 for length in [1024,8192,32760]:assert plan_query(length,163584,7)['mode']=='draft'
 for length in [32761,32768]:assert plan_query(length,163584,1)['mode']=='ar_tail'
 reject(lambda:plan_query(32769,163584,7));reject(lambda:plan_query(1024,163584,8))
 assert not torch.cuda.is_initialized()
 return {'completed':True,'passed':True,'CPU_only':True,'GPU':0,'weights':0,'target':0,'HTTP':0,'cases':cases,'elapsed_s':time.perf_counter()-t,'Torch_version':torch.__version__,'scope':'Reduced40to8 actualTorchCPU projection/consumer lifecycle. No checkpoint quality, full query device, capture, acceptance or performance qualification.'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
