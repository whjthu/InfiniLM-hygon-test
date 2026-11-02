"""CPU-only reduced40->8 actual333Session; no checkpoint/Native/GPU."""
from pathlib import Path
import argparse,json,sys,threading,time

def main():
 a=argparse.ArgumentParser();a.add_argument('--incoming',required=True);a.add_argument('--output',required=True);a=a.parse_args()
 D=Path(__file__).resolve().parent;sys.path[:0]=[str(D),str(Path(a.incoming)/'Kimi-long-draft-RPC-CPU-r333')]
 import torch
 from long_context_r323 import LongCommittedFeatureContext
 from long_rpc import DraftSession,TorchFeatureCodec
 from context import ProjectedOnlyContext
 assert not torch.cuda.is_initialized();torch.set_num_threads(4);torch.manual_seed(384)
 class Norm(torch.nn.Module):
  variance_epsilon=1e-6
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(8))
  def forward(self,x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.variance_epsilon)).to(x.dtype)*self.weight
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.context_proj=torch.nn.Linear(40,8,bias=False);self.context_norm=Norm()
 cases=[];numeric=[]
 def expect(label,fn):
  try:fn()
  except Exception as error:cases.append(dict(case=label,passed=True,error=repr(error)));return
  raise AssertionError('Expected rejection '+label)
 for dtype in [torch.bfloat16,torch.float32]:
  model=Model().to(dtype).eval();before={k:v.detach().clone()for k,v in model.state_dict().items()}
  class Codec(TorchFeatureCodec):
   def decode(self,body,rows):return super().decode(body,rows).to(dtype)
  def pair():return [(c,DraftSession(c,None,Codec('cpu',raw_dim=40),enabled=True))for c in [LongCommittedFeatureContext(model,'384frozen',enabled=True),ProjectedOnlyContext(model,'384frozen',enabled=True,discard_raw=True)]]
  def call(c,s,cmd,part=None,**fields):
   meta=dict(cmd=cmd,seq=s.sequence)
   if cmd!='close':meta.update(epoch=c.epoch if cmd!='begin'else fields.pop('epoch'),request=1)
   body=b''
   if part is not None:
    n=part.shape[1];past=c.length;meta.update(past=past,rows=n,capture_epoch=s.last_capture_epoch+1)
    if cmd=='append_committed':meta['target_commit']=dict(capture_epoch=meta['capture_epoch'],past_before=past,past_after=past+n,accepted_inputs=n,complete=True)
    body=part.contiguous().view(torch.uint16).numpy().tobytes()
   meta.update(fields);return s.dispatch(meta,body)
  with torch.inference_mode():
   assert ProjectedOnlyContext.query is LongCommittedFeatureContext.query and ProjectedOnlyContext.query_plan is LongCommittedFeatureContext.query_plan
   default=ProjectedOnlyContext(model,'384default',enabled=True);assert not default.discard_raw;default.close()
   for L in [513,8635,32638]:
    pairs=pair();parts=(torch.randn(1,L+16,40,dtype=torch.bfloat16)*.1);frozen=parts.clone()
    for c,s in pairs:
     call(c,s,'begin',epoch=1,prompt_tokens=L)
     for p in range(0,L,512):call(c,s,'append_prompt',parts[:,p:min(p+512,L)])
     call(c,s,'seal',past=L)
    for cycle,n in enumerate([0,1,7,8]):
     if n:
      past=L+sum([0,1,7,8][:cycle]);part=parts[:,past:past+n]
      for c,s in pairs:call(c,s,'append_committed',part)
     off,on=pairs[0][0],pairs[1][0];assert on.raw is None and off.raw is not None
     x=off.projected.float();y=on.projected.float();assert torch.isfinite(x).all()and torch.isfinite(y).all()
     nr=float(torch.linalg.vector_norm(x-y)/(torch.linalg.vector_norm(x)+1e-30));mx=float((x-y).abs().max());assert mx==0
     payload=on.payload_bytes();assert payload['raw']==0 and payload['projected']==on.length*8*model.context_proj.weight.element_size() and payload['published_raw_input_bytes']==on.length*40*model.context_proj.weight.element_size()
     numeric.append(dict(dtype=str(dtype),L=L,cycle=cycle,NRMSE=nr,maxabs=mx,payload=payload))
    assert torch.equal(parts,frozen)
    for c,s in pairs:
     assert call(c,s,'retire')['removed']is True;assert c.raw is None and c.projected is None
     call(c,s,'begin',epoch=2,prompt_tokens=1);call(c,s,'append_prompt',parts[:,:1]);assert call(c,s,'retire')['removed']is True
     call(c,s,'close');assert c.close()is False
    cases.append(dict(case='actual333Session512chunks_partial1_7_8_new_epoch_retire_close',dtype=str(dtype),L=L,passed=True))
   # This reaches the original Session -> inherited query -> AR-tail branch.
   # A simple reduced projection model has no draft layers/head to fake-load.
   pairs=pair();tail=torch.zeros(1,32761,40,dtype=torch.bfloat16)
   for c,s in pairs:
    call(c,s,'begin',epoch=1,prompt_tokens=32761)
    for p in range(0,32761,512):call(c,s,'append_prompt',tail[:,p:min(p+512,32761)])
    call(c,s,'seal',past=32761)
    answer=call(c,s,'query',past=32761,anchor=100,K=7)
    assert answer['mode']=='ar_tail'and answer['proposals']==[]
    assert call(c,s,'retire')['removed']is True;call(c,s,'close')
   cases.append(dict(case='actual333Session32761_inherited_query_ARtail_no_head_access',dtype=str(dtype),passed=True))
   c,s=pair()[1];call(c,s,'begin',epoch=1,prompt_tokens=2)
   expect('fence_failure_no_publish',lambda:c.append_prompt(1,1,0,torch.ones(1,1,40,dtype=dtype),lambda:(_ for _ in ()).throw(RuntimeError('fence failure'))))
   assert c.length==0 and c.projected is None and c.raw is None
   expect('stale_epoch',lambda:c.append_prompt(2,1,0,torch.ones(1,1,40,dtype=dtype),lambda:None))
   owner_errors=[]
   def foreign():
    try:c.close()
    except AssertionError as e:owner_errors.append(repr(e))
   t=threading.Thread(target=foreign);t.start();t.join(2);assert not t.is_alive()and owner_errors;c.close()
   cases.append(dict(case='foreign_owner_rejected',dtype=str(dtype),passed=True))
   c,s=pair()[1];call(c,s,'begin',epoch=1,prompt_tokens=1)
   expect('nonfinite_frame',lambda:call(c,s,'append_prompt',torch.full((1,1,40),float('nan'),dtype=torch.bfloat16)))
   assert s.poisoned and c.length==0 and c.projected is None;c.close()
   assert all(torch.equal(v,before[k])for k,v in model.state_dict().items())
 result=dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_tensors=0,cases=cases,numeric=numeric,scope='Actual Torch reduced40to8 projection/norm and333Session; query method inherited unchanged by source. No draft5layer/checkpoint/head/selector/Model quality or performance qualification.')
 Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
