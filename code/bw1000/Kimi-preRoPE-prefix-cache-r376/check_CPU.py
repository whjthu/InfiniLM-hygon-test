"""Actual pinned tiny factory/333 Session, fixed BF16 wire, F32 reference-only decode."""
from pathlib import Path
import sys,json,time,threading,copy,gc,resource,hashlib

def main():
 D=Path(__file__).resolve().parent;R=Path('/refactor');inc=R/'incoming'
 sys.path[:0]=[str(D),str(inc/'Kimi-long-draft-RPC-CPU-r333'),str(inc)]
 import torch
 from dflash2_creator_factory_r9007 import make_factory
 from long_context_r323 import LongCommittedFeatureContext
 from long_rpc import DraftSession,TorchFeatureCodec
 from prefix_cache import CachedContext,PrivateSession,committing_codec,full_query,version
 torch.set_num_threads(4);torch.manual_seed(376)
 config=json.loads((inc/'config.json').read_bytes());config.update(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4)
 config['dflash_config']=dict(config['dflash_config'],selector_rank=8,selector_top_k=4)
 results=[];cases=[];closes=[];start=time.perf_counter()
 def progress(**data):
  p=R/'candidates'/D.name/'check-progress.json';p.write_text(json.dumps(data)+'\n')
 def diff(a,b):
  x=a.detach().float();y=b.detach().float();finite=bool(torch.isfinite(x).all()and torch.isfinite(y).all())
  assert finite
  nr=float(torch.linalg.vector_norm(x-y)/(torch.linalg.vector_norm(x)+1e-30))
  return dict(finite=True,NRMSE=nr,maxabs=float((x-y).abs().max()),reference_RMS=float(x.square().mean().sqrt()),above_02_diagnostic=nr>.02)
 def storage(values):
  seen={}
  for v in values:
   s=v.untyped_storage();seen[(str(v.device),s.data_ptr())]=s.nbytes()
  return sum(seen.values())
 class FullContext(LongCommittedFeatureContext):
  def __init__(self,*a,**k):super().__init__(*a,**k);self.last_debug=None
  def query(self,epoch,request,head,anchor,K=7,*,return_debug=False):
   plan=self.query_plan(epoch,request,anchor,K)
   if plan['mode']=='ar_tail':return dict(plan,values=None)
   values,debug=full_query(self.model,head,self.projected,anchor,K);self.last_debug=debug
   return dict(plan,values=(values,debug)if return_debug else values)
 class Trace:
  def __init__(self,model):self.values={};self.handles=[];self.model=model
  def __enter__(self):
   for i,l in enumerate(self.model.layers):
    for name in ['kv_a_proj_with_mqa','kv_a_layernorm','kv_b_proj','o_proj']:
     def hook(m,a,out,key=str(i)+':'+name):self.values[key]=out.detach().clone()
     self.handles.append(l.self_attn.get_submodule(name).register_forward_hook(hook))
   return self
  def __exit__(self,*a):
   for h in self.handles:h.remove()
 class Harness:
  def __init__(self,model,head,dtype,on):
   self.context=CachedContext(model,'CPU376fixed',enabled=True,cache_enabled=True)if on else FullContext(model,'CPU376fixed',enabled=True)
   c=self.context;self.fences=[];self.fail_fence=False;self.dtype=dtype;self.head=head;self.capture=0;self.costs=[]
   owner=self
   class TypedCodec(TorchFeatureCodec):
    def decode(self,body,rows):return super().decode(body,rows).to(dtype)
    def fence(self):
     super().fence()
     pending=getattr(c,'pending',None);old=getattr(c,'cache',{})
     vals=list(old.values())+(list(pending['values'].values())if pending else[])
     owner.fences.append(dict(pending=pending is not None,cached_length=getattr(c,'cached_length',0),canonical_length=c.length,publish_count=getattr(c,'publish_count',0),cache_storage_bytes=storage(vals)))
     if owner.fail_fence and pending is not None:raise RuntimeError('Injected query final fence failure')
   codec=(committing_codec(TypedCodec,c)if on else TypedCodec)('cpu',raw_dim=160)
   self.session=(PrivateSession if on else DraftSession)(c,head,codec,enabled=True)
  def call(self,cmd,body=b'',**fields):
   meta=dict(cmd=cmd,seq=self.session.sequence)
   if cmd!='close':meta.update(epoch=self.context.epoch if cmd!='begin'else fields.pop('epoch'),request=1)
   meta.update(fields);tick=time.perf_counter();answer=self.session.dispatch(meta,body)
   self.costs.append(dict(cmd=cmd,past=fields.get('past'),rows=fields.get('rows'),wall_s=time.perf_counter()-tick));return answer
  def append(self,cmd,part):
   c=self.context;n=part.shape[1];self.capture+=1;past=c.length
   fields=dict(past=past,rows=n,capture_epoch=self.capture)
   if cmd=='append_committed':fields['target_commit']=dict(capture_epoch=self.capture,past_before=past,past_after=past+n,accepted_inputs=n,complete=True)
   body=part.contiguous().view(torch.uint16).numpy().tobytes()
   return self.call(cmd,body,**fields)
  def begin(self,epoch,prompt):
   self.call('begin',epoch=epoch,prompt_tokens=prompt.shape[1])
   for p in range(0,prompt.shape[1],512):self.append('append_prompt',prompt[:,p:p+512])
   self.call('seal',past=self.context.length)
  def query(self):
   self.session._owner()
   with Trace(self.context.model)as trace:
    tick=time.perf_counter();answer=self.call('query',past=self.context.length,anchor=163584,K=7);wall=time.perf_counter()-tick
   return answer,wall,trace.values
  def close(self):
   if not self.session.closed:self.session.close()
 def expect(name,fn,types=(Exception,)):
  progress(phase='expected_rejection',name=name)
  try:fn()
  except types as e:cases.append(dict(name=name,passed=True,actual_error=repr(e)));return
  raise AssertionError('Expected rejection '+name)

 for dtype in [torch.bfloat16,torch.float32]:
  progress(phase='construct_tiny_model',dtype=str(dtype))
  startup=time.perf_counter();torch.manual_seed(376);model=make_factory(config,inc)().to(dtype).eval();head=torch.nn.Linear(32,163840,bias=False,dtype=dtype).eval();startup=time.perf_counter()-startup
  weights={k:v.detach().clone()for k,v in model.state_dict().items()};head_before=head.weight.detach().clone();ids=[(n,id(v))for n,v in model.named_modules()]
  off=Harness(model,head,dtype,False);on=Harness(model,head,dtype,True)
  default=CachedContext(model,'CPUdefaultOFF',enabled=True);assert default.cache_enabled is False;default.close()
  assert all(type(a.self_attn).forward is type(b.self_attn).forward for a,b in zip(model.layers,on.context.model.layers))
  assert all(a.self_attn.rotary_emb is b.self_attn.rotary_emb and a.self_attn.kv_a_layernorm is b.self_attn.kv_a_layernorm for a,b in zip(model.layers,on.context.model.layers))
  with torch.inference_mode():
   for epoch,L in enumerate([513,8635,32638],1):
    source=(torch.randn(1,L+56,160,dtype=torch.bfloat16)*.1)
    off_offset=len(off.costs);on_offset=len(on.costs)
    off.begin(epoch,source[:,:L]);on.begin(epoch,source[:,:L]);qrows=[];tick=time.perf_counter()
    for cycle in range(8):
     progress(phase='actual_full_sequence',dtype=str(dtype),prompt=L,cycle=cycle)
     if cycle:
      part=source[:,L+(cycle-1)*8:L+cycle*8];off.append('append_committed',part);on.append('append_committed',part)
     cached_before=on.context.cached_length;published_before=on.context.publish_count
     a,ta,sa=off.query();b,tb,sb=on.query();c=on.context
     assert set(sa)==set(sb);stage={key:diff(sa[key],sb[key])for key in sa}
     stats=copy.deepcopy(c.last_stats)
     expected=L+8 if cycle==0 else 16
     assert all(v['computed_rows']==expected for v in stats['linear_calls'])
     assert c.pending is None and c.cached_length==c.length and len(c.cache)==10
     assert stats['published_after_original_fence']
     fence=copy.deepcopy(on.fences[-1]);assert fence['pending']and fence['cached_length']==cached_before and fence['publish_count']==published_before
     assert c.publish_count==published_before+1
     qrows.append(dict(cycle=cycle,past=c.length,OFF_complete_dispatch_s=ta,ON_complete_dispatch_s=tb,hidden=diff(off.context.last_debug['hidden'],c.last_debug['hidden']),logits=diff(off.context.last_debug['logits'],c.last_debug['logits']),proposal_matches=sum(x==y for x,y in zip(a['proposals'],b['proposals'])),stages=stage,cache_stats=stats,original_final_fence_before_publication=fence,cache_logical_bytes=sum(t.numel()*t.element_size()for t in c.cache.values()),cache_storage_bytes=storage(c.cache.values())))
    total=time.perf_counter()-tick
    off.call('retire');on.call('retire');assert not on.context.cache and on.context.pending is None
    of=off.costs[off_offset:];oc=on.costs[on_offset:]
    results.append(dict(dtype=str(dtype),prompt=L,rows=qrows,tiny_model_head_startup_CPU_s=startup,OFF_protocol_costs=of,ON_protocol_costs=oc,OFF_complete_sequence_CPU_s=sum(v['wall_s']for v in of),ON_complete_sequence_CPU_s=sum(v['wall_s']for v in oc),ON_peak_live_cache_storage_bytes=max(v['cache_storage_bytes']for v in on.fences),full8cycle_interleaved_with_numeric_checks_CPU_wall_s=total,cold_first_retained=True,accepted_prefix_counts=[8]*7,sequence_scope='Originalrund8query + sevenaccepted8: fixedgeometry copied across3lengths; allpromptappend/seal/cold8query/acceptedappend/retire captured perarm. CPUtrace clones/validation/cat/pack/head/fences included in query wall; numeric comparisons outsidedispatch and separately affect interleaved wall; no GPU claim'))
   off.call('close');on.call('close');closes.append(dict(dtype=str(dtype),OFF=off.costs[-1],ON=on.costs[-1],all_cache_refs_dropped=not on.context.cache and on.context.pending is None))
   assert ids==[(n,id(v))for n,v in model.named_modules()]
   assert all(torch.equal(v,weights[k])for k,v in model.state_dict().items())and torch.equal(head_before,head.weight)
   assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
  # Small real session negatives and0/1/3/8/9 append cache paths.
  h=Harness(model,head,dtype,True);prompt=torch.randn(1,513,160,dtype=torch.bfloat16)*.1
  with torch.inference_mode():
   h.begin(1,prompt);h.query();old=h.context.cached_length
   h.query();assert all(v['computed_rows']==8 for v in h.context.last_stats['linear_calls']);assert h.context.cached_length==old
   cases.append(dict(name=str(dtype)+' repeat0append',passed=True))
   answer=h.call('query',past=h.context.length,anchor=163584,K=1);assert len(answer['proposals'])==1 and all(v['computed_rows']==8 for v in h.context.last_stats['linear_calls'])
   cases.append(dict(name=str(dtype)+' K1_keeps_fixed8_positions',passed=True))
   for n in [1,3,8]:
    h.append('append_committed',torch.randn(1,n,160,dtype=torch.bfloat16)*.1);h.query();assert all(v['computed_rows']==n+8 for v in h.context.last_stats['linear_calls'])
   h.append('append_committed',torch.randn(1,8,160,dtype=torch.bfloat16)*.1);h.append('append_committed',torch.randn(1,1,160,dtype=torch.bfloat16)*.1)
   h.query();assert h.context.last_stats['mode']=='full_refresh_gap'
   assert all(v['computed_rows']==h.context.length+8 for v in h.context.last_stats['linear_calls'])
   cases.append(dict(name=str(dtype)+' partial1_3_8_and_legalcumulative9_fullrefresh',passed=True))
   errors=[];before=h.context.cached_length
   def foreign():
    try:h.query()
    except BaseException as e:errors.append(repr(e))
   t=threading.Thread(target=foreign);t.start();t.join();assert errors and not h.session.poisoned and h.context.cached_length==before
   cases.append(dict(name=str(dtype)+' foreignowner_rejected_without_foreign_mutation',passed=True))
   # Exact real RPC rejection must poison and drop newly introduced storage.
   bad=dict(cmd='query',seq=h.session.sequence,epoch=999,request=1,past=h.context.length,anchor=163584,K=7)
   expect(str(dtype)+' stale_epoch',lambda:h.session.dispatch(bad,b''));assert h.session.poisoned and not h.context.cache and h.context.pending is None;h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query();h.fail_fence=True
   expect(str(dtype)+' final_query_fence',lambda:h.query());assert h.session.poisoned and not h.context.cache and h.context.pending is None;h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query()
   p=model.layers[0].self_attn.kv_a_proj_with_mqa.weight;copyweight=p.clone();p.add_(.001)
   expect(str(dtype)+' KVweightversion',lambda:h.query());assert not h.context.cache and h.context.pending is None;p.copy_(copyweight);h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query()
   capture=h.capture+1;past=h.context.length;bad=dict(cmd='append_committed',seq=h.session.sequence,epoch=1,request=1,past=past,rows=1,capture_epoch=capture,target_commit=dict(capture_epoch=capture,past_before=past,past_after=past+1,accepted_inputs=1,complete=False))
   expect(str(dtype)+' rejected_target_commit',lambda:h.session.dispatch(bad,torch.zeros(1,1,160,dtype=torch.bfloat16).view(torch.uint16).numpy().tobytes()));assert h.context.length==past and not h.context.cache;h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query()
   original_close=LongCommittedFeatureContext.close
   def broken_close(c):raise RuntimeError('Injected original close exception')
   LongCommittedFeatureContext.close=broken_close
   try:expect(str(dtype)+' close_exception',h.close);assert not h.context.cache and h.context.pending is None
   finally:LongCommittedFeatureContext.close=original_close;h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query()
   def broken_projection(m,a):raise RuntimeError('Injected layer2 projection failure')
   hook=model.layers[2].self_attn.kv_b_proj.register_forward_pre_hook(broken_projection)
   try:expect(str(dtype)+' layer2_projection_failure',lambda:h.query());assert not h.context.cache and h.context.pending is None
   finally:hook.remove();h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query();past=h.context.length
   hook=model.context_proj.register_forward_pre_hook(broken_projection)
   try:expect(str(dtype)+' append_projection_failure',lambda:h.append('append_committed',torch.zeros(1,1,160,dtype=torch.bfloat16)));assert h.context.length==past and not h.context.cache and h.context.pending is None
   finally:hook.remove();h.close()
   h=Harness(model,head,dtype,True);h.begin(1,prompt);h.query()
   def nonfinite(m,a,out):return out*float('nan')
   hook=model.layers[2].self_attn.kv_b_proj.register_forward_hook(nonfinite)
   try:expect(str(dtype)+' nonfinite_query',lambda:h.query());assert not h.context.cache and h.context.pending is None
   finally:hook.remove();h.close()
   assert all(torch.equal(v,weights[k])for k,v in model.state_dict().items())and torch.equal(head_before,head.weight)
  gc.collect()
 # Tail uses real32760/32761 boundary, no cache construction in originalARtail.
 h=Harness(model,head,torch.float32,True)
 with torch.inference_mode():
  prompt=torch.zeros(1,32760,160,dtype=torch.bfloat16);h.begin(1,prompt);h.query();h.append('append_committed',torch.zeros(1,1,160,dtype=torch.bfloat16));ans=h.call('query',past=h.context.length,anchor=163584,K=7);assert ans['mode']=='ar_tail'and not h.context.cache and h.context.pending is None;h.close()
 cases.append(dict(name='32760_to32761_ARtail_original',passed=True))
 with torch.inference_mode():assert version(torch.ones(1))is None
 cases.append(dict(name='inference_version_unavailable_supported_under_frozenweight_contract',passed=True))
 progress(phase='complete',cases=len(cases),full_sequences=len(results))
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_loaded=False,actual333_Session=True,actual_attention_forward_unchanged=True,cache_defaultOFF=True,wire_dtype='OriginalBF16; CPUF32case onlycasts decodedBF16 tofloat32 forreference',results=results,cases=cases,normal_final_closes=closes,total_CPU_wall_s=time.perf_counter()-start,peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,quality='Finite/stage/finalnumerics/proposals diagnostics only, no targetacceptance or Modelquality; no tokenbitwise requirement',timing='CPUdiagnostic completequery includes traceclones/pack/norm/attention/head/selector/pendingfinite/cat/codec fences. Not GPU cost; subsequent devicecard must report cold+8cycle sequentialOFF/ON plus startup and appends, not steadyonly.')))

if __name__=='__main__':main()
