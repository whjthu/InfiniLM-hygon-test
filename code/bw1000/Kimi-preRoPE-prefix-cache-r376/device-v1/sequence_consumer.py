"""Exact18context/144query consumer; all numeric fields saved before investigation stop."""
import time,traceback,copy
import torch
from long_context_r323 import LongCommittedFeatureContext
from long_rpc import DraftSession
from prefix_cache import CachedContext,PrivateSession,committing_codec,full_query

ALERT=.02
INVESTIGATE=.10

def plan():
 return [dict(ordinal=i*6+j,L=L,phase='warm'if j<2 else'formal',cache=enabled,arm=('ON'if enabled else'OFF'),sequence_in_geometry=j)
         for i,L in enumerate([1024,8635,32638])for j,enabled in enumerate([False,True,False,True,True,False])]

class FullContext(LongCommittedFeatureContext):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.last_debug=None;self.last_query_values=None
 def query(self,epoch,request,head,anchor,K=7,*,return_debug=False):
  self.last_debug=self.last_query_values=None;plan=self.query_plan(epoch,request,anchor,K)
  if plan['mode']=='ar_tail':return dict(plan,values=None)
  values,debug=full_query(self.model,head,self.projected,anchor,K);self.last_debug=debug;self.last_query_values=values
  return dict(plan,values=(values,debug)if return_debug else values)
 def close(self):
  try:return super().close()
  finally:self.last_debug=self.last_query_values=None
 def retire(self,*args):
  try:return super().retire(*args)
  finally:self.last_debug=self.last_query_values=None

class ObservedCache(CachedContext):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.last_query_values=None
 def query(self,*args,return_debug=False,**kwargs):
  self.last_debug=self.last_query_values=None
  out=super().query(*args,return_debug=True,**kwargs)
  if out['mode']=='draft':self.last_query_values=out['values'][0]
  if not return_debug and out['mode']=='draft':out=dict(out,values=out['values'][0])
  return out
 def close(self):
  try:return super().close()
  finally:self.last_query_values=None

class StageSamples:
 """All5layers/4stages, fixed3contextrows+all8draft; not fullactivation equality."""
 def __init__(self,model,L):self.model=model;self.L=L;self.values={};self.handles=[]
 def __enter__(self):
  for i,layer in enumerate(self.model.layers):
   for name in ['kv_a_proj_with_mqa','kv_a_layernorm','kv_b_proj','o_proj']:
    def hook(module,args,out,key=str(i)+':'+name,kind=name):
     rows=sorted(set([0,self.L//2,self.L-1]))+list(range(self.L,self.L+8))if kind!='o_proj'else list(range(8))
     ids=torch.tensor(rows,dtype=torch.long,device=out.device)
     self.values[key]=out.detach().index_select(1,ids)
    self.handles.append(layer.self_attn.get_submodule(name).register_forward_hook(hook))
  return self
 def __exit__(self,*args):
  for h in self.handles:h.remove()
  self.handles=[]

def metric(reference,value):
 a=reference.float();b=value.float();finite=bool(torch.isfinite(a).all()and torch.isfinite(b).all())
 nr=float(torch.linalg.vector_norm(a-b)/(torch.linalg.vector_norm(a)+1e-30))if finite else None
 return dict(finite=finite,NRMSE=nr,maxabs=float((a-b).abs().max())if finite else None,above_02_alert=(nr is None or nr>ALERT))

def run_sequence(model,head,base_codec,device,raw_dim,weight_tag,body,spec,save,reference=None,*,remaining=lambda:None,fail_numeric=False):
 """Continuous checked8cycle wall includes inline readback/metrics/save overhead.

Complete query/append dispatch wall sums additionally expose engine boundaries;
no crosshost clocks and no component time substituted for emitter/Model cost.
"""
 enabled=spec['cache'];c=ObservedCache(model,weight_tag,enabled=True,cache_enabled=True)if enabled else FullContext(model,weight_tag,enabled=True)
 codec=(committing_codec(base_codec,c)if enabled else base_codec)(device,raw_dim=raw_dim)
 session=(PrivateSession if enabled else DraftSession)(c,head,codec,enabled=True)
 q=dict(**spec,completed=False,passed=False,queries=[],commands=[],numeric_stop_threshold=INVESTIGATE,alert_threshold=ALERT,quality_qualified=False)
 references=[];capture=0
 def command(cmd,data=b'',**fields):
  remaining();meta=dict(cmd=cmd,seq=session.sequence)
  if cmd!='close':meta.update(epoch=1,request=1)
  meta.update(fields);tick=time.perf_counter();out=session.dispatch(meta,data)
  q['commands'].append(dict(cmd=cmd,past=fields.get('past'),rows=fields.get('rows'),wall_s=time.perf_counter()-tick));return out
 def append(cmd,past,n):
  nonlocal capture
  capture+=1;fields=dict(past=past,rows=n,capture_epoch=capture)
  if cmd=='append_committed':fields['target_commit']=dict(capture_epoch=capture,past_before=past,past_after=past+n,accepted_inputs=n,complete=True)
  return command(cmd,body(past,n),**fields)
 begin=time.perf_counter();save(q)
 try:
  tick=time.perf_counter();command('begin',prompt_tokens=spec['L'])
  for past in range(0,spec['L'],512):append('append_prompt',past,min(512,spec['L']-past))
  command('seal',past=c.length);q['prefix_begin_append_seal_wall_s']=time.perf_counter()-tick;save(q)
  sequence_start=time.perf_counter()
  for cycle in range(8):
   if cycle:append('append_committed',c.length,8)
   expected=spec['L']+cycle*8;assert c.length==expected and c.phase=='decode'
   ids=torch.tensor(sorted(set([0,c.length//2,c.length-1])),device=c.raw.device,dtype=torch.long)
   raw_canary=c.raw.index_select(1,ids);projected_canary=c.projected.index_select(1,ids)
   row=dict(cycle=cycle,past=c.length,completed=False);q['queries'].append(row)
   answer=None
   try:
    with StageSamples(c.model,c.length)as stages:
     tick=time.perf_counter();answer=command('query',past=c.length,anchor=163584,K=7);row['complete_query_dispatch_wall_s']=time.perf_counter()-tick
   except BaseException as e:
    row.update(dispatch_error=repr(e),traceback=traceback.format_exc())
    # Fresh currentdebug only; source failures before production explicitly unavailable.
    debug=c.last_debug
    row['produced_hidden_available']=debug is not None and 'hidden'in debug
    row['produced_logits_available']=debug is not None and 'logits'in debug
    row['observed_proposals']=None if c.last_query_values is None else c.last_query_values.detach().cpu().tolist()
    save(q);raise
   debug=c.last_debug
   produced=dict(hidden=debug['hidden'].detach().float().cpu(),logits=debug['logits'].detach().float().cpu(),stages={k:v.float().cpu()for k,v in stages.values.items()},proposals=c.last_query_values.detach().cpu().tolist()[0])
   refs=reference[cycle]if reference is not None else produced
   if fail_numeric:refs=dict(refs,hidden=torch.zeros_like(produced['hidden']),logits=torch.zeros_like(produced['logits']))
   row.update(hidden=metric(refs['hidden'],produced['hidden']),logits=metric(refs['logits'],produced['logits']),stages={k:metric(refs['stages'][k],v)for k,v in produced['stages'].items()},proposals=produced['proposals'],response_proposals=answer['proposals'],proposal_position_matches=sum(a==b for a,b in zip(refs['proposals'],produced['proposals'])),state=dict(length=c.length,epoch=c.epoch,phase=c.phase),canary_scope='3fixedrows raw/projected, no fullcanary claim',completed=True)
   row['canary_passed']=bool(torch.equal(raw_canary,c.raw.index_select(1,ids))and torch.equal(projected_canary,c.projected.index_select(1,ids)))
   assert len(produced['proposals'])==7 and produced['proposals']==answer['proposals']and all(type(v)is int and 0<=v<163840 for v in produced['proposals'])
   if enabled:
    stats=c.last_stats;assert c.pending is None and c.cached_length==c.length and len(c.cache)==10
    assert stats['published_after_original_fence']and len(stats['linear_calls'])==10
    assert all(v['computed_rows']==(spec['L']+8 if cycle==0 else 16)for v in stats['linear_calls'])
    row['cache']=dict(mode=stats['mode'],cached_length=c.cached_length,delta=stats['delta'],computed_rows=sorted(set(v['computed_rows']for v in stats['linear_calls'])),base_compute_host_wall_s=sum(v['compute_host_wall_s']for v in stats['linear_calls']),cat_host_wall_s=sum(v['cat_host_wall_s']for v in stats['linear_calls']),cat_output_bytes=sum(v['cat_output_bytes']for v in stats['linear_calls']),pending_logical_bytes=stats['pending_bytes'])
   references.append(produced);save(q)
   # All stage/hidden/logit/proposal/state fields are already persisted.
   if not row['canary_passed']or any(not n['finite']for n in [row['hidden'],row['logits']]+list(row['stages'].values())):raise RuntimeError('State/canary/nonfinite check failed; fields saved')
   if any(row[k]['NRMSE']>INVESTIGATE for k in ['hidden','logits']):
    row['investigation_stop']=True;save(q);raise RuntimeError('Finite large drift investigation stop; not a globalquality judgment')
  q['continuous_checked_8cycle_wall_s']=time.perf_counter()-sequence_start
  q['query_dispatch_sum_s']=sum(v['wall_s']for v in q['commands']if v['cmd']=='query')
  q['accepted_append_dispatch_sum_s']=sum(v['wall_s']for v in q['commands']if v['cmd']=='append_committed')
  q['engine_dispatch_8cycle_sum_s']=q['query_dispatch_sum_s']+q['accepted_append_dispatch_sum_s']
  tick=time.perf_counter();command('retire');assert command('close')['closed'];q['retire_close_wall_s']=time.perf_counter()-tick
  assert c.raw is None and c.projected is None and (not enabled or not c.cache and c.pending is None)
  q.update(passed=True,normal_session_close=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());save(q);raise
 finally:
  try:
   if not session.closed:session.close()
  except BaseException as e:q.update(passed=False,close_error=repr(e));raise
  finally:q.update(completed=True,total_with_prefix_numeric_and_close_wall_s=time.perf_counter()-begin);save(q)
 return q,references

def run_plan(model,head,base_codec,device,raw_dim,weight_tag,body,save_sequence,*,remaining=lambda:None,after_sequence=lambda q:None):
 references={};records=[]
 for spec in plan():
  remaining()
  q,ref=run_sequence(model,head,base_codec,device,raw_dim,weight_tag,body,spec,save_sequence,reference=references.get(spec['L']),remaining=remaining)
  if spec['phase']=='warm'and not spec['cache']:references[spec['L']]=ref
  records.append(q);after_sequence(q)
 return records,references
