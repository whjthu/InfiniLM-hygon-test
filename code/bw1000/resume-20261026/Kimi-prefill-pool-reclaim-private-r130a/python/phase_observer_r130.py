"""Bounded controller-local phase spans. No GPU event, sync, logits copy, network or file IO."""
import time,contextlib
class PhaseTrace:
 def __init__(self,node,stage,clock=None):
  self.node=node;self.stage=stage;self.clock=clock or time.perf_counter;self.enabled=False;self.active=False;self.events=[];self.stack=[];self.cycle=None;self.limit=6000;self.dropped=0;self.cycles=[]
 def configure(self,enabled,label):
  assert type(enabled)is bool and isinstance(label,str)and len(label)<=100
  assert not self.stack
  self.enabled=enabled;self.active=enabled;self.label=label;self.events=[];self.stack=[];self.cycle=None;self.dropped=0;self.cycles=[]
 def begin(self,name,**meta):
  if not self.active:return None
  assert len(self.events)+len(self.stack)<self.limit,'bounded phase capacity'
  token={'name':name,'start':self.clock(),'parent':self.stack[-1]['id']if self.stack else None,'id':len(self.events)+len(self.stack),'cycle':self.cycle,'node':self.node,'PP_stage':self.stage,'meta':meta};self.stack.append(token);return token
 def end(self,token):
  if token is None:return
  assert self.stack and self.stack[-1]is token,'phase stack must close LIFO'
  self.stack.pop();end=self.clock();assert end>=token['start'];self.events.append(dict(token,end=end,wall_s=end-token['start']))
 @contextlib.contextmanager
 def scope(self,name,**meta):
  token=self.begin(name,**meta)
  try:yield
  finally:self.end(token)
 def call(self,name,fn,*args,**kwargs):
  with self.scope(name):return fn(*args,**kwargs)
 def start_cycle(self,step,ids,widths):
  assert not self.stack
  self.cycle=step;self.active=self.enabled and 2<=step<34
  if self.active:self.cycles.append({'step':step,'logical':list(ids),'widths':list(widths),'flattened_T':sum(widths),'actual_active_C':len(ids)})
  return self.begin('cycle',logical=list(ids),widths=list(widths),flattened_T=sum(widths))
 def end_cycle(self,token):
  self.end(token);assert not self.stack;self.active=False
 def export(self):
  assert not self.stack
  return {'enabled':self.enabled,'clock':'one controller perf_counter; node clocks NOT synchronized','node':self.node,'PP_stage':self.stage,'warm_verify_cycles_skipped':2,'max_recorded_steady_cycles':32,'recorded_cycles':len(self.cycles),'actual_T16_cycles':sum(x['flattened_T']==16 for x in self.cycles),'cycle_geometry':self.cycles,'events':list(self.events),'dropped':self.dropped,'GPU_MoE_events':'UNKNOWN: current provider PROFILE_R9125 adds persegment sync/keeps last layer; no non-sync all-layer event consumer enabled','GPU_busy_PPorCCL':'UNKNOWN','warning':'nested parent/child and StageControl/memory_guard spans overlap; never add across labels or nodes'}
class PhaseControl:
 def __init__(self,control,trace):self.control=control;self.trace=trace
 def __getattr__(self,name):return getattr(self.control,name)
 def collect(self,label,value):
  with self.trace.scope('StageControl.collect',label=label):return self.control.collect(label,value)
def interval_union(intervals):
 rows=sorted((float(a),float(b))for a,b in intervals);total=0.;end=None
 for a,b in rows:
  assert b>=a
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total
def summarize(trace):
 events=trace['events'];by={}
 for e in events:by.setdefault(e['name'],[]).append((e['start'],e['end']))
 cycles=[e for e in events if e['name']=='cycle'];cycle_union=interval_union([(e['start'],e['end'])for e in cycles]);inside=[e for e in events if e['cycle']is not None and e['name']!='cycle'];covered=interval_union([(e['start'],e['end'])for e in inside]);return {'local_union_s_by_phase':{k:interval_union(v)for k,v in by.items()},'recorded_cycle_union_s':cycle_union,'recorded_phase_union_s':covered,'unattributed_local_cycle_wall_s':max(0.,cycle_union-covered),'not_GPU_idle':True,'never_add_nested_labels':True,'crosshost_subtraction_permitted':False}
def prefill_interference(requests,prefill_calls):
 valid=[]
 for x in prefill_calls:
  a,b=x.get('micro_prefill_host_start'),x.get('micro_prefill_host_end')
  if isinstance(a,(int,float))and isinstance(b,(int,float))and b>=a:valid.append((a,b))
 total=len(prefill_calls);complete=total>0 and len(valid)==total
 coverage={'available_calls':len(valid),'expected_calls':total,'complete':complete,'status':'complete'if complete else'partial'if valid else'UNKNOWN_no_micro_prefill_timestamps'};out=[]
 for q in requests:
  visible=[e for e in q.events if not e['EOS']];first=visible[0]['time']if visible else None;last=visible[-1]['time']if visible else None
  known=interval_union([(max(a,first),min(b,last))for a,b in valid if first is not None and b>first and a<last])if first is not None and valid else None
  out.append({'logical':q.logical,'first_nonEOS_monotonic':first,'last_nonEOS_monotonic':last,'N_nonEOS':len(visible),'prefill_host_wall_overlapping_visible_firstlast_s':known if complete else None,'prefill_timing_coverage':dict(coverage),'known_partial_overlap_lower_bound_s':known if not complete else None,'actual_firstlast_unchanged':True,'overlap_is_not_GPU_prefill_busy':True})
 return out
