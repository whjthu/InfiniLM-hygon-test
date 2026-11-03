"""Frozen six-request Model experiment. No public request may select its mode."""
from contextlib import contextmanager
from quiescent_mode import QuiescentMode
MODES=(0,1,0,1,1,0)
PROMPT_SHA='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
class RequestPolicy:
 def __init__(self,context,config,*,provider_validator=None):
  from .native_binding import validate_mode_functions
  assert config['r438_model_modes']==list(MODES)and config['r438_output_budget']==128
  self.control=QuiescentMode(context);self.index=0;self.layers=int(context.cfg['num_hidden_layers']);assert self.layers==43
  self.binding_proof=(validate_mode_functions if provider_validator is None else provider_validator)(self.control.lib)
 @property
 def poisoned(self):return self.control.poisoned
 def forward(self,*args):return self.control.forward(*args)
 @contextmanager
 def request(self,req,record):
  i=self.index
  if i>=len(MODES):raise RuntimeError('Frozen six-request Model plan exhausted; no extra request')
  if req.max_new_tokens!=128 or len(req.prompt_token_ids)!=8489:raise ValueError('Unchanged full8489 input/128 output required')
  audit=record['admission']['history_render_audit']
  if audit['Native_prompt_token_ids_SHA256']!=PROMPT_SHA or record['admission']['text_delivery']!='P1':raise ValueError('Frozen history or fixed incremental delivery changed')
  if audit['pending_thinking_prefix_open']:raise ValueError('This pair freezes ordinary chat, not thinking budget')
  self.index+=1;mode=MODES[i];record.update(r438_index=i,r438_mode=mode,r438_warm=i<2,r438_binding_proof=self.binding_proof)
  try:
   with self.control.request(mode):
    record['normal_reset_before']=True
    yield self
  finally:
   record['r438_quiescent']=self.control.quiescent;record['r438_poisoned']=self.poisoned;record['r438_actual_owner_counts']=self.control.last_counts
  counts=self.control.last_counts
  if counts is None or len(counts)!=8:raise RuntimeError('Missing real eight-owner Model counters')
  trace=record['Native_continuation'];one_token_prefill=sum(t['tokens']==1 for t in trace if t['kind']=='prefill')
  decode=sum(t['calls']for t in trace if t['kind']=='decode_summary');t1=self.layers*(one_token_prefill+decode)
  forwards=record['actual_forward_completed'];entered=self.layers*forwards
  if t1<=0 or forwards!=record['actual_forward_attempts']:raise RuntimeError('Incomplete original Native continuation')
  for c in counts:
   if not all(type(v)is int and v>=0 for v in c.values()):raise RuntimeError('Invalid typed owner counters')
   if not(c['entered_valid_MoE']==c['host_returned_not_GPU_completed']==entered and c['T1']==t1 and c['unwound']==0 and c['last_mode']==mode and c['armed_mode1_T1']==mode*t1):raise RuntimeError('Actual factory/T1 path coverage differs')
   for prefix in ['routed','shared']:
    if c[prefix+'_hits']!=mode*t1 or c[prefix+'_calls']!=c[prefix+'_hits']+c[prefix+'_fallback']:raise RuntimeError('Actual fused Model path missing or counter accounting differs')
  record['r438_actual_T1_coverage']=dict(passed=True,MoE_layers=self.layers,completed_forwards=forwards,one_token_prefill_forwards=one_token_prefill,decode_forwards=decode,T1_per_owner=t1,mode=mode)
