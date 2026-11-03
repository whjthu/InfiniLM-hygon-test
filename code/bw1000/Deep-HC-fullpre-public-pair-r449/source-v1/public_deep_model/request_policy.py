"""Six internal fullpre modes; no public input can choose or arm the policy."""
from contextlib import contextmanager
from .fullpre_control import QuiescentMode
MODES=(0,1,0,1,1,0)
PROMPT_SHA='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
class RequestPolicy:
 def __init__(self,context,config,*,provider_validator=None):
  from .native_binding import validate_mode_functions
  assert config['r449_fullpre_modes']==list(MODES)and config['r449_output_budget']==128 and config['r449_r366_fixed']==0
  self.control=QuiescentMode(context);self.index=0;self.groups=2*int(context.cfg['num_hidden_layers']);assert self.groups==86
  self.binding_proof=(validate_mode_functions if provider_validator is None else provider_validator)(self.control.lib)
 @property
 def poisoned(self):return self.control.poisoned
 @property
 def normal_reset_after(self):return self.control.normal_reset_after
 def forward(self,*args):return self.control.forward(*args)
 @contextmanager
 def request(self,req,record):
  i=self.index
  if i>=len(MODES):raise RuntimeError('Frozen six fullpre requests exhausted; no seventh')
  if req.max_new_tokens!=128 or len(req.prompt_token_ids)!=8489:raise ValueError('Frozen8489 input/128 budget required')
  audit=record['admission']['history_render_audit']
  if audit['Native_prompt_token_ids_SHA256']!=PROMPT_SHA or record['admission']['text_delivery']!='P1'or audit['pending_thinking_prefix_open']:raise ValueError('Frozen full history/nonthinking/P1 differs')
  self.index+=1;mode=MODES[i];record.update(r449_index=i,r449_mode=mode,r449_warm=i<2,r449_r366_fixed=0,r449_binding_proof=self.binding_proof)
  try:
   with self.control.request(mode):record['normal_reset_before']=True;yield self
  finally:
   record.update(r449_quiescent=self.control.quiescent,r449_poisoned=self.poisoned,r449_actual_owner_counts=self.control.last_counts)
  counts=self.control.last_counts
  if counts is None or len(counts)!=8:raise RuntimeError('Missing actual8 HC counters')
  trace=record['Native_continuation'];one_prefill=sum(t['tokens']==1 for t in trace if t['kind']=='prefill');decode=sum(t['calls']for t in trace if t['kind']=='decode_summary')
  forwards=record['actual_forward_completed'];t1=one_prefill+decode
  if t1<=0 or forwards!=record['actual_forward_attempts']:raise RuntimeError('Incomplete original continuation')
  for owner,c in enumerate(counts):
   a,b=c['legacy'],c['fullpre']
   if c['owner']!=owner or c['created']!=86 or c['ready']!=86:raise RuntimeError('Actual86 HC constructors/ready missing')
   if not all(type(v)is int and v>=0 for v in [*a.values(),*b.values()]):raise RuntimeError('Invalid typed HC counters')
   if not(a['created']==a['ready']==a['completed']==a['f32_t1']==0 and a['pre']==86*forwards and a['eligible_bf16_t1']==86*t1 and a['other_t']==86*(forwards-t1)):raise RuntimeError('Actual HC0/86groups forward coverage differs')
   expected=dict(observed=mode*86*forwards,eligible=mode*86*t1,selected=mode*86*t1,completed=mode*86*t1,prefill=mode*86*(forwards-t1),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
   if b!=expected:raise RuntimeError('Actual fullpre select/completion/fallback differs')
  record['r449_actual_T1_coverage']=dict(passed=True,HC_groups=86,completed_forwards=forwards,one_token_prefill_forwards=one_prefill,decode_forwards=decode,T1_HC_per_owner=86*t1,mode=mode,r366_fixed=0)
