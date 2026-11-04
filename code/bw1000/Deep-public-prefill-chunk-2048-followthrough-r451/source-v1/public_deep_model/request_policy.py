"""Six internal chunks, fullpre and r366 stay zero; no public arming field."""
from contextlib import contextmanager
from .fullpre_control import QuiescentMode
from .chunk_contract import CHUNKS,descriptor_for_index
MODES=(0,)*6
PROMPT_SHA='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
class RequestPolicy:
 @property
 def chunk_size(self):
  if self.current_binding is None or not self.control.active:raise RuntimeError('Chunk descriptor accessed outside original admitted request')
  return self.current_binding['prefill_chunk']
 def __init__(self,context,config,*,provider_validator=None):
  from .native_binding import validate_mode_functions
  assert config['r451_chunks']==list(CHUNKS)and config['r451_output_budget']==128 and config['r451_r366_fixed']==config['r451_fullpre_fixed']==0
  from .native_binding import binding
  self.binding_descriptor=binding()[0];self.current_binding=None
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
  self.index+=1;mode=0;self.current_binding=descriptor_for_index(self.binding_descriptor,i)
  record.update(r451_index=i,r451_mode=0,r451_chunk=CHUNKS[i],r451_warm=i<2,r451_r366_fixed=0,r451_fullpre_fixed=0,r451_binding_proof=self.binding_proof,prefill_binding=self.current_binding)
  try:
   with self.control.request(mode):record['normal_reset_before']=True;yield self
  finally:
   record.update(r451_quiescent=self.control.quiescent,r451_poisoned=self.poisoned,r451_actual_owner_counts=self.control.last_counts)
  counts=self.control.last_counts
  if counts is None or len(counts)!=8:raise RuntimeError('Missing actual8 HC counters')
  trace=record['Native_continuation'];one_prefill=sum(t['tokens']==1 for t in trace if t['kind']=='prefill');decode=sum(t['calls']for t in trace if t['kind']=='decode_summary')
  forwards=record['actual_forward_completed'];t1=one_prefill+decode
  if t1<0 or forwards!=record['actual_forward_attempts']or(t1==0 and not req.cancel_event.is_set()):raise RuntimeError('Incomplete original continuation without a cooperative cancel intent')
  for owner,c in enumerate(counts):
   a,b=c['legacy'],c['fullpre']
   if c['owner']!=owner or c['created']!=86 or c['ready']!=86:raise RuntimeError('Actual86 HC constructors/ready missing')
   if not all(type(v)is int and v>=0 for v in [*a.values(),*b.values()]):raise RuntimeError('Invalid typed HC counters')
   if not(a['created']==a['ready']==a['completed']==a['f32_t1']==0 and a['pre']==86*forwards and a['eligible_bf16_t1']==86*t1 and a['other_t']==86*(forwards-t1)):raise RuntimeError('Actual HC0/86groups forward coverage differs')
   expected=dict(observed=mode*86*forwards,eligible=mode*86*t1,selected=mode*86*t1,completed=mode*86*t1,prefill=mode*86*(forwards-t1),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
   if b!=expected:raise RuntimeError('Actual fullpre select/completion/fallback differs')
  record['r451_actual_T1_coverage']=dict(passed=True,HC_groups=86,completed_forwards=forwards,one_token_prefill_forwards=one_prefill,decode_forwards=decode,T1_HC_per_owner=86*t1,mode=mode,r366_fixed=0)
