"""Finite wide-quality admission, reusing immutable449 genuine reset/drain control."""
from contextlib import contextmanager
from public_deep_model.fullpre_control import QuiescentMode
from paired_plan import validate_subset
from coverage import validate

class RequestPolicy:
 def __init__(self,context,config,*,provider_validator=None):
  from public_deep_model.native_binding import validate_mode_functions
  assert config['HC_mode']==0 and config['r450_r366_fixed']==0
  self.rows=validate_subset(config['r450_plan'],config['r450_frozen512']);self.index=0;self.control=QuiescentMode(context)
  self.binding_proof=(provider_validator or validate_mode_functions)(self.control.lib)
 @property
 def poisoned(self):return self.control.poisoned
 @property
 def normal_reset_after(self):return self.control.normal_reset_after
 def forward(self,*args):return self.control.forward(*args)
 @contextmanager
 def request(self,req,record):
  if self.index>=len(self.rows):raise RuntimeError('Finite wide plan exhausted; no extra request')
  entry=self.rows[self.index];audit=record['admission']['history_render_audit']
  if req.max_new_tokens!=4096 or len(req.prompt_token_ids)!=entry['prompt_tokens']or audit['Native_prompt_token_ids_SHA256']!=entry['prompt_IDs_SHA256']:raise ValueError('Frozen complete fewshot input/budget differs')
  if record['admission']['diagnostic_user']!='Deep-r450-'+entry['case_id']or record['admission']['text_delivery']!='P1'or audit['pending_thinking_prefix_open']:raise ValueError('Frozen case/user/nonthinking/P1 differs')
  self.index+=1;record.update(r450_logical_index=entry['logical_index'],r450_case_id=entry['case_id'],r450_mode=entry['mode'],r450_r366_fixed=0,r450_binding_proof=self.binding_proof)
  try:
   with self.control.request(entry['mode']):record['normal_reset_before']=True;yield self
  finally:record.update(normal_reset_after=self.normal_reset_after,r450_quiescent=self.control.quiescent,r450_poisoned=self.poisoned,r450_actual_owner_counts=self.control.last_counts)
  # Zero eligible T1 is a legitimate short/empty answer, retained in quality denominator.
  record['generated_token_ids']=list(req.generated_token_ids)
  record['r450_actual_T1_coverage']=validate(record,entry)
