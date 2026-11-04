"""Fixed startup mode across unlimited admitted C1 requests, with real HC coverage."""
import threading
from contextlib import contextmanager
from .fullpre_control import QuiescentMode
from .profile import mode,NAME
class RequestPolicy:
 def __init__(self,context,config,*,provider_validator=None):
  from .native_binding import validate_mode_functions
  if (config['TP'],config['PP'],config['rows'],config['context'],config['HC_mode'])!=(8,1,1,32768,0):raise ValueError('Fixed qualified HC0/C1/TP8/32K contract required')
  self._mode=mode(config);self.control=QuiescentMode(context);self.index=0;self.groups=2*int(context.cfg['num_hidden_layers']);assert self.groups==86
  self.owner=threading.get_ident();self.closed=False
  self.binding_proof=(validate_mode_functions if provider_validator is None else provider_validator)(self.control.lib)
  self._arm_quiescent()
 @property
 def poisoned(self):return self.control.poisoned
 @property
 def normal_reset_after(self):return self.control.normal_reset_after
 def _arm_quiescent(self):
  if self.owner!=threading.get_ident()or self.closed:raise RuntimeError('Profile changes require the actual Native owner')
  c=self.control;c.reset_and_drain();c.check(c.lib.deep_hc_fullpre_enable_r377(0));c.check(c.lib.deep_hc_model_epilogue_set_r366(0));c.check(c.lib.deep_hc_fullpre_clear_r377());c.check(c.lib.deep_hc_fullpre_enable_r377(self._mode))
  if c.lib.deep_hc_fullpre_enabled_r377()!=self._mode or c.lib.deep_hc_model_epilogue_get_r366()!=0:raise RuntimeError('Actual Native fixed profile readback differs')
 def ready(self):
  c=self.control
  if self.closed or self.poisoned or not c.quiescent or c.lib.deep_hc_fullpre_enabled_r377()!=self._mode or c.lib.deep_hc_model_epilogue_get_r366()!=0:raise RuntimeError('Profile is not quiescent/armed at READY')
  return dict(profile=NAME if self._mode else None,fixed_mode=self._mode,HC_mode=0,r366_fixed=0,C1=True,HC_groups=86,Native_mode_readback=self._mode,provider=self.binding_proof)
 def close(self):
  if self.closed:return
  if self.owner!=threading.get_ident():raise RuntimeError('Profile close is owner-thread only')
  if self.poisoned:
   self.closed=True;return # Never retry a failed reset; original Context destroy owns containment.
  c=self.control;c.reset_and_drain();c.check(c.lib.deep_hc_fullpre_enable_r377(0));c.check(c.lib.deep_hc_fullpre_clear_r377());self.closed=True
 def forward(self,*args):return self.control.forward(*args)
 @contextmanager
 def request(self,req,record):
  if self.closed:raise RuntimeError('Closed fixed profile')
  if not req.prompt_token_ids or type(req.max_new_tokens)is not int or not 1<=req.max_new_tokens<=8192 or len(req.prompt_token_ids)+req.max_new_tokens>32768:raise ValueError('Original public complete-context/output bounds required')
  i=self.index;self.index+=1;mode=self._mode
  record.update(r454_index=i,r454_mode=mode,r454_profile=NAME if mode else None,r454_r366_fixed=0,r454_binding_proof=self.binding_proof)
  primary=None
  try:
   with self.control.request(mode):record['normal_reset_before']=True;yield self
  except BaseException as error:primary=error;raise
  finally:
   record.update(r454_quiescent=self.control.quiescent,r454_poisoned=self.poisoned,r454_actual_owner_counts=self.control.last_counts)
   if not self.poisoned:
    # request() already performed the original reset/drain; no additional GPU
    # wait is introduced just to restore this engine's immutable fixed mode.
    try:
     if not self.control.quiescent:raise RuntimeError('Cannot restore profile before real completion')
     self.control.check(self.control.lib.deep_hc_fullpre_enable_r377(mode))
     if self.control.lib.deep_hc_fullpre_enabled_r377()!=mode:raise RuntimeError('Fixed mode readback changed')
    except BaseException as error:
     self.control.poisoned=True
     if primary is None:raise
     if hasattr(primary,'add_note'):primary.add_note('Fixed profile restore: '+repr(error))
  counts=self.control.last_counts
  if counts is None or len(counts)!=8:raise RuntimeError('Missing actual8 HC counters')
  trace=record['Native_continuation'];one_prefill=sum(t['tokens']==1 for t in trace if t['kind']=='prefill');decode=sum(t['calls']for t in trace if t['kind']=='decode_summary')
  forwards=record['actual_forward_completed'];t1=one_prefill+decode
  if t1<0 or forwards!=record['actual_forward_attempts']:raise RuntimeError('Incomplete original continuation')
  for owner,c in enumerate(counts):
   a,b=c['legacy'],c['fullpre']
   if c['owner']!=owner or c['created']!=86 or c['ready']!=86:raise RuntimeError('Actual86 HC constructors/ready missing')
   if not all(type(v)is int and v>=0 for v in [*a.values(),*b.values()]):raise RuntimeError('Invalid typed HC counters')
   if not(a['created']==a['ready']==a['completed']==a['f32_t1']==0 and a['pre']==86*forwards and a['eligible_bf16_t1']==86*t1 and a['other_t']==86*(forwards-t1)):raise RuntimeError('Actual HC0/86groups forward coverage differs')
   expected=dict(observed=mode*86*forwards,eligible=mode*86*t1,selected=mode*86*t1,completed=mode*86*t1,prefill=mode*86*(forwards-t1),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
   if b!=expected:raise RuntimeError('Actual fullpre select/completion/fallback differs')
  record['r454_actual_T1_coverage']=dict(passed=True,HC_groups=86,completed_forwards=forwards,one_token_prefill_forwards=one_prefill,decode_forwards=decode,T1_HC_per_owner=86*t1,mode=mode,r366_fixed=0)
