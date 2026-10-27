"""Existing decoder events only. No CUDA/Runtime/sync API added by Python."""
import os,json,ctypes as C,time
class ModelEvents:
 def __init__(self,lib,control,node,records,enabled=None,clock=time.perf_counter):
  self.lib=lib;self.control=control;self.node=node;self.records=records;self.clock=clock;self.optin=(os.environ.get('KIMI_MODEL_SPARSE_PROFILE_R135')=='1')if enabled is None else enabled;self.current=None;self.next=None
  if self.optin:
   assert os.environ.get('INFINILM_KIMI_MODEL_EVENTS_R8669ZQ')=='1','event pools must be prepared BEFORE Runtime construction'
   for name,args,restype in [('configure',[C.c_char_p],C.c_char_p),('read',[],C.c_char_p),('disable',[],C.c_int)]:
    f=getattr(lib,'modern_kimi_model_events_'+name+'_r8669zq');f.argtypes=args;f.restype=restype
 def set_next(self,label,on):
  assert type(on)is bool and isinstance(label,str)and len(label)<=100 and self.next is None
  assert not on or self.optin,'profile case requires explicit startup optin'
  self.next={'label':label,'requested':on,'verify_calls':0,'prefill_calls':0,'armed':False,'host_adapter_s':0.,'actual_forward_map':[],'verify_host_samples':[]}
 def after_original_reset(self):
  self.finish_after_original_drain('original_next_case_reset')
  self.current=self.next;self.next=None
 def before_forward(self,kind,ids,past,widths,serial):
  q=self.current
  if q is None:return
  if kind=='prefill':q['prefill_calls']+=1;return
  assert kind=='verify'
  index=q['verify_calls'];q['verify_calls']+=1
  # Arm immediately before a REAL eligible verify, avoiding nonexistent future calls
  # after early EOS. Newly configured epoch starts at that next RUN ordinal zero.
  if not(q['requested']and not q['armed']and index>=2 and sum(widths)==64):return
  t=self.clock();p={'label':q['label'],'selected_forward_ordinals':[0]};reply=json.loads(self.lib.modern_kimi_model_events_configure_r8669zq(json.dumps(p).encode()));assert reply['configured']and reply['max_calls']==3
  mapping={'case':q['label'],'actual_native_serial':serial,'actual_verify_index_zero_based':index,'worker_RUN_ordinal_in_epoch':0,'C':len(ids),'logical':list(ids),'past':list(past),'widths':list(widths),'T':sum(widths),'epoch':reply['epoch']}
  packets=self.control.collect('model-event-config:'+q['label'],mapping);assert all(x==mapping for x in packets),'allnode model-event scheduling diverged'
  q.update(armed=True,epoch=reply['epoch']);q['actual_forward_map'].append(mapping);q['host_adapter_s']+=self.clock()-t
 def after_forward(self,kind,record):
  q=self.current
  if q is not None and kind=='verify'and len(q['verify_host_samples'])<16:q['verify_host_samples'].append({'verify_index_zero_based':q['verify_calls']-1,'serial':record['serial'],'CAPI_maxhost_s':record['CAPI_maxhost_s'],'T':sum(record['width']),'C':len(record['width'])})
 def finish_after_original_drain(self,boundary):
  q=self.current
  if q is None:return
  if q['armed']:
   t=self.clock();raw=json.loads(self.lib.modern_kimi_model_events_read_r8669zq());assert raw['all8_published']and raw['epoch']==q['epoch']and len(raw['owners'])==8
   for owner in raw['owners'].values():
    assert len(owner['calls'])==1 and owner['calls'][0]['forward_ordinal']==0 and owner['read_after_original_reset_drain']
   assert self.lib.modern_kimi_model_events_disable_r8669zq()==0
   q['raw_decoder_owner_events']=raw;from summarize_decoder_events_r135 import summarize;q['decoder_summary']=summarize(raw);q['host_adapter_s']+=self.clock()-t;q['coverage']='one REAL T64 verify after >=2 warm verifies, 8localowners'
  else:q['coverage']='profile_OFF'or'no eligible T64 verify'
  if q['requested']and not q['armed']:q['coverage']='UNKNOWN: no eligible real T64 verify after two warm verifies; no event configured'
  q['publish_boundary']=boundary;q['scopes']='attention including AttnRes/norm/MLA-or-KDA/projection and TP waits; FFN including routed/shared MoE and TP waits; task/model edges aggregate'
  q['UNKNOWN_fine_scopes']=['router/dequant/GEMM/dispatch split','pure CCL','pure PP','independent head','active GPU kernel busy']
  self.records.append(q);self.current=None
 def after_normal_destroy(self):self.finish_after_original_drain('original_normal_destroy')
def overhead(before,on,after):
 assert before>0 and on>0 and after>0;baseline=(before+after)/2;ratio=on/baseline
 return {'profile_CAPI_wall_over_OFF_bracket':ratio,'perturbation_pct':100*(ratio-1),'within_3pct_diagnostic_gate':ratio<=1.03,'not_kernel_optimization':True,'clock':'local host CAPI comparison; never crosshost subtract'}
