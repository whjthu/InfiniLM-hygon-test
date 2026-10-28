"""Long-input AR using the original Native math: seed<=512, then width<=8
true input tokens plus complete-prefix commits. No repeated prefill/reset,
no drafted input, no prediction/acceptance filtering of the actual prompt.
"""
import time
from types import SimpleNamespace
from batch_policy_r9128 import greedy
CONTEXT=32768

def prefill_true_inputs(native,prompt,cancelled,deadline,clock=time.monotonic):
 if not prompt or len(prompt)>CONTEXT:raise ValueError('r179 full prompt outside32768')
 started=clock();past=0;transactions=0;status='exception';seed=list(prompt[:512]);record_start=len(getattr(native,'calls',[]))
 try:
  selected,_,_=native.call('prefill',[1],[0],[seed],False)
  past=len(seed);token=int(selected[-1])
  if getattr(native,'input_commit_progress',None):native.input_commit_progress(past,len(prompt),transactions,clock()-started)
  while past<len(prompt):
   if cancelled(-past):status='canceled';return None,past,status,transactions
   if clock()>=deadline:status='timeout';return None,past,status,transactions
   actual=list(prompt[past:past+8]);selected,_,_=native.call('verify',[1],[past],[actual],False)
   # Consume every genuine input token, independent of model predictions.
   # Intermediate selected IDs are discarded until the final input position.
   native.commit([1],[SimpleNamespace(accepted=len(actual)-1,terminal=False)])
   past+=len(actual);transactions+=1;token=int(selected[-1])
   if (past%512==0 or past==len(prompt))and getattr(native,'input_commit_progress',None):native.input_commit_progress(past,len(prompt),transactions,clock()-started)
  status='complete';return token,past,None,transactions
 finally:
  records=getattr(native,'calls',[])[record_start:]
  observation={'prompt_tokens':len(prompt),'initial_seed_tokens':len(seed),'input_tokens_committed':past,'true_input_width_max':8,'width8_verify_fullcommit_count':transactions,'full_prompt_consumed':past==len(prompt),'status':status,'prefill_wall_s':max(0.,clock()-started),'actual_forward_records':len(records),'recorded_completed_forward_CAPI_maxhost_s_sum':sum(float(r['CAPI_maxhost_s'])for r in records)if records and all(isinstance(r.get('CAPI_maxhost_s'),(int,float))for r in records)else None,'numeric_equivalence_qualified':False}
  if not hasattr(native,'long_prefill_observations'):native.long_prefill_observations=[]
  native.long_prefill_observations.append(observation)

def generate(native,prompt,cap,eos,emit,cancelled,deadline,clock=time.monotonic):
 if not(1<=cap<=8192 and prompt and len(prompt)+cap<=CONTEXT):raise ValueError('r179 complete prompt+unchanged output budget outside32768')
 token,past,ended,_=prefill_true_inputs(native,prompt,cancelled,deadline,clock)
 if ended:return [],ended
 generated=[token];emit(token)
 while token not in eos and len(generated)<cap:
  if cancelled(len(generated)):return generated,'canceled'
  if clock()>=deadline:return generated,'timeout'
  selected,_,_=native.call('verify',[1],[past],[[token]],False)
  decision=greedy(token,[],selected,eos,cap-len(generated))
  native.commit([1],[decision])
  token=int(decision.emitted[0]);generated.append(token);emit(token)
  if not decision.terminal:past+=len(decision.commit_inputs)
 return generated,'eos'if token in eos else'length'


def generate_scoped_request(native,*args,**kwargs):
 """For a new service caller without an outer reset/drain owner. Any caller
 still owns final engine destroy on a fatal Native/reset error; no retry.
 Existing target_protocol already has its four-node finally reset/drain and
 should call generate directly to retain that collective cleanup boundary.
 """
 native.reset(False)
 try:return generate(native,*args,**kwargs)
 finally:native.reset(False)
