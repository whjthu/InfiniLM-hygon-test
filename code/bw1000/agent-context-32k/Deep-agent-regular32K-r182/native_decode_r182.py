"""Owner-thread continuation; only existing Native forward is called."""
def generate(req,forward,check_rc,finite_check,reasons,context=32768,chunk_size=1024,eos=1,vocab=129280):
 ids=list(req.prompt_token_ids);budget=req.max_new_tokens
 if type(budget)is not int or budget<1:raise ValueError('Native budget must be the original positive JSON integer')
 if not ids or len(ids)+budget>context:raise ValueError('Full prompt plus budget exceeds actual Native context; no crop')
 if not 1<=chunk_size<=4096:raise ValueError('Native per-forward token cap4096 is preserved')
 trace=[];past=0;token=None;decode_summary=None
 def sample(value):
  token=int(value[0])
  if not 0<=token<vocab:raise RuntimeError('Native sampled token outside actual vocabulary')
  return token
 for offset in range(0,len(ids),chunk_size):
  if req.cancel_event.is_set():return req.abort_reason or reasons.CANCELED,trace
  part=ids[offset:offset+chunk_size];code,logits,sampled=forward(part,list(range(past,past+len(part))),[0],[past],[past+len(part)],[0,len(part)]);check_rc(code);finite_check(logits)
  trace.append({'kind':'prefill','past':past,'tokens':len(part),'total':past+len(part),'emitted':False});past+=len(part);token=sample(sampled)
 # Every input token has been consumed exactly once, with no reset. Discard
 # intermediate chunk samples; only the final prefill sample enters AR output.
 assert past==len(ids)and token is not None
 if req.cancel_event.is_set():return req.abort_reason or reasons.CANCELED,trace
 req.emit_token(token);trace[-1]['emitted']=True;generated=1
 if token==eos:return reasons.EOS_TOKEN,trace
 while generated<budget:
  if req.cancel_event.is_set():return req.abort_reason or reasons.CANCELED,trace
  if past+1>context:raise RuntimeError('Unexpected Native context overflow')
  code,logits,sampled=forward([token],[past],[0],[past],[past+1],[0,1]);check_rc(code);finite_check(logits)
  if decode_summary is None:decode_summary={'kind':'decode_summary','calls':0,'first_past':past,'last_total':past};trace.append(decode_summary)
  decode_summary['calls']+=1;decode_summary['last_total']=past+1;past+=1;token=sample(sampled);req.emit_token(token);generated+=1
  if token==eos:return reasons.EOS_TOKEN,trace
 return reasons.LENGTH,trace
