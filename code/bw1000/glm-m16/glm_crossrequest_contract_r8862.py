"""CPU-only reference contract for two separate, explicitly named fixtures."""
import hashlib,json
I512_SHA256='edfad62bc39efdccf133b4d95ac3b07a44fe76356e013881e67926e017c326fc'
def validate_fixture(q):
 assert q['completed'] and q['padding_requested'] is False and q['CUDA_initialized'] is False
 assert len(q['cases'])==32 and q['total_effective_input_tokens']==16384
 p=[c['input_ids']for c in q['cases']]
 assert len(set(map(tuple,p)))==32 and all(len(v)==512 and all(type(t)is int and 0<=t<154880 for t in v)for v in p)
 assert [c['logical_row']for c in q['cases']]==list(range(32))
 assert q['capacity']==640 and q['max_output_tokens']==128
 for c in q['cases']:assert hashlib.sha256(json.dumps(c['input_ids'],separators=(',',':')).encode()).hexdigest()==c['input_ids_SHA256']
 return p
def validate_reference(kind,arm,prompts,generated,stopped,limit,short_gold,i512_inputs,ledger,historical_i512=None):
 assert kind in ('short','I512','qa') and arm in ('O','R')
 assert len(prompts)==len(generated)==len(stopped) and len(prompts)in(16,32) and limit>0
 if kind=='qa':return True,[] # Same-logical-row repeats and known answers checked separately.
 expected=short_gold['input_ids']if kind=='short'else i512_inputs[:len(prompts)]
 if prompts!=expected:return False,[{'fixture_identity_mismatch':kind}]
 if kind=='short':
  want=[g[:limit]for g in short_gold['generated_ids']]
  ok=generated==want and (limit!=128 or stopped==short_gold['stopped_at_EOS'])
  return ok,[]if ok else[{'historical_short_full_ID_or_EOS_mismatch':True}]
 assert limit in(4,128)and all(len(p)==512 for p in prompts)
 assert historical_i512 is not None,'Frozen477fullgoldrequired_no_rebaseline'
 want=[g[:limit]for g in historical_i512['generated_ids'][:len(prompts)]]
 eos=historical_i512['stopped_at_EOS'][:len(prompts)]if limit==128 else[False]*len(prompts)
 ok=generated==want and stopped==eos
 return ok,[]if ok else[{'477_original_full_ID_or_EOS_mismatch':True}]
