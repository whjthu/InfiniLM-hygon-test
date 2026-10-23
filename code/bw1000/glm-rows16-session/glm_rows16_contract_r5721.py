"""CPU-only reference contract for two separate, explicitly named fixtures."""
import hashlib,json
I512_SHA256='2f1076c4933394579097fc58fb399b53667f11cbce81a06d263aa5b4e2d83bbc'
def validate_fixture(q):
 assert q['completed'] and q['padding_requested'] is False and q['CUDA_initialized'] is False
 assert len(q['cases'])==16 and q['total_effective_input_tokens']==8192
 p=[c['input_ids']for c in q['cases']]
 assert len(set(map(tuple,p)))==16 and all(len(v)==512 and all(type(t)is int and 0<=t<154880 for t in v)for v in p)
 assert [c['logical_row']for c in q['cases']]==list(range(16))
 assert q['capacity']==640 and q['max_output_tokens']==128
 for c in q['cases']:assert hashlib.sha256(json.dumps(c['input_ids'],separators=(',',':')).encode()).hexdigest()==c['input_ids_SHA256']
 return p
def validate_reference(kind,arm,prompts,generated,stopped,limit,short_gold,i512_inputs,ledger):
 assert kind in ('short','I512','qa') and arm in ('O','R')
 assert len(prompts)==len(generated)==len(stopped)==16 and limit>0
 if kind=='qa':return True,[] # Same-logical-row repeats and known answers checked separately.
 expected=short_gold['input_ids']if kind=='short'else i512_inputs
 if prompts!=expected:return False,[{'fixture_identity_mismatch':kind}]
 if kind=='short':
  want=[g[:limit]for g in short_gold['generated_ids']]
  ok=generated==want and (limit!=128 or stopped==short_gold['stopped_at_EOS'])
  return ok,[]if ok else[{'historical_short_full_ID_or_EOS_mismatch':True}]
 assert limit==128 and all(len(p)==512 for p in prompts)
 key=('I512',hashlib.sha256(json.dumps(prompts,separators=(',',':')).encode()).hexdigest(),limit)
 if key not in ledger:
  if arm!='O':return False,[{'missing_same_input_OFF_baseline':True}]
  ledger[key]={'generated_ids':[list(g)for g in generated],'stopped_at_EOS':list(stopped)}
  return True,[]
 q=ledger[key];ok=generated==q['generated_ids']and stopped==q['stopped_at_EOS']
 return ok,[]if ok else[{'same_input_OFF_full_ID_or_EOS_mismatch':True,'scope':'regression reference, not independent original mathematical proof'}]
