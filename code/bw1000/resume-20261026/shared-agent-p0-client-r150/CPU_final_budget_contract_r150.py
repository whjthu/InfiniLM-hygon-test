"""Actual1149 SSE replay stays rejected as truncated; final budget source proof."""
import pathlib,json,hashlib,base64,importlib.util,tempfile,ast
D=pathlib.Path(__file__).resolve().parent;P=D/'client_p0_r150.py';s=importlib.util.spec_from_file_location('client149',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);q=json.loads((D/'actual-Deep1149-SSE-fixture.json').read_bytes());raw={}
for name,x in q['files'].items():
 b=base64.b64decode(x['base64']);assert hashlib.sha256(b).hexdigest()==x['SHA256'];raw[name]=b
decoded=[]
for serial in range(7,12):
 message,reason,usage=m.sse_message(raw['request-%03d.wire'%serial]);disk=json.loads(raw['request-%03d.decoded.json'%serial]);assert message==disk['message']and reason==disk['finish_reason']and usage==disk['usage'];decoded.append((message,reason,usage,{'HTTP_status':200}))
assert all(x[1]=='tool_calls'for x in decoded[:4]);assert decoded[-1][1]=='length'and decoded[-1][2]=={'prompt_tokens':393,'completion_tokens':96,'total_tokens':489}
saved=json.loads(raw['sse-four-tool/result.json']);values=json.loads(raw['sse-four-tool/input.json']);assert saved=={'nonce':values['nonce'],'sum':values['a']+values['b']}
class Replay:
 def __init__(self):self.calls=[]
 def request(self,*args,**kwargs):self.calls.append(kwargs);return decoded[len(self.calls)-1]
client=Replay();rand=m.secrets.randbelow;hexfn=m.secrets.token_hex;draws=iter([values['a']-10,values['b']-10]);m.secrets.randbelow=lambda n:next(draws);m.secrets.token_hex=lambda n:values['nonce'];error=None
try:
 with tempfile.TemporaryDirectory(prefix='actual-final-budget-')as tmp:
  try:m.workflow(client,pathlib.Path(tmp),True)
  except AssertionError as e:error=repr(e)
  else:raise AssertionError('Old truncated final must remain rejected')
finally:m.secrets.randbelow=rand;m.secrets.token_hex=hexfn
assert len(client.calls)==5 and client.calls[-1]['max_tokens']==192 and all('max_tokens'not in x for x in client.calls[:-1]);assert error=='AssertionError()'
# Revert just the one bounded workload budget: AST equals exact parent.
parent=D.parent/'shared-agent-p0-client-r147/client_p0_r147.py';reverted=P.read_text().replace("message,finish,usage,e=client.request(messages,'none',stream,TOOLS,max_tokens=192);wire.append(e)","message,finish,usage,e=client.request(messages,'none',stream,TOOLS);wire.append(e)");assert ast.dump(ast.parse(reverted),include_attributes=False)==ast.dump(ast.parse(parent.read_bytes()),include_attributes=False)
out={'CPU_only':True,'GPU':False,'live_HTTP_rerun':False,'actual_step_replayed':'910374.1149','actual_SSE_four_tool_calls_valid':True,'actual_nonce_sum_file_correct':True,'actual_final_finish_reason':'length','actual_final_generated_tokens':96,'actual_final_prompt_tokens':393,'actual_final_capacity_plus_new_budget':585,'native_context_capacity':1024,'new_final_budget':192,'nonfinal_tool_budgets_unchanged':True,'actual_truncated_final_still_rejected_by_unchanged_stop_gate':True,'only_source_change':'Four-tool workflow final answer request max_tokens=192 instead of default96; no assertion or parser changes','AST_exact_parent_after_reverting_one_budget':True,'passed':True,'client_SHA256':hashlib.sha256(P.read_bytes()).hexdigest(),'qualification':'CPU evidence only; no retrospective old P0 pass and no assertion that model will EOS under192. New live fullHTTP/SSE still required.'};(D/'CPU-final-budget-contract-r150.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
