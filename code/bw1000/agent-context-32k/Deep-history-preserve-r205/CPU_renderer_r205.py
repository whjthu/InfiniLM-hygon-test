"""Synthetic renderer-only history/generation separation proof; no Model."""
import argparse,copy,hashlib,importlib.util,json,pathlib
D=pathlib.Path(__file__).resolve().parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class StructuralTokenizer:
 def encode(self,text,add_special_tokens=False):assert add_special_tokens is False;return list(text.encode())
def run(old_dir):
 old=load('old_prepare_history_r205',pathlib.Path(old_dir)/'chat_prepare_r173.py');new=load('new_prepare_history_r205',D/'chat_prepare_r173.py');tools=[{'type':'function','function':{'name':'fixture_tool','parameters':{'type':'object'}}}]
 R1='SYNTHETIC_REASON_ONE';R2='SYNTHETIC_REASON_TWO';DEV='SYNTHETIC_DEVELOPER';U1='SYNTHETIC_USER_ONE';U2='SYNTHETIC_USER_TWO';C1='SYNTHETIC_VISIBLE_ONE';TR='SYNTHETIC_TOOL_RESULT'
 ordinary=[{'role':'system','content':'Synthetic system.'},{'role':'developer','content':DEV},{'role':'user','content':U1},{'role':'assistant','content':C1,'reasoning_content':R1},{'role':'user','content':U2}]
 closed=[{'role':'system','content':'Synthetic system.'},{'role':'user','content':U1},{'role':'assistant','content':'','reasoning_content':R1,'tool_calls':[{'id':'fixture-call1','type':'function','function':{'name':'fixture_tool','arguments':{}}}]},{'role':'tool','tool_call_id':'fixture-call1','name':'fixture_tool','content':TR}]
 closed2=closed+[{'role':'assistant','content':'','reasoning_content':R2,'tool_calls':[{'id':'fixture-call2','type':'function','function':{'name':'fixture_tool','arguments':{}}}]},{'role':'tool','tool_call_id':'fixture-call2','name':'fixture_tool','content':'SYNTHETIC_TOOL_RESULT_TWO'},{'role':'user','content':U2}]
 developer=[{'role':'system','content':'Synthetic system.'},{'role':'user','content':U1},{'role':'assistant','content':C1},{'role':'developer','content':DEV},{'role':'assistant','content':'Synthetic second answer.','reasoning_content':R2},{'role':'user','content':U2}]
 no_rc=[{'role':'system','content':'Synthetic system.'},{'role':'user','content':U1},{'role':'assistant','content':C1},{'role':'user','content':U2}]
 rows=[]
 def check(label,messages,thinking,available,choice='auto',extra=None,generation=True,expect_same=False):
  data={'_agent_choice':choice,'parallel_tool_calls':False};kw={'enable_thinking':thinking,'tools':tools if available else None};kw.update(extra or{});before=copy.deepcopy(messages);observed=[];fn=new.encoder.encode_messages
  def capture(prepared,**kwargs):observed.append((copy.deepcopy(prepared),copy.deepcopy(kwargs)));return fn(prepared,**kwargs)
  new.encoder.encode_messages=capture
  try:prompt,_=new.prepare(messages,data,generation,kw,StructuralTokenizer())
  finally:new.encoder.encode_messages=fn
  previous,_=old.prepare(messages,data,generation,kw,StructuralTokenizer());assert messages==before and len(observed)==1
  prepared,forwarded=observed[0];assert forwarded['drop_thinking']is False and forwarded['preserve_history_reasoning']is True and forwarded['thinking_mode']==('thinking'if thinking else'chat')and forwarded.get('reasoning_effort')==kw.get('reasoning_effort')
  rc=[m['reasoning_content']for m in messages if m.get('reasoning_content')];assert all(x in prompt for x in rc)
  content=[m['content']for m in messages if isinstance(m.get('content'),str)and m['content']];assert all(x in prompt for x in content)
  for reason in rc:assert new.encoder.ASSISTANT_SP_TOKEN+'<think>'+reason+'</think>'in prompt,'Historical reasoning must have its own correct open/close markers'
  merged=new.encoder.merge_tool_messages(copy.deepcopy(prepared));has_pending=bool(merged and merged[-1].get('role')in ['user','developer']and merged[-1].get('task')is None)
  expected=new.encoder.ASSISTANT_SP_TOKEN+('<think>'if thinking else'</think>')
  if has_pending:assert prompt.endswith(expected)if generation else not prompt.endswith(expected)
  if kw.get('reasoning_effort')=='max':assert(new.encoder.REASONING_EFFORT_MAX in prompt)==thinking
  # New optional flag defaults to legacy behavior, byte-for-byte.
  legacy_kwargs={'thinking_mode':'thinking'if thinking else'chat','reasoning_effort':kw.get('reasoning_effort')};assert old.encoder.encode_messages(prepared,**legacy_kwargs)==new.encoder.encode_messages(prepared,**legacy_kwargs)
  if expect_same:assert prompt==previous
  rows.append({'case':label,'historical_reasoning_fields':len(rc),'all_input_text_and_reasoning_present':True,'caller_unchanged':True,'historical_RC_open_close_correct':True,'current_generation_thinking':thinking,'pending_prefix_checked':has_pending,'current_generation_prefix_correct':True,'effort_unchanged':True,'legacy_default_renderer_byte_equivalent':True,'private_old_new_byte_equivalent':prompt==previous,'prompt_chars':len(prompt),'prompt_SHA256':hashlib.sha256(prompt.encode()).hexdigest()})
 check('multiuser_auto_no_tools_thinking',ordinary,True,False)
 check('multiuser_none_no_tools_thinking',ordinary,True,False,'none')
 check('multiuser_explicit_drop_false',ordinary,True,False,extra={'drop_thinking':False})
 check('multiuser_explicit_clear_false',ordinary,True,False,extra={'clear_thinking':False})
 check('multiuser_tools_thinking_unchanged',ordinary,True,True,expect_same=True)
 check('multiuser_auto_no_tools_chat',ordinary,False,False)
 check('multiuser_auto_tools_chat',ordinary,False,True)
 check('closed_tool_none_thinking',closed,True,False,'none')
 check('closed_tool_named_thinking_unchanged',closed,True,True,'fixture_tool',expect_same=True)
 check('closed_tool_named_chat',closed,False,True,'fixture_tool')
 check('assistant_after_last_user_chat',ordinary+[{'role':'assistant','content':'Synthetic current answer.','reasoning_content':R2}],False,False)
 check('developer_to_historical_assistant_chat',developer,False,False)
 check('two_closed_tools_then_new_user_chat',closed2,False,False,'none')
 check('current_chat_max_does_not_enable_thinking',ordinary,False,False,extra={'reasoning_effort':'max'})
 check('current_thinking_max_kept',ordinary,True,False,extra={'reasoning_effort':'max'})
 check('current_chat_high_kept',ordinary,False,False,extra={'reasoning_effort':'high'})
 check('no_generation_prompt_keeps_history',ordinary,False,False,generation=False)
 check('chat_without_RC_byte_unchanged',no_rc,False,False,expect_same=True)
 check('chat_tools_without_RC_byte_unchanged',no_rc,False,True,expect_same=True)
 check('single_user_thinking_byte_unchanged',[{'role':'user','content':U1}],True,False,expect_same=True)
 return {'passed':True,'scope':'20 synthetic renderer-only cases; no actual tokenizer or Model claim','cases':rows,'no_blind_tail_replacement':True,'Model_called':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old-dir',required=True);x=p.parse_args();print(json.dumps(run(pathlib.Path(x.old_dir)),separators=(',',':')))
