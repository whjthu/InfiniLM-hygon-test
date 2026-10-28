"""Pure renderer/audit contracts. No real tokenizer, weights, Native or Model."""
import copy,hashlib,importlib.util,json,pathlib,sys
from chat_prepare_r173 import prepare,encoder
from history_render_audit_r206 import audit
class StructuralTokenizer:
 def encode(self,text,add_special_tokens=False):assert add_special_tokens is False;return list(text.encode())
def run():
 reason1='SYNTHETIC_REPEAT_REASON';reason2='SYNTHETIC_DISTINCT_REASON';messages=[{'role':'system','content':'Original system.'},{'role':'developer','content':'Retain this developer.'},{'role':'user','content':'First user.'},{'role':'assistant','reasoning_content':reason1,'content':reason1},{'role':'user','content':'Second user.'},{'role':'assistant','reasoning_content':reason2,'content':'Second visible answer.'},{'role':'user','content':'Third user.'},{'role':'assistant','reasoning_content':reason1,'content':'Repeated RC is another real field.'},{'role':'user','content':'Current user.'}];before=copy.deepcopy(messages);cases=[]
 for thinking,effort,generation in[(False,None,True),(True,None,True),(True,'max',True),(False,'max',True),(False,None,False)]:
  seen=[]
  def observe(prepared,mode,eff,prompt,gen):seen.append(audit(encoder,prepared,mode,eff,prompt,gen))
  prompt,_=prepare(messages,{'_agent_choice':'none','parallel_tool_calls':False},generation,{'enable_thinking':thinking,'reasoning_effort':effort,'tools':None},StructuralTokenizer(),history_audit=observe);q=seen[0]
  assert q['qualified']and q['input_RC_fields']==q['rendered_RC_contributions']==3 and q['RC_order_preserved']and q['content_one_contribution_each_and_ordered']and q['all_content_contribution_slices_equal']and q['source_RC_SHA256_sequence']==[hashlib.sha256(v.encode()).hexdigest()for v in[reason1,reason2,reason1]]and messages==before
  if generation:assert q['pending_thinking_prefix_open']==thinking and q['pending_chat_prefix_closed']==(not thinking)
  if not thinking:assert encoder.REASONING_EFFORT_MAX not in prompt
  cases.append({'thinking':thinking,'generation':generation,'RC_contributions':3,'content_contributions':q['rendered_content_contributions'],'qualified':True})
 closed=[{'role':'system','content':'Original system.'},{'role':'user','content':'Use supplied tool.'},{'role':'assistant','content':'','reasoning_content':reason1,'tool_calls':[{'id':'fixture1','type':'function','function':{'name':'fixture_tool','arguments':{'value':1}}}]},{'role':'tool','tool_call_id':'fixture1','name':'fixture_tool','content':'{"value":1}'},{'role':'assistant','content':'','reasoning_content':reason2,'tool_calls':[{'id':'fixture2','type':'function','function':{'name':'fixture_tool','arguments':{'value':2}}}]},{'role':'tool','tool_call_id':'fixture2','name':'fixture_tool','content':'{"value":2}'}];tools=[{'type':'function','function':{'name':'fixture_tool','parameters':{'type':'object'}}}]
 for thinking in[False,True]:
  seen=[]
  def observe(prepared,mode,eff,prompt,gen):seen.append(audit(encoder,prepared,mode,eff,prompt,gen))
  prepare(closed,{'_agent_choice':'fixture_tool','parallel_tool_calls':False},True,{'enable_thinking':thinking,'tools':tools},StructuralTokenizer(),history_audit=observe);q=seen[0];assert q['qualified']and q['input_RC_fields']==q['rendered_RC_contributions']==2 and q['content_one_contribution_each_and_ordered'];cases.append({'thinking':thinking,'with_tools_and_merged_results':True,'RC_contributions':2,'content_contributions':q['rendered_content_contributions'],'qualified':True})
 return {'passed':True,'CPU_renderer_only':True,'real_tokenizer_or_Model':False,'duplicate_RC_fields_and_RC_quoted_in_visible_content':True,'ordered_dedicated_contributions_checked':True,'cases':cases}
def saved_request_contract(old):
 old=pathlib.Path(old);root=pathlib.Path('/refactor/results/Deep-agent-regular32K-r196-910374.1206/regular');assert hashlib.sha256((root/'result.json').read_bytes()).hexdigest()=='e45158f2f75a25ef86fee207668a22ec159fc69c5411bd21b5eeb13f5af76fff'
 source=old/'overlay/python/infinilm/server/agent_protocol.py';assert hashlib.sha256(source.read_bytes()).hexdigest()=='c91ab84a5ddc800602dcae0319ad5c02ad3b01c1936a2bd57bce90c2ad15abc3';sp=importlib.util.spec_from_file_location('history_cpu206_protocol',source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);results=[];sources=[]
 for serial in[8,9]:
  p=root/'thinking'/('request-%03d.request.json'%serial);body=json.loads(p.read_bytes());data,schemas,choice=m.prepare_agent_request(body,'deepseek_v4');data['_agent_choice']=choice;sources.append(hashlib.sha256(p.read_bytes()).hexdigest())
  for thinking in[False,True]:
   kw=copy.deepcopy(data['chat_template_kwargs']);kw['enable_thinking']=thinking;kw.pop('thinking_mode',None);seen=[]
   def observe(prepared,mode,eff,prompt,gen):seen.append(audit(encoder,prepared,mode,eff,prompt,gen))
   prompt,_=prepare(data['messages'],data,True,kw,StructuralTokenizer(),history_audit=observe);q=seen[0];assert q['qualified']and q['input_RC_fields']==q['rendered_RC_contributions']==serial-1 and q['content_one_contribution_each_and_ordered']and q['pending_thinking_prefix_open']==thinking and q['pending_chat_prefix_closed']==(not thinking);results.append({'thinking':thinking,'with_tools':bool(kw.get('tools')),'RC_fields':q['input_RC_fields'],'content_parts':q['input_content_parts'],'RC_and_content_contributions_ordered':True,'prompt_SHA256':q['actual_prompt_SHA256'],'chars':len(prompt)})
 assert not any(x in sys.modules for x in ['torch','transformers','_infinilm'])
 return {'passed':True,'scope':'Actual saved1206 named-tools request008 and none request009 through current pure renderer/audit; structural tokenizer only, not real token count or Model','request_SHA256':sources,'cases':results}
if __name__=='__main__':print(json.dumps(run(),separators=(',',':')))
