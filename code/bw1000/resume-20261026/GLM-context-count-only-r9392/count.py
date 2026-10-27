"""CPU tokenizer capacity estimate only: synthetic history is NOT model output/P0.

Loads no model weights. Run inside DTK CPU container with /old/offline-deps.
Uses exact client tool schemas and request text; illustrative complete history
estimates the budget before real generation, whose lengths may differ.
"""
import hashlib,importlib.util,json,pathlib,sys
from transformers import PreTrainedTokenizerFast
R=pathlib.Path('/refactor');D=R/'incoming/GLM-agent-service-r143b'
P=D/'overlay/python/infinilm/server/agent_protocol.py'
assert hashlib.sha256(P.read_bytes()).hexdigest()=='21a4d833eda5815c1feeba1a1e0ed5c9976fd2dcd5a65af11b7d4197a7ad2907'
s=importlib.util.spec_from_file_location('count_only_agent_protocol',P);protocol=importlib.util.module_from_spec(s);sys.modules[s.name]=protocol;s.loader.exec_module(protocol)
TOOLS=[
 {'type':'function','function':{'name':'read_file','description':'Read file.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'calculate','description':'Add integers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'write_file','description':'Write file.','parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content'],'additionalProperties':False}}},
]
messages=[{'role':'system','content':'Use tools; never invent their results.'},{'role':'user','content':'Read input.json, calculate a+b, write result.json with nonce and sum, read result.json, then report nonce and sum.'}]
tokenizer=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True)
steps=[('read_file',{'path':'input.json'},{'a':73,'b':82,'nonce':'31af75ce'}),('calculate',{'a':73,'b':82},{'sum':155}),('write_file',{'path':'result.json','content':'{"nonce":"31af75ce","sum":155}'},{'ok':True}),('read_file',{'path':'result.json'},{'nonce':'31af75ce','sum':155})]
checks=[]
for i in range(5):
 name=steps[i][0]if i<4 else None
 data={'messages':messages,'tools':TOOLS,'tool_choice':{'type':'function','function':{'name':name}}if name else'none','parallel_tool_calls':False,'max_tokens':96,'chat_template_kwargs':{'enable_thinking':False}}
 prepared,_,_=protocol.prepare_agent_request(data,'glm5');prompt=tokenizer.apply_chat_template(prepared['messages'],tokenize=False,add_generation_prompt=True,**prepared['chat_template_kwargs']);ids=tokenizer.encode(prompt,add_special_tokens=False)
 checks.append({'step':i+1,'next':name or'final','prompt_tokens':len(ids),'output_budget':96,'total_budget':len(ids)+96,'fits_640':len(ids)+96<=640,'fits_1024':len(ids)+96<=1024,'prompt_SHA256':hashlib.sha256(prompt.encode()).hexdigest()})
 if i<4:
  name,args,result=steps[i];callid='call_'+str(i)+'_0123456789abcdef';messages.append({'role':'assistant','content':'','tool_calls':[{'id':callid,'type':'function','function':{'name':name,'arguments':json.dumps(args,separators=(',',':'))}}]});messages.append({'role':'tool','tool_call_id':callid,'content':json.dumps(result,separators=(',',':'))})
result={'CPU_only':True,'model_weights_loaded':False,'synthetic_capacity_example_only':True,'real_model_outputs':False,'not_P0_evidence':True,'model':'/old/models/ZhipuAI-GLM-5.3','checks':checks,'all_fit_640':all(x['fits_640']for x in checks)}
p=R/'results/GLM-context-count-only-r9392.json'
with p.open('x')as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
