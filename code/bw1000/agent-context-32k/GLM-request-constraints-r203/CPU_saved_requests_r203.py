"""Three exact saved GLM second requests; official tokenizer only; safe summary."""
import argparse,copy,hashlib,importlib.abc,importlib.util,json,os,pathlib,re,sys
from api_request_constraints_r203 import copied_messages,make_encode_chat,request_constraints
from CPU_contract_r203 import run as fake_contract,source_pin,load
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();D=pathlib.Path(__file__).resolve().parent
class BlockBackend(importlib.abc.MetaPathFinder,importlib.abc.Loader):
 def find_spec(self,fullname,path=None,target=None):
  # Availability metadata checks may request a spec without importing it.
  # Block actual module creation, rather than breaking optional find_spec().
  if fullname.split('.')[0]in ['torch','_infinilm','infinilm']:return importlib.util.spec_from_loader(fullname,self)
 def create_module(self,spec):raise ImportError('Backend import blocked in tokenizer-only CPU comparison')
 def exec_module(self,module):raise ImportError('Backend import blocked in tokenizer-only CPU comparison')
def no_backend():assert not any(x.split('.')[0]in ['torch','_infinilm','infinilm']for x in sys.modules)
def deny_weights(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0]);suffix=pathlib.Path(path).suffix
  if suffix in ['.safetensors','.pt','.pth','.ckpt']or re.search(r'(?:pytorch_model|model-\d+-of-\d+)\.bin$',path):raise RuntimeError('Weight payload opening forbidden in tokenizer-only CPU comparison')
def main():
 a=argparse.ArgumentParser();a.add_argument('--source',default='/refactor/incoming/GLM-regular-service-r192');a.add_argument('--run',required=True);x=a.parse_args();assert re.fullmatch('GLM-regular-service-r192-[0-9a-f]{16}',x.run)
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);no_backend();sys.meta_path.insert(0,BlockBackend());sys.addaudithook(deny_weights)
 mf=json.loads((D/'manifest-r203.json').read_bytes())
 for p,h in mf['source_SHA256'].items():assert sha(D/p)==h
 source=pathlib.Path(x.source);source_pin(source);fake=fake_contract(source);protocol=load('r203_actual_GLM_protocol',source/'overlay/python/infinilm/server/agent_protocol.py')
 os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0')
 from transformers import PreTrainedTokenizerFast
 no_backend();model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tokenizer=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True);no_backend()
 template=tokenizer.get_chat_template();assert hashlib.sha256(template.encode()).hexdigest()=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c';encode=make_encode_chat(tokenizer);root=pathlib.Path('/refactor/results')/('GLM-regular-live-r192-'+x.run)
 pins={'HTTP':'6957b54d0a3f5f63138503f648a190879eceb00798ae2707881d283019aa11d1','SSE':'5af35e4d25b3951ae8a4dbfedf509d371681a21f0baa92df8db2674c01ad9e53','thinking':'6957b54d0a3f5f63138503f648a190879eceb00798ae2707881d283019aa11d1'};rows=[]
 for label,h in pins.items():
  p=root/label/'request-002.request.json';w=root/label/'request-002.wire';assert sha(w)==h and b'Multiple calls with parallel_tool_calls=false'in w.read_bytes();raw=p.read_bytes();body=json.loads(raw);original=copy.deepcopy(body);data,schemas,choice=protocol.prepare_agent_request(body,'glm5');data['_agent_choice']=choice;data['_agent_schemas']=schemas;kwargs=data['chat_template_kwargs'];messages=data['messages'];before=copy.deepcopy(messages);old=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs);oldids=tokenizer.encode(old,add_special_tokens=False);new,ids=encode(messages,data,True,kwargs)
  assert body==original and messages==before and len(kwargs['tools'])==len(body['tools'])==3 and len(schemas)==1 and list(schemas)==[choice]
  rules=request_constraints(data,kwargs['tools']);assert rules in new and 'at most one real tool call'in rules;copied=copied_messages(messages,data,kwargs);expected=copy.deepcopy(messages);expected[0]['content']+='\n\n'+rules;assert copied==expected
  assert kwargs['clear_thinking']is False and body['max_tokens']in [4096,8192]and len(ids)+body['max_tokens']<=32768
  thoughts=[m['reasoning_content']for m in messages if m.get('reasoning_content')];assert all((v in old)==(v in new)for v in thoughts)
  alternate=copy.deepcopy(data);alternate['parallel_tool_calls']=True;parallel,_=encode(messages,alternate,True,kwargs);assert parallel!=new
  auto=copy.deepcopy(data);auto['_agent_choice']='auto';auto['parallel_tool_calls']=True;untouched,_=encode(messages,auto,True,kwargs);assert untouched==old
  required=copy.deepcopy(data);required['_agent_choice']='required';rp,_=encode(messages,required,True,kwargs);assert 'at least one real tool call'in rp
  none=copy.deepcopy(body);none['tool_choice']='none';nd,ns,nc=protocol.prepare_agent_request(none,'glm5');nd['_agent_choice']=nc;assert not ns and not nd['chat_template_kwargs']['tools'];np,_=encode(nd['messages'],nd,True,nd['chat_template_kwargs']);assert 'Do not invoke any tool.'in np
  rows.append({'request_SHA256':hashlib.sha256(raw).hexdigest(),'wire_SHA256':h,'old_prompt_SHA256':hashlib.sha256(old.encode()).hexdigest(),'new_prompt_SHA256':hashlib.sha256(new.encode()).hexdigest(),'old_prompt_tokens':len(oldids),'new_prompt_tokens':len(ids),'output_budget':body['max_tokens'],'full_tool_count':len(kwargs['tools']),'named_schema_count':len(schemas),'parallel_flag_changes_prompt':True,'auto_parallel_true_byte_identical':True,'copied_history_and_reasoning_preserved':True,'existing_reasoning_fields':len(thoughts),'clear_thinking_false_retained':True,'required_and_none_API_rules':True})
 no_backend();report={'completed':True,'CPU_only':True,'scope':'Three saved GLM requests, official tokenizer/template counterfactual preparation only; new Model/HTTP/GPU effect NOT_RUN','source_manifest_SHA256':sha(D/'manifest-r203.json'),'old_GLM_source_manifest_SHA256':sha(source/'source-manifest-r192.json'),'tokenizer_json_SHA256':sha(model/'tokenizer.json'),'official_template_SHA256':hashlib.sha256(template.encode()).hexdigest(),'fake_contract':fake,'requests':rows,'Torch_imported':False,'Native_loaded':False,'weights_opened':False,'new_Model_PASS':False};print(json.dumps(report,separators=(',',':')))
if __name__=='__main__':
 try:main()
 except BaseException as e:print(json.dumps({'completed':False,'CPU_only':True,'error_class':type(e).__name__,'error_SHA256':hashlib.sha256(str(e).encode()).hexdigest(),'Torch_imported':'torch'in sys.modules,'Native_loaded':'_infinilm'in sys.modules,'new_Model_PASS':False}));raise SystemExit(1)
