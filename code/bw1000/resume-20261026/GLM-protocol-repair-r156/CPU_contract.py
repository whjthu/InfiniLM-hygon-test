"""CPU real official-template and strict parser contracts, not model output/P0."""
import copy,hashlib,importlib.util,json,os,pathlib,re,sys,types,datetime
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.environ.get('USE_TORCH')=='0'and os.environ.get('USE_TF')=='0'
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert re.fullmatch(r'\d+\.\d+',step)
m=json.loads((D/'source-manifest.json').read_bytes())
for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
base=D/'overlay/python/infinilm/server';pkg=types.ModuleType('glm_protocol_contract_r156');pkg.__path__=[str(base)];sys.modules[pkg.__name__]=pkg
name=pkg.__name__+'.agent_protocol';s=importlib.util.spec_from_file_location(name,base/'agent_protocol.py');protocol=importlib.util.module_from_spec(s);sys.modules[name]=protocol;s.loader.exec_module(protocol)
name=pkg.__name__+'.tool_parsers';s=importlib.util.spec_from_file_location(name,base/'tool_parsers/__init__.py',submodule_search_locations=[str(base/'tool_parsers')]);parsers=importlib.util.module_from_spec(s);sys.modules[name]=parsers;s.loader.exec_module(parsers)
fixture=json.loads((D/'requests-fixture.json').read_bytes());raw=copy.deepcopy(next(x['request']for x in fixture['cases']if x['case']=='request-002.request.json'));checks=[]
prepared,schemas,choice=protocol.prepare_agent_request(raw,'glm5');assert [t['function']['name']for t in prepared['chat_template_kwargs']['tools']]==['read_file','calculate','write_file']and list(schemas)==['read_file']and choice=='read_file';checks.append('glm_named_all_template_tools_but_only_named_parser_schema')
good='<tool_call>read_file<arg_key>path</arg_key><arg_value>input.json</arg_value></tool_call>';call=parsers.parse_xml_call(good,schemas,'glm5');assert call['function']['name']=='read_file'and json.loads(call['function']['arguments'])=={'path':'input.json'};checks.append('CPU_literal_parser_named_read_allowed')
bad='<tool_call>calculate<arg_key>a</arg_key><arg_value>12</arg_value><arg_key>b</arg_key><arg_value>34</arg_value></tool_call>'
try:parsers.parse_xml_call(bad,schemas,'glm5')
except protocol.AgentProtocolError:checks.append('CPU_literal_parser_other_declared_tool_still_rejected_for_named_read')
else:raise AssertionError('Named restriction loosened')
try:parsers.ToolCallParser(schemas,choice=choice,parser='glm5',parallel=False).feed('',final=True)
except protocol.AgentProtocolError:checks.append('named_missing_call_still_rejected')
else:raise AssertionError('Missing named call accepted')
for selected in ['none','auto','required']:
 test=copy.deepcopy(raw);test['tool_choice']=selected;p,sc,ch=protocol.prepare_agent_request(test,'glm5');expected=[]if selected=='none'else['read_file','calculate','write_file'];assert list(sc)==expected and[t['function']['name']for t in p['chat_template_kwargs']['tools']or[]]==expected
checks.append('none_auto_required_semantics_preserved')
# Other parser retains old selected-template filtering. No Deep/Kimi behavior change.
p,sc,ch=protocol.prepare_agent_request(raw,'deepseek_v4');assert[t['function']['name']for t in p['chat_template_kwargs']['tools']]==['read_file']and list(sc)==['read_file'];checks.append('non_glm_named_filter_unchanged')
from transformers import PreTrainedTokenizerFast
tok=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True);template=tok.get_chat_template();assert hashlib.sha256(template.encode()).hexdigest()==m['actual_template_SHA256']
def render(data):
 p,sc,ch=protocol.prepare_agent_request(data,'glm5');text=tok.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True,**p['chat_template_kwargs']);ids=tok.encode(text,add_special_tokens=False);return text,len(ids),p
variants=[];outputs={}
for enabled in [False,True]:
 test=copy.deepcopy(raw);test['chat_template_kwargs']['enable_thinking']=enabled;outputs[enabled]=render(test)[0]
assert outputs[False]==outputs[True]and outputs[False].endswith('<think>');checks.append('enable_thinking_false_true_still_identical_no_fake_disable_mapping')
for effort in ['low','high','max']:
 test=copy.deepcopy(raw);test['reasoning_effort']=effort;text,n,p=render(test);assert p['chat_template_kwargs']['reasoning_effort']==effort and'Reasoning Effort: '+effort.capitalize()in text and text.endswith('<think>');variants.append({'explicit_reasoning_effort':effort,'prompt_tokens':n,'tail256':text[-256:],'prompt_SHA256':hashlib.sha256(text.encode()).hexdigest(),'still_thinking':True,'total_with256':n+256,'fits640':n+256<=640})
assert len({v['prompt_SHA256']for v in variants})==3;checks.append('existing_top_level_effort_passthrough_renders_real_low_high_max')
history={'messages':[{'role':'user','content':'old question'},{'role':'assistant','content':'old answer','reasoning_content':'HISTORY_REASONING_R156'},{'role':'user','content':'new question'}],'tool_choice':'none','reasoning_effort':'low','chat_template_kwargs':{'clear_thinking':False}}
a=render(history)[0];history['chat_template_kwargs']['clear_thinking']=True;b=render(history)[0];assert 'HISTORY_REASONING_R156'in a and'HISTORY_REASONING_R156'not in b and a.endswith('<think>')and b.endswith('<think>');checks.append('clear_thinking_only_clears_eligible_history_not_current_thinking')
# Compact illustrative complete history capacity, not generated calls or P0 proof.
steps=[('read_file',{'path':'input.json'},{'a':73,'b':82,'nonce':'31af75ce'}),('calculate',{'a':73,'b':82},{'sum':155}),('write_file',{'path':'result.json','content':'{"nonce":"31af75ce","sum":155}'},{'ok':True}),('read_file',{'path':'result.json'},{'nonce':'31af75ce','sum':155})];messages=copy.deepcopy(raw['messages']);capacity=[]
for i in range(5):
 data=copy.deepcopy(raw);data.update(messages=copy.deepcopy(messages),reasoning_effort='low',tool_choice={'type':'function','function':{'name':steps[i][0]}}if i<4 else'none');text,n,_=render(data);capacity.append({'step':i+1,'prompt_tokens':n,'total_with128':n+128,'total_with256':n+256,'fits640_with256':n+256<=640,'synthetic_history_only':True,'generated_reasoning_not_included':True})
 if i<4:
  fn,args,result=steps[i];cid='call_r156_'+str(i);messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':cid,'type':'function','function':{'name':fn,'arguments':json.dumps(args,separators=(',',':'))}}]},{'role':'tool','tool_call_id':cid,'content':json.dumps(result,separators=(',',':'))}])
cuda=sys.modules.get('torch.cuda');flag=cuda.__dict__.get('_initialized')if cuda else None;assert flag in(False,None);assert not any(n.startswith(('infinicore','_infinilm'))for n in sys.modules)
out={'completed':True,'passed':True,'CPU_only':True,'actual_step':step,'GPU':False,'Native':False,'model_weights_loaded':False,'actual_model_output_generated':False,'Torch_imported':'torch'in sys.modules,'CUDA_initialized_flag':flag,'manifest_SHA256':sha(D/'source-manifest.json'),'template_SHA256':m['actual_template_SHA256'],'checks':checks,'effort_variants':variants,'synthetic_complete_history_capacity':capacity,'limits':['CPU literal parser contract is not model P0','enable_thinking remains unsupported/ignored; low still generates reasoning','Full real reasoning history may exceed640 even if first request fits; no truncation/adoption claim','775 remains failed; no actual repair GPU run'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
target=R/'results'/('GLM-protocol-repair-r156-'+step+'.json')
with target.open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'passed':True,'CPU_only':True,'actual_step':step,'path':str(target),'SHA256':sha(target),'checks':checks,'capacity':capacity}))
