"""Official-tokenizer CPU diff; no model weights/Native/GPU or output generation.

Current prepare is used byte-exact. Candidate A changes only prompt-visible
tools in a copied prepared dict, never a deployed overlay or parser schema.
"""
import copy,hashlib,importlib.util,json,os,pathlib,re,sys,datetime
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.environ.get('USE_TORCH')=='0'and os.environ.get('USE_TF')=='0'
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert re.fullmatch(r'\d+\.\d+',step)
manifest=json.loads((D/'source-manifest.json').read_bytes())
for n,h in manifest['files_SHA256'].items():assert sha(D/n)==h,n
fixture=json.loads((D/'requests-fixture.json').read_bytes())
P=R/'incoming/GLM-agent-service-r143c/overlay/python/infinilm/server/agent_protocol.py';assert sha(P)=='21a4d833eda5815c1feeba1a1e0ed5c9976fd2dcd5a65af11b7d4197a7ad2907'
s=importlib.util.spec_from_file_location('glm_r152_actual_agent_protocol',P);protocol=importlib.util.module_from_spec(s);sys.modules[s.name]=protocol;s.loader.exec_module(protocol)
from transformers import PreTrainedTokenizerFast
model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tokenizer=PreTrainedTokenizerFast.from_pretrained(str(model),local_files_only=True)
template=tokenizer.get_chat_template();assert isinstance(template,str)
keyword=re.compile(r'think|reason|tool_choice|tools|generation_prompt',re.I)
lines=template.splitlines();relevant=[{'line':i+1,'text':line}for i,line in enumerate(lines)if keyword.search(line)]
cfgpath=model/'tokenizer_config.json';cfg=json.loads(cfgpath.read_bytes());cfg_fields={k:v for k,v in cfg.items()if re.search(r'think|reason',k,re.I)}
out={'completed':False,'CPU_only':True,'actual_step':step,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Native_imported':False,'GPU':False,'model_weights_loaded':False,'model_output_generated':False,'original775_not_requalified':True,'fixture_SHA256':sha(D/'requests-fixture.json'),'source_manifest_SHA256':sha(D/'source-manifest.json'),'actual_protocol_SHA256':sha(P),'official_tokenizer_path':str(model),'tokenizer_class':type(tokenizer).__name__,'tokenizer_config_SHA256':sha(cfgpath),'tokenizer_config_thinking_fields':cfg_fields,'chat_template_SHA256':hashlib.sha256(template.encode()).hexdigest(),'chat_template':template,'chat_template_relevant_lines':relevant,'literal_keyword_present':{k:k in template for k in ['enable_thinking','thinking','clear_thinking','reasoning_effort','tools','tool_choice']},'cases':[]}
for case in fixture['cases']:
 variants=[];prompts={}
 for enabled in [False,True]:
  raw=copy.deepcopy(case['request']);raw.setdefault('chat_template_kwargs',{})['enable_thinking']=enabled
  prepared,schemas,choice=protocol.prepare_agent_request(raw,'glm5')
  for visibility in ['current']+(['candidate_A_all_declared_tools']if isinstance(raw.get('tool_choice'),dict)else[]):
   data=copy.deepcopy(prepared)
   if visibility!='current':data['chat_template_kwargs']['tools']=copy.deepcopy(raw['tools'])
   prompt=tokenizer.apply_chat_template(data['messages'],tokenize=False,add_generation_prompt=bool(data.get('add_generation_prompt',True)),**data['chat_template_kwargs']);ids=tokenizer.encode(prompt,add_special_tokens=False)
   prompts[(enabled,visibility)]=prompt
   variants.append({'enable_thinking':enabled,'visibility':visibility,'CPU_prompt_hypothesis_only':visibility!='current','prompt_tokens':len(ids),'prompt_SHA256':hashlib.sha256(prompt.encode()).hexdigest(),'tail_256_chars':prompt[-256:],'parser_thinking_initial_mode':prompt.rstrip().endswith(('<think>','<|open|>think<|sep|>')),'declared_tools':[x['function']['name']for x in raw.get('tools',[])],'template_visible_tools':[x['function']['name']for x in data['chat_template_kwargs'].get('tools')or[]],'parser_allowed_functions':list(schemas),'parser_choice':choice,'original_max_tokens':raw['max_tokens'],'original_total_budget':len(ids)+raw['max_tokens'],'budget_checks':{str(n):{'total':len(ids)+n,'fits640':len(ids)+n<=640}for n in [24,96,128,256]}})
 out['cases'].append({'case':case['case'],'original_request_SHA256':case['original_request_SHA256'],'variants':variants,'False_True_current_prompt_identical':prompts[(False,'current')]==prompts[(True,'current')]})
assert 'torch'not in sys.modules and not any(n.startswith(('infinicore','_infinilm'))for n in sys.modules)
out.update(completed=True,Torch_imported=False,result_scope='CPU actual official-template evidence only; candidate A is copied prompt kwargs, no deployed code change or model qualification')
target=R/'results'/('GLM-template-diff-r152-'+step+'.json')
with target.open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'completed':True,'CPU_only':True,'actual_step':step,'path':str(target),'SHA256':sha(target),'cases':out['cases'],'literal_keyword_present':out['literal_keyword_present']},ensure_ascii=False))
