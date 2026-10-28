"""Official GLM tokenizer CPU diff plus truthful import provenance.
No Native/model/GPU API is called. Torch import is observed, not prohibited.
CUDA initialized flag and open device descriptors are inspected as plain metadata.
"""
import copy,hashlib,importlib.util,json,os,pathlib,re,sys,datetime,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.environ.get('USE_TORCH')=='0'and os.environ.get('USE_TF')=='0'
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert re.fullmatch(r'\d+\.\d+',step)
out={'completed':False,'CPU_only':True,'actual_step':step,'Native_imported':False,'GPU':False,'model_weights_loaded':False,'application_GPU_API_calls':0,'import_phases':[],'torch_import_requests':[],'Native_import_attempts':[],'prior152_failure_preserved':True}
def metadata():
 torch=sys.modules.get('torch');cuda=sys.modules.get('torch.cuda');fds=[]
 fdroot=pathlib.Path('/proc/self/fd')
 if fdroot.is_dir():
  for p in fdroot.iterdir():
   try:value=os.readlink(p)
   except OSError:continue
   if value.startswith(('/dev/kfd','/dev/dri/render','/dev/nvidia')):fds.append(value)
 return {'Torch_imported':torch is not None,'Torch_file':getattr(torch,'__file__',None),'CUDA_module_imported':cuda is not None,'CUDA_initialized_flag':cuda.__dict__.get('_initialized')if cuda else None,'driver_device_fds':fds}
def phase(label):out['import_phases'].append({'phase':label,**metadata()})
class ImportAudit:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch'and not out['torch_import_requests']:
   out['torch_import_requests'].append({'phase':out['import_phases'][-1]['phase']if out['import_phases']else'initial','module':fullname,'stack':[{'file':f.filename,'line':f.lineno,'function':f.name,'source':f.line}for f in traceback.extract_stack(limit=16)[:-1]]})
  if fullname.startswith(('infinicore','_infinilm')):
   out['Native_import_attempts'].append(fullname);raise ImportError('CPU-only probe forbids Native import '+fullname)
  return None
audit=ImportAudit();sys.meta_path.insert(0,audit);phase('initial_before_protocol')
try:
 manifest=json.loads((D/'source-manifest.json').read_bytes())
 for n,h in manifest['files_SHA256'].items():assert sha(D/n)==h,n
 fixture=json.loads((D/'requests-fixture.json').read_bytes())
 P=R/'incoming/GLM-agent-service-r143c/overlay/python/infinilm/server/agent_protocol.py';assert sha(P)=='21a4d833eda5815c1feeba1a1e0ed5c9976fd2dcd5a65af11b7d4197a7ad2907'
 phase('before_protocol_import')
 s=importlib.util.spec_from_file_location('glm_r155_actual_agent_protocol',P);protocol=importlib.util.module_from_spec(s);sys.modules[s.name]=protocol;s.loader.exec_module(protocol)
 phase('after_protocol_import')
 phase('before_transformers_import')
 from transformers import PreTrainedTokenizerFast
 phase('after_transformers_import')
 phase('before_official_tokenizer_load')
 model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tokenizer=PreTrainedTokenizerFast.from_pretrained(str(model),local_files_only=True)
 phase('after_official_tokenizer_load')
 template=tokenizer.get_chat_template();assert isinstance(template,str)
 keyword=re.compile(r'think|reason|tool_choice|tools|generation_prompt',re.I)
 lines=template.splitlines();relevant=[{'line':i+1,'text':line}for i,line in enumerate(lines)if keyword.search(line)]
 cfgpath=model/'tokenizer_config.json';cfg=json.loads(cfgpath.read_bytes());cfg_fields={k:v for k,v in cfg.items()if re.search(r'think|reason',k,re.I)}
 out.update({'completed':False,'CPU_only':True,'actual_step':step,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Native_imported':False,'GPU':False,'model_weights_loaded':False,'model_output_generated':False,'original775_not_requalified':True,'fixture_SHA256':sha(D/'requests-fixture.json'),'source_manifest_SHA256':sha(D/'source-manifest.json'),'actual_protocol_SHA256':sha(P),'official_tokenizer_path':str(model),'tokenizer_class':type(tokenizer).__name__,'tokenizer_config_SHA256':sha(cfgpath),'tokenizer_config_thinking_fields':cfg_fields,'chat_template_SHA256':hashlib.sha256(template.encode()).hexdigest(),'chat_template':template,'chat_template_relevant_lines':relevant,'literal_keyword_present':{k:k in template for k in ['enable_thinking','thinking','clear_thinking','reasoning_effort','tools','tool_choice']},'cases':[]})
 phase('before_template_render')
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

 phase('after_all_template_renders');observed=metadata();out.update(observed)
 assert observed['CUDA_initialized_flag']in(False,None)and not observed['driver_device_fds'],'Actual CUDA/device context evidence found; do not label CPU-only pass'
 assert not any(n.startswith(('infinicore','_infinilm'))for n in sys.modules)
 out.update(completed=True,Native_imported=False,result_scope='Actual official-template CPU evidence. Torch import is reported truthfully; no application GPU API/Native/model call. Candidate A is copied kwargs only, not deployed code or model qualification.')
except BaseException as error:
 out.update(completed=False,error=repr(error),traceback=traceback.format_exc(),**metadata())
finally:
 if audit in sys.meta_path:sys.meta_path.remove(audit)
 target=R/'results'/('GLM-template-diff-r155-'+step+'.json')
 with target.open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps({'completed':out['completed'],'CPU_only':True,'actual_step':step,'path':str(target),'SHA256':sha(target),'Torch_imported':out.get('Torch_imported'),'CUDA_initialized_flag':out.get('CUDA_initialized_flag'),'error':out.get('error')},ensure_ascii=False))
if not out['completed']:raise SystemExit(1)
