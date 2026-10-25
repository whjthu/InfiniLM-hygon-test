import pathlib,json,hashlib,os,ast,subprocess,time,traceback
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();result={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};start=time.monotonic()
try:
 from GLM_transient_preflight_r9059 import validate
 result['parent648_preflight']=validate();assert result['parent648_preflight']['model_library_metadata']['library_SHA256']=='fbd07490dd9887868e4a6f5303170969dfa188e9cbcb39f7dca708b3a303741a'
 from transformers import PreTrainedTokenizerFast
 import torch
 assert not torch.cuda.is_initialized();model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tok=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True);tok.chat_template=(model/'chat_template.jinja').read_text();kwargs={'reasoning_effort':'low'}
 if 'enable_thinking'in tok.chat_template:kwargs['enable_thinking']=False
 def ids(text,forced=False):
  rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,**kwargs)
  if forced:assert rendered.rstrip().endswith('<think>');rendered+='</think>'+chr(10)
  return [int(x)for x in tok.encode(rendered,add_special_tokens=False)]
 src=inc/'quality-bench-mmlu-r9132.json';data=json.loads(src.read_text());gsm=inc/'service-gsm8k-source.json';gdata=json.loads(gsm.read_text());out={'source_SHA256':sha(src),'GSM_source_SHA256':sha(gsm),'CPU_only':True,'max_context':640,'maximum_new_tokens':256,'scheduled_count':len(data['cases']),'cases':[],'excluded':[],'service_cases':[],'service_excluded':[],'answer_continuations':{x:tok.encode(' '+x,add_special_tokens=False)for x in 'ABCD'},'generation_config':json.loads((model/'generation_config.json').read_text()),'MMLU_source':{k:v for k,v in data.items()if k!='cases'},'GSM_source':{k:v for k,v in gdata.items()if k!='cases'},'prompt_protocol':'MMLU zero-shot chat forcedemptythinking finalA-D likelihood; GSM actualnatural generation zero-shot withrealcheckpoint template, noforcedclose; noansweringoldprompt','tokenizer_SHA256':sha(model/'tokenizer.json'),'template_SHA256':sha(model/'chat_template.jinja'),'kwargs':kwargs}
 for c in data['cases']:
  text='The following is a multiple-choice question about '+c['subject'].replace('_',' ')+'. Choose the correct answer and reply with its letter only.'+chr(10)*2+c['question']+chr(10)+chr(10).join(chr(65+i)+'. '+v for i,v in enumerate(c['choices']))+chr(10)+'Answer:';v=ids(text,True)
  if len(v)+max(map(len,out['answer_continuations'].values()))>640:out['excluded'].append({'id':c['id'],'input_tokens':len(v),'reason':'wholequestion+fullanswercontinuation exceedscontext640; notruncation'});continue
  out['cases'].append({'id':c['id'],'subject':c['subject'],'gold':c['answer'],'input_ids':v,'input_tokens':len(v),'source_index':c['source_index'],'prompt':text})
 for c in gdata['cases']:
  text=c['question']+chr(10)+'Solve this problem. End with #### followed by the numeric answer.';v=ids(text)
  if len(v)>384:out['service_excluded'].append({'id':c['id'],'input_tokens':len(v),'reason':'prompt+256output exceeds640; notruncation'});continue
  if len(out['service_cases'])<32:out['service_cases'].append({'id':c['id'],'question':c['question'],'gold_numeric':c['gold_numeric'],'input_ids':v,'input_tokens':len(v),'max_new_tokens':256})
 assert len(out['service_cases'])==32 and len({x['id']for x in out['service_cases']})==32 and len(out['cases'])>=16
 assert all(1<=len(v)<=2 for v in out['answer_continuations'].values());from schedule_r9132 import selftest;result['scheduler_CPU']=selftest()
 for p in inc.glob('*.py'):ast.parse(p.read_text())
 library=pathlib.Path(result['parent648_preflight']['model_library_metadata']['library_actual_path']);symbols=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True);required=['modern_glm_model_forward','glm_attention_cohort_policy_r9054','glm_attention_cohort_counts_r9054','glm_reduced_output_policy_r9032','modern_glm_model_destroy','modern_glm_model_shutdown'];assert all(x in symbols for x in required);result['ABI_symbols']=required
 target=r/'results/GLM-actual-concurrency-tokenized-r9132.json';assert not target.exists();target.write_text(json.dumps(out,ensure_ascii=False,indent=2));result.update(completed=True,program_path=str(target),program_SHA256=sha(target),MMLU_kept=len(out['cases']),MMLU_excluded=out['excluded'],MMLU_newdisjoint=data.get('disjoint_prior_r9102',False),service_distinct=32,service_input_range=[min(x['input_tokens']for x in out['service_cases']),max(x['input_tokens']for x in out['service_cases'])],service_excluded=out['service_excluded'],source_SHA256={p.name:sha(p)for p in inc.iterdir()if p.is_file()},elapsed_seconds=time.monotonic()-start)
except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc())
(inc/'CPU-result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result.get(k)for k in ['completed','producer','error','program_SHA256','MMLU_kept','MMLU_excluded','MMLU_newdisjoint','service_distinct','service_input_range','service_excluded']}));assert result['completed']
