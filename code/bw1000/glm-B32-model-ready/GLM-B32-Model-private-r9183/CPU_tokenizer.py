import pathlib,json,hashlib,os,traceback
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'weights_loaded':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 from transformers import PreTrainedTokenizerFast
 import torch
 assert not torch.cuda.is_initialized();model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tok=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True);tok.chat_template=(model/'chat_template.jinja').read_text();kwargs={'reasoning_effort':'low'}
 if'enable_thinking'in tok.chat_template:kwargs['enable_thinking']=False
 old=json.loads((r/'results/GLM-actual-concurrency-tokenized-r9132.json').read_text());src=inc/'quality-source-fresh64.json';data=json.loads(src.read_text());cases=data['cases'][32:64];assert len(cases)==32;assert not({x['id']for x in cases}&{x['id']for x in old['service_cases']});out=dict(old);out.update(cases=[],scheduled_count=0,excluded=[],service_cases=[],service_excluded=[],MMLU_source={'not_run':True},GSM_source={k:v for k,v in data.items()if k!='cases'},GSM_source_SHA256=sha(src),prompt_protocol='officialtestfresh64 latter32 disjoint GLM681/694 pilot32; zero-shot naturaltemplate lowreasoning noforcedthinkclose; noquestionpadding/truncation; C32M16vsC32M32 newraggedON both',tokenizer_SHA256=sha(model/'tokenizer.json'),template_SHA256=sha(model/'chat_template.jinja'),kwargs=kwargs)
 for c in cases:
  text=c['question']+chr(10)+'Solve this problem. End with #### followed by the numeric answer.';render=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,**kwargs);ids=[int(x)for x in tok.encode(render,add_special_tokens=False)]
  if len(ids)>384:out['service_excluded'].append({'id':c['id'],'input_tokens':len(ids),'reason':'wholeprompt+256output>640; no truncation'});continue
  out['service_cases'].append({'id':c['id'],'question':c['question'],'gold_numeric':c['gold_numeric'],'input_ids':ids,'input_tokens':len(ids),'max_new_tokens':256})
 q['excluded']=out['service_excluded'];assert len(out['service_cases'])==32, q['excluded'];target=r/'results/GLM-B32-tokenized-r9183.json';assert not target.exists();target.write_text(json.dumps(out,ensure_ascii=False,indent=2));q.update(completed=True,program_path=str(target),program_SHA256=sha(target),source_SHA256=sha(src),MMLU_not_rerun=True,service_distinct=32,disjoint_GLM681_694=True,service_input_range=[min(x['input_tokens']for x in out['service_cases']),max(x['input_tokens']for x in out['service_cases'])])
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(inc/'tokenizer-result.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
