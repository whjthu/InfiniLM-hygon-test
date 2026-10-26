import pathlib,json,hashlib,os,time

def prepare():
 inc=pathlib.Path(__file__).parent;out=inc/'program-r9241.json'
 if os.environ.get('SLURM_PROCID')!='0':
  end=time.monotonic()+90
  while not out.exists():assert time.monotonic()<end;time.sleep(.2)
  return json.loads(out.read_bytes())
 r=pathlib.Path('/refactor');base=json.loads((r/'results/GLM-B32-tokenized-r9183.json').read_bytes());raw=inc/'quality-source-fresh64.json';assert hashlib.sha256(raw.read_bytes()).hexdigest()=='7aae3e4f1927dec55d11defc5801a6cc07451a7350e4920b33df3d0708a81a22';data=json.loads(raw.read_bytes());byid={v['id']:v for v in data['cases']};assert len(byid)==64
 from transformers import PreTrainedTokenizerFast
 import torch
 assert not torch.cuda.is_initialized();model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tok=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True);tok.chat_template=(model/'chat_template.jinja').read_text();kwargs=base['kwargs'];main={v['id']:v for v in base['service_cases']};cases=[]
 ordered=list(main)+[v['id']for v in data['cases']if v['id']not in main];assert len(ordered)==64
 for ident in ordered:
  c=byid[ident];text=c['question']+chr(10)+'Solve this problem. End with #### followed by the numeric answer.';render=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,**kwargs);ids=[int(x)for x in tok.encode(render,add_special_tokens=False)];assert 0<len(ids)<=384
  if ident in main:assert ids==main[ident]['input_ids']and c['gold_numeric']==main[ident]['gold_numeric']
  cases.append({'id':ident,'question':c['question'],'gold_numeric':c['gold_numeric'],'input_ids':ids,'input_tokens':len(ids),'max_new_tokens':256})
 result=dict(base);result.update(service_cases=cases,GSM_optional64_scope='fixed64, first32 identical719 selectionbenchmark; latter32 notnewglobally but64unique withinreview; no64concurrency',GSM_source_SHA256=hashlib.sha256(raw.read_bytes()).hexdigest());blob=json.dumps(result,ensure_ascii=False,indent=2).encode()
 if out.exists():assert out.read_bytes()==blob
 else:tmp=out.with_suffix('.tmp');tmp.write_bytes(blob);tmp.replace(out)
 return result
