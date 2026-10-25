import pathlib,json,hashlib,os,sys,time,inspect
from transformers import PreTrainedTokenizerFast
from deepseek_v4_reference_template_r18_r2 import apply_reference_fallback,encode_reference_single_user
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');src=r/'incoming/distinct32-tasks-r9064.json';out=r/'results/Deep-distinct32-frozen-r9070b.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();started=time.monotonic();q=json.loads(src.read_bytes());assert len(q['cases'])==32
assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
tok=PreTrainedTokenizerFast.from_pretrained(str(model),local_files_only=True);template=apply_reference_fallback(tok)
def encode(text):return encode_reference_single_user(tok,text)
historical=r/'results/deep-serving-cases-r2130-910374.1018.json';h=json.loads(historical.read_bytes());assert all(encode(c['prompt_text'])==c['input_ids']for c in h['cases']);assert template['template_sha256']==h['template_provenance']['template_sha256']
filler='补充说明：本段仅用于补充文档背景，不改变原始资料，也不增加任务条件。请以原始资料和任务要求为准。'
def exact512(case):
 def text(n):return '原始资料：'+case['context']+'\n补充背景：'+(filler*64)[:n]+'\n任务：'+case['instruction']
 lo,hi=0,len(filler)*64;assert len(encode(text(lo)))<512<len(encode(text(hi)))
 while hi-lo>1:
  mid=(lo+hi)//2
  if len(encode(text(mid)))<512:lo=mid
  else:hi=mid
 # BPE token counts need not be monotone at adjacent characters; bounded exact search.
 for n in range(max(0,lo-96),min(len(filler)*64,hi+96)+1):
  for ending in ['', ' ', ' .', ' x', '\n']:
   # Only supplementary background changes; never alter original task/context.
   t=text(n).replace('\n任务：',ending+'\n任务：');ids=encode(t)
   if len(ids)==512:
    assert case['context']in t and case['instruction']in t and ids==encode(t)
    return dict(id=case['id'],category=case['category'],requested_style=case['requested_style'],prompt_text=t,input_ids=ids,input_tokens=512,context_padding_chars=n,context_padding_ending=ending,original_context_complete=True,original_instruction_complete=True,max_new_tokens=128,stop_first_eos=True,input_IDs_SHA256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
 raise RuntimeError('Exact512notfound_STOP_no_input_truncation:'+case['id'])
primary=[exact512(c)for c in q['cases']];assert len({x['input_IDs_SHA256']for x in primary})==32
short=[]
for c in q['cases'][:8]:
 ids=encode(c['prompt_text']);short.append(dict(id=c['id'],prompt_text=c['prompt_text'],input_ids=ids,input_tokens=len(ids),max_new_tokens=128,stop_first_eos=True,EOS_generation_not_guaranteed=True))
long=[]
for c in q['cases']:
 if c['requested_style']!='long_natural':continue
 body=c['context'];paragraph='\n说明文档需要保留问题背景、适用范围、实施过程和验证方式。评审者应能从记录中分辨已确认的事实、暂定的假设以及需要继续验证的事项。发现错误时应记录原始输入与原因，修正后保留前后结果，以便复核。'
 while len(encode('资料：'+body+'\n任务：'+c['instruction']))<512:body+=paragraph
 t='资料：'+body+'\n任务：'+c['instruction'];ids=encode(t);assert 512<=len(ids)<=768 and len(ids)+128<=1024
 long.append(dict(id=c['id'],prompt_text=t,input_ids=ids,input_tokens=len(ids),max_new_tokens=128,stop_first_eos=True,constructed_context=True,not_production_trace=True))
files={}
for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
 p=model/name
 if p.exists():files[str(p)]=dict(SHA256=sha(p),bytes=p.stat().st_size)
helper=pathlib.Path(inspect.getfile(encode_reference_single_user));result=dict(hardware='BW1000',level='CPU_TOKENIZED_FROZEN_INPUTS_NOT_MODEL_TESTED',status='FROZEN_DISTINCT32_I512',producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_seconds=time.monotonic()-started,CPU_only=True,no_native_DSO=True,no_GPU=True,no_weights_loaded=True,tokenizer_source=str(model),tokenizer_provenance=files,historical1018=dict(case_file=str(historical),SHA256=sha(historical),all_old_case_IDs_reencoded_exact=True,template_SHA_exact=True,historical_full_tokenizer_bytes_SHA='NOT_RECORDED; currentfileSHA plusknowncase/templateidentity, notclaimunrecordedhistoricalfilehash'),template_provenance=template,reference_helper=dict(path=str(helper),SHA256=sha(helper)),source_tasks=dict(path=str(src),SHA256=sha(src)),primary=primary,natural_short_EOS=short,natural_long=long,limits=['Primaryexact512 includesexplicitbenignconstructedcontext; taskcontext/instruction preserved, nofront512truncation orEOSpadding.','32distinctactualIDsequences verified, buttasktextnotproductiontraces.','NaturalshortEOScase actualgeneration/EOS/cachecompaction nottested.','NaturalpanelsoutsideadoptedI512scope mustkeepCastOFF.','Original1018alreadydestroyed; inputsfornextnecessaryload, notnewModelrun.'])
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=True,producer=result['producer'],CPU_seconds=result['CPU_seconds'],path=str(out),SHA256=sha(out),distinct32=len(primary),all_I512=all(x['input_tokens']==512 for x in primary),natural_short_lengths=[x['input_tokens']for x in short],natural_long_lengths=[x['input_tokens']for x in long])))
