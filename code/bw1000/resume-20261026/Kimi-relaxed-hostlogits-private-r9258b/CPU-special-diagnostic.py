import pathlib,json,os,hashlib
import torch
from transformers import AutoTokenizer
D=pathlib.Path('/refactor/incoming/Kimi-relaxed-hostlogits-r9258a');m=pathlib.Path('/shared/Kimi-K3')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(m/'config.json')=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
assert sha(m/'model.safetensors.index.json')=='a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd'
tok=AutoTokenizer.from_pretrained(str(m),trust_remote_code=True,local_files_only=True);V=json.loads((m/'config.json').read_bytes())['text_config']['vocab_size']
fixed=json.loads(pathlib.Path('/refactor/incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_bytes())
allids=sorted(set(map(int,tok.all_special_ids))|{int(k)for k,v in tok.added_tokens_decoder.items()if v.special}|set(fixed['eos_token_ids']))
# This diagnostic does not change a protection set or the real tokenizer.
extra=[{'id':i,'decoded_token':str(tok.convert_ids_to_tokens(i)),'registered':str(tok.added_tokens_decoder.get(i)),'all_special_ids':i in tok.all_special_ids}for i in allids if not 0<=i<V]
data=json.loads((D/'frozen-quality-data-r9258.json').read_bytes());import sys;sys.path.insert(0,str(D/'runtime'));from task_quality_r9258 import prepare
coverage=[]
for phase in ['pilot8','heldout96']:
 for q in data[phase]:
  text,_=prepare(q);other,_=prepare(dict(q,answer='gold_metadata_probe_must_not_appear_918262'))
  assert text==other
  ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
  ids2=tok.apply_chat_template([{'role':'user','content':other}],tokenize=True,add_generation_prompt=True)
  assert ids==ids2
  coverage.append({'id':q['id'],'phase':phase,'input_tokens':len(ids),'input_IDs_valid':all(0<=x<V for x in ids),'context_fits':1<=len(ids)<=512 and len(ids)+512<=1024,'gold_metadata_independent_prompt_and_tokenarray':True})
out={'GPU':False,'CUDA_initialized':torch.cuda.is_initialized(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'model_path':str(m),'tokenizer_class':type(tok).__name__,'tokenizer_len':len(tok),'V':V,'config_SHA256':sha(m/'config.json'),'index_SHA256':sha(m/'model.safetensors.index.json'),'tokenizer_files_SHA256':{p.name:sha(p)for p in m.iterdir()if p.is_file()and p.name in ['tokenization_kimi.py','tokenizer_config.json','tiktoken.model']},'declared_special_count':len(allids),'outside_model_declared_specials':extra,'EOS_frozen_unchanged':fixed['eos_token_ids'],'coverage':coverage,'all104_IDs_and_context_valid':all(x['input_IDs_valid']and x['context_fits']for x in coverage)}
assert not torch.cuda.is_initialized()
pathlib.Path('/refactor/results/Kimi-relaxed-special-diagnostic-r9258b.json').write_bytes(json.dumps(out,ensure_ascii=False,indent=2).encode());print(json.dumps({'actual_step':out['actual_step'],'extra':extra,'all104valid':out['all104_IDs_and_context_valid']}),flush=True)
