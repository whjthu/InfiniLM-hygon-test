"""Single pure-data preflight shared by CPU, pre-GPU bootstrap and run entry."""
from pathlib import Path
import json,hashlib
GOLD_REMOTE_PATH='incoming/Deep-C4-independent64-r9278/dataset64.json'
GOLD_REMOTE_SHA='dc1e0d07fa8d456c8e912959eb54d70abb04a37738119ee9e2b3d8b4daa91918'
GOLD_SEMANTIC_COPY_SHA='053f9762acd68a60d3eab9ec7868d19378a43257892bc3ed6a9c9689229e168b'
TOKENIZED_REMOTE_SHA='4ef9fcad8bcf5529936d2e4eeb07083d74812f21893662effdafe89bd4ca114d'
IDS=['gsm8k-test-516','gsm8k-test-1055','gsm8k-test-757','gsm8k-test-584']
def validate_payloads(prepared,dataset,model,cfg):
 allgold=dataset['cases'];gold={x['id']:x for x in allgold};assert len(gold)==len(allgold)==64
 admitted=prepared['cases'];allcases={x['id']:x for x in admitted};skipped=prepared['skipped'];skipids=[x['id']for x in skipped]
 assert len(allcases)==len(admitted)==prepared['accepted_count'] and len(skipids)==len(set(skipids))
 assert set(allcases).isdisjoint(skipids)and set(allcases)|set(skipids)==set(gold),'prepared admission must partition exact frozen64 gold'
 assert prepared['planned_count']==64 and len(prepared['planned_ids'])==64 and set(prepared['planned_ids'])==set(gold)
 assert prepared['source_SHA256']==GOLD_REMOTE_SHA
 assert prepared['shot_count']==3 and prepared['capacity']==1024 and prepared['maximum_new_tokens']==512
 from deep_owner_consumer_r168a import native_expected_kinds
 assert len(native_expected_kinds(cfg))==43
 assert cfg['eos_token_id']==1 and cfg['num_hidden_layers']==43 and len(cfg['compress_ratios'])==44
 assert len(IDS)==len(set(IDS))==4 and set(IDS)<=set(allcases)and not set(IDS)&set(skipids)
 for n,h in prepared['tokenizer_files_SHA256'].items():assert hashlib.sha256((model/n).read_bytes()).hexdigest()==h,n
 cases=[allcases[i]for i in IDS]
 for x in cases:
  g=gold[x['id']];assert x['question']==g['question'] and x['question']in x['prompt_text']
  assert x['prompt_not_truncated']is True and x['no_padding']is True and x['gold_used_in_prompt']is False
  assert x['max_new_tokens']==512 and x['input_tokens']==len(x['input_ids']) and 64<x['input_tokens']<=512 and x['input_tokens']+512<=1024
  assert all(type(i)is int and 0<=i<cfg['vocab_size']for i in x['input_ids'])
  assert 'gold_answer'in g and 'gold_numeric'in g and 'source_index'in g
 metadata={'planned':64,'admitted':len(admitted),'rejected':len(skipped),'selected_ids':IDS,'selected_count':4,'selected_rejected':False,'selected_input_tokens':[x['input_tokens']for x in cases],'selected_questions_exact_original':True,'shot_count':3,'cap':512,'context':1024,'no_new_tokenization':True,'prompt_crop_or_answer_change':False}
 return gold,cases,metadata

def read_pinned(path,expected):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,str(path)
 return raw

def prepare_pilot(r,manifest,src,model,cfg):
 tp=r/manifest['tokenized_path'];assert manifest['existing_remote_SHA256'][manifest['tokenized_path']]==TOKENIZED_REMOTE_SHA
 raw=read_pinned(tp,TOKENIZED_REMOTE_SHA);prepared=json.loads(raw)
 gp=r/GOLD_REMOTE_PATH;graw=read_pinned(gp,GOLD_REMOTE_SHA)
 copyraw=read_pinned(src/'dataset64.json',GOLD_SEMANTIC_COPY_SHA)
 dataset=json.loads(graw);assert dataset==json.loads(copyraw),'semantic sourcecopy must match all64 originalquestions/answers/metadata'
 gold,cases,metadata=validate_payloads(prepared,dataset,model,cfg)
 metadata.update(tokenized_SHA256=TOKENIZED_REMOTE_SHA,gold_remote_path=str(gp),gold_remote_SHA256=GOLD_REMOTE_SHA,gold_semantic_copy_SHA256=GOLD_SEMANTIC_COPY_SHA,all64_parsed_gold_equal=True)
 return prepared,gold,cases,metadata
