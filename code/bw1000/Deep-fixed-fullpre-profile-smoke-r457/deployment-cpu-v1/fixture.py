"""Exact first GSM/MMLU bodies and prior Deep IDs; no encoding or reselection."""
from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
def assets(card,local_files=None):
 def get(key):
  path=Path(card[key+'_path'])if local_files is None else Path(local_files)/{'cases':'cases.json','requests':'requests-private.json','encoding':'encoding.json'}[key]
  assert sha(path)==card[key+'_SHA256'];return json.loads(path.read_bytes())
 cases=get('cases')['cases'];requests=get('requests')['requests'];encoded=get('encoding');assert encoded['actual_CPU']=='910407.945'and encoded['completed']and encoded['passed']and encoded['actual_Deep_codec']and encoded['actual_public_protocol']and not encoded['GPU']and not encoded['Native_dlopen']
 out=[]
 for i,suite in enumerate(['gsm8k','mmlu']):
  c=next(c for c in cases if c['suite']==suite);e=next(e for e in encoded['cases']if e['suite']==suite);entry=card['plan'][i]
  assert c['id']==e['case_id']==entry['case_id']and entry['index']==i and entry['selected_ordinal']==e['selected_ordinal']
  r=next(r for r in requests if r['case_id']==c['id']);raw=compact(r['body'])
  assert hashlib.sha256(raw).hexdigest()==entry['request_body_SHA256']==e['request_body_SHA256']==r['request_body_SHA256']
  assert e['prompt_IDs_SHA256']==entry['prompt_IDs_SHA256']and e['prompt_tokens']==entry['prompt_tokens']and e['prompt_tokens']+4096<=32768
  body=r['body'];assert body['stream']is True and body['text_delivery']=='P1'and body['max_tokens']==4096 and body['chat_template_kwargs']=={'enable_thinking':False}
  assert entry['mode']==1 and entry['warmup']is False and entry['budget']==4096 and entry['thinking']is False
  out.append((entry,raw,body,c))
 assert len(card['plan'])==len(out)==2;return out
