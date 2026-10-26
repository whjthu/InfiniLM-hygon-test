import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/deep-c4-smallt-model-r9254-910374.1112.json';raw=p.read_bytes();d=json.loads(raw);rows=[]
for t in d.get('quality_trials',[])+d.get('optional_B32_trials',[]):
 for v in t['records']:
  if v['id']=='gsm8k-test-953':rows.append({'label':t['label'],'B':t['B'],'smallt_arm':t['smallt_arm'],'record':{k:v[k]for k in ['id','stop_reason','input_tokens','score']},'generated_count':len(v['generated_ids']),'text':v['text'],'common_decode_useful_tok_s':t['common_decode_useful_tok_s']})
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'step':d['step'],'completed':d['completed'],'source_SHA256':hashlib.sha256(raw).hexdigest(),'discordant_id':'gsm8k-test-953','rows':rows,'GPU_submitted':False}))
