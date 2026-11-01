from pathlib import Path
import json,hashlib
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label='GLM-latest99-semantic74-Model-r331'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
p=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+label+'.json')
q=json.loads(p.read_bytes());out['supervisor']=q;out['supervisor_SHA256']=sha(p)
out['owners']=[]
for stage in range(4):
 for rank in range(8):
  p=R/'results'/('GLM-latest-full-owner-r290-'+label+'-stage'+str(stage)+'-rank'+str(rank)+'.json')
  if p.exists():
   v=json.loads(p.read_bytes());rows=v.get('service_requests',[])
   out['owners'].append({'stage':stage,'rank':rank,'SHA256':sha(p),'count':len(rows),
    'normal_shutdown':v.get('normal_shutdown'),'normal_reset_close':v.get('normal_reset_close'),
    'service_error':v.get('service_error'),'cleanup':v.get('cleanup'),
    'signatures':[(r['actual_prompt_token_ids_SHA256'],r['generated_token_ids_SHA256'],r['generated_tokens'],r['reason'])for r in rows]})
print(json.dumps(out))
