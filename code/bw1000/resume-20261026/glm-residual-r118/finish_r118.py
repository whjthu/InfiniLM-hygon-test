import json,pathlib,hashlib,collections,statistics
f=pathlib.Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群')
q=read(f/'allowner-scalar755-r118.json');e=read(root/'GLM-down-row2-Model-private-r9284b/terminal-evidence-r9287.json')
expected={(x['stage'],x['rank']):x['SHA256'] for x in e['audit']['owners']}
assert len(q['owners'])==16 and all(expected[x['stage'],x['rank']]==x['SHA256'] for x in q['owners'])
byowner=[]
for o in q['owners']:
 g=collections.defaultdict(lambda:dict(calls=0,CAPI=0.,HOST_pipeline=[0.,0.,0.]))
 for k,v in o['groups'].items():
  label,op,T,B=k.split('/');key=(label,op);z=g[key];z['calls']+=v['calls'];z['CAPI']+=v['HOST_CAPI_seconds'];z['HOST_pipeline']=[x+y for x,y in zip(z['HOST_pipeline'],v['HOST_pipeline_sum'])]
 byowner.append(dict(stage=o['stage'],rank=o['rank'],groups={a+'/'+b:v for (a,b),v in g.items()}))
selected=collections.defaultdict(lambda:dict(calls=0,maxduration_owner_CAPI_sum=0.,same_owner_HOST_pipeline_sum=[0.,0.,0.],selected_owner_count={}))
for c in q['aligned_call_summary']:
 key=f"{c['label']}/{c['op']}/T{c['T']}/B{c['B']}";z=selected[key];v=c['maximum_duration_owner'];z['calls']+=1;z['maxduration_owner_CAPI_sum']+=v['HOST_CAPI_seconds'];z['same_owner_HOST_pipeline_sum']=[x+y for x,y in zip(z['same_owner_HOST_pipeline_sum'],v['HOST_pipeline'])];owner=f"s{v['stage']}/r{v['rank']}";z['selected_owner_count'][owner]=z['selected_owner_count'].get(owner,0)+1
print(json.dumps(dict(owner_phase_totals=byowner,selected_maxduration_shape_groups=selected),indent=2))
(f/'allowner-summary755-r118.json').write_text(json.dumps(dict(all16_SHA_match_local_audit=True,owners=byowner,maximum_duration_shape_groups=selected,limits='Per-call maximum-duration selection; three phases always selected from the same owner. Cross-rank host clocks not synchronized; this is not exact distributed critical-path or GPUbusy attribution.'),indent=2),encoding='utf8')
