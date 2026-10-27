import pathlib,json,re
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run=json.loads((r/'incoming/Kimi-ctx2k-Model-r127/launcher/run-context-public.json').read_bytes())['run'];out={'scope':'Existing478failedguard/lastcase samples only; no newGPU, nonce/argv/activations excluded','nodes':[]}
def scrub(x):
 if isinstance(x,str):return re.sub(r'(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])','<hex-redacted>',x)
 if isinstance(x,list):return [scrub(v)for v in x]
 if isinstance(x,dict):return {k:scrub(v)for k,v in x.items()if not any(t in k.lower()for t in ['nonce','argv','args'])}
 return x
for n in range(4):
 p=r/'results'/('Kimi-ctx2k-r127-'+run+'-node'+str(n)+'.json');q=json.loads(p.read_bytes());out['nodes'].append({'node':n,'failed_memory_guard':scrub(q.get('failed_memory_guard')),'last_memory_guard':scrub(q.get('last_memory_guard')),'cases':[{'label':c['label'],'memory_before_case':scrub(c.get('memory_before_case')),'memory_minimum_case':scrub(c.get('memory_minimum_case'))}for c in q.get('batch_cases',[])]})
print(json.dumps(out))
