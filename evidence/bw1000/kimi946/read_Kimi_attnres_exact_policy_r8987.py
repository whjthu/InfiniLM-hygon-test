import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/execution';out={'CPU_only':True,'files':[]}
for n in ['owner_affinity_r2392.hpp','submit.hpp']:
 p=base/n;lines=p.read_text().splitlines();text=p.read_text()if n.startswith('owner_')else '\n'.join(s for s in lines if 'owner'in s or '2392'in s)
 out['files'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':text})
print(json.dumps(out))
