import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'scope':'Existing764 terminal per-owner reports only; no new GPU or Model upper-bound claim','nodes':[]}
for n in range(4):
 p=r/'results'/('Kimi-TP32-I96-MoE-r120-910375.764-node'+str(n)+'.json');q=json.loads(p.read_bytes())
 assert q['completed']and q['normal_CUDA_cleanup']
 owners=[]
 for o in q['native']['reports']:
  owners.append({k:v for k,v in o.items()if k!='arms'})
  owners[-1]['arms_memory']=[{k:v for k,v in a.items()if any(t in k.lower()for t in ['free','memory','bytes','t','arm']) and k not in ['output_bits','raw_output']}for a in o['arms']]
 out['nodes'].append({'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'node_rank':n,'owners':owners})
print(json.dumps(out))
