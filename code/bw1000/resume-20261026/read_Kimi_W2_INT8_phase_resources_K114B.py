import pathlib,json,hashlib,re
b=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/candidates/Kimi-W2-INT8-phase-K114B');p=b/'provider.s';s=p.read_text().splitlines();out={'provider_assembly_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'kernels':[]}
for i,line in enumerate(s):
 if '-- Begin function'in line:
  end=next((j for j in range(i+1,len(s))if'-- End function'in s[j]),len(s));body=s[i:end+1];res=[x.strip()for x in body if any(k in x for k in ['.amdhsa_group_segment_fixed_size','.amdhsa_private_segment_fixed_size','.amdhsa_next_free_vgpr','.amdhsa_next_free_sgpr'])]
  if res:out['kernels'].append({'function':line.strip(),'resources':res,'body_SHA256':hashlib.sha256(('\n'.join(body)+'\n').encode()).hexdigest()})
print(json.dumps(out))
