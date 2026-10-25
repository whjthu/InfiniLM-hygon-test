import pathlib,subprocess,json,ast,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming'
assert subprocess.run(['bash','-n',str(i/'GLM_attention_cohort_layer_r9054_inside.sh')]).returncode==0
m=json.loads((i/'GLM-attention_cohort-layer-manifest-r9054.json').read_bytes())
for n,h in m['sources_SHA256'].items():
 p=i/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 if n.endswith('.py'):compile(p.read_bytes(),n,'exec')
d=(i/'GLM_attention_cohort_layer_r9054.py').read_text();c=(i/'controller_GLM_attention_cohort_layer_r9054.py').read_text()
for node in ast.walk(ast.parse(d)):
 if isinstance(node,ast.Constant)and isinstance(node.value,str)and node.value.startswith('incoming/')and'manifest'in node.value:assert(r/node.value).exists(),node.value
assert "base=r/'candidates/glm-attention-cohort-r9054'"in d
assert 'glm_attention_cohort_policy_r9054.argtypes=[I,I]'in d and 'glm_attention_cohort_counts_r9054.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]'in d
assert 'glm_attention_cohort_counts_r9054(rank,v,4)'in d and 'return [v[0]+v[2],v[1]+v[3],0]'in d
assert 'glm_attention_cohort_policy_r9054(rank,mode)'in d and 'glm_attention_cohort_policy_r9054(rank,99)'in d
assert "'reduced_delta':delta"in d and "d['final_reduced_stats']"in c
assert "entry_delta[3:]==[1,0,0]"in d and "a['entry_delta']==[0,0,0,1,0,0]"in c
assert "report['profile']==0"in d and 'V(decode_out.data_ptr()),0)'in d and 'lookahead'not in d
p=r/'results/GLM-attention_cohort-layer-r9054-frontend.json';assert not p.exists();p.write_text(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'build':False,'frontend_manifest_and_actual_CAPI_arity_schema':'PASS','DSO':m['candidate_DSO_SHA256']}))
exec(compile((i/'launch_GLM_attention_cohort_layer_r9054.py').read_bytes(),'launch_GLM_attention_cohort_layer_r9054.py','exec'))
