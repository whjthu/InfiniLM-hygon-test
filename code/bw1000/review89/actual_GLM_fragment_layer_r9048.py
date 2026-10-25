import pathlib,subprocess,json,ast,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming'
assert subprocess.run(['bash','-n',str(i/'GLM_fragment_layer_r9048_inside.sh')]).returncode==0
m=json.loads((i/'GLM-fragment-layer-manifest-r9048.json').read_bytes())
for n,h in m['sources_SHA256'].items():
 p=i/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 if n.endswith('.py'):compile(p.read_bytes(),n,'exec')
driver=(i/'GLM_fragment_layer_r9048.py').read_text();controller=(i/'controller_GLM_fragment_layer_r9048.py').read_text()
for node in ast.walk(ast.parse(driver)):
 if isinstance(node,ast.Constant)and isinstance(node.value,str)and node.value.startswith('incoming/')and'manifest'in node.value:assert(r/node.value).exists(),node.value
assert 'GLM-fused-'not in driver and 'glm_fused_'not in driver and "q['final_reduced_stats']"in driver and "'reduced_delta':delta"in driver and "d['final_reduced_stats']"in controller
assert "base=r/'candidates/glm-gate-fragment-r9048'"in driver
assert "lib.glm_gate_fragment_policy_r9048.argtypes=[I]"in driver
assert "glm_gate_fragment_counts_r9048(v,2)"in driver
assert "entry_delta[3:]==[1,0,0]"in driver and "a['entry_delta']==[0,0,0,1,0,0]"in controller
assert "report['profile']==0"in driver and "'route_fields_from_original_report'"in driver
assert 'V(decode_out.data_ptr()),0)'in driver
receipt=r/'results/GLM-fragment-layer-r9048-frontend.json';assert not receipt.exists();receipt.write_text(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'build':False,'frontend_and_manifest_paths_and_field_schema':'PASS','DSO_unchanged':m['candidate_DSO_SHA256']}))
exec(compile((i/'launch_GLM_fragment_layer_r9048.py').read_bytes(),'launch_GLM_fragment_layer_r9048.py','exec'))
