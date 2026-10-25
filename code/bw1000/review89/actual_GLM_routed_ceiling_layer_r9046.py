import pathlib,subprocess,json,ast,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming'
assert subprocess.run(['bash','-n',str(i/'GLM_routed_ceiling_layer_r9046_inside.sh')]).returncode==0
m=json.loads((i/'GLM-routed_ceiling-layer-manifest-r9046.json').read_bytes())
for n,h in m['sources_SHA256'].items():
 p=i/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 if n.endswith('.py'):compile(p.read_bytes(),n,'exec')
driver=(i/'GLM_routed_ceiling_layer_r9046.py').read_text();controller=(i/'controller_GLM_routed_ceiling_layer_r9046.py').read_text()
for node in ast.walk(ast.parse(driver)):
 if isinstance(node,ast.Constant)and isinstance(node.value,str)and node.value.startswith('incoming/')and'manifest'in node.value:assert(r/node.value).exists(),node.value
assert "base=r/'candidates/glm-routed-ceiling-r9046'"in driver
assert 'glm_routed_ceiling_counts_r9046(v,3)'in driver and 'return [v[0]+v[2],v[1],0]'in driver
assert 'glm_routed_ceiling_policy_r9046(2 if index==0 else mode)'in driver
assert 'glm_gate_lookahead_policy_r9042(0)'in driver
assert "'posttimer_fixture_validation_ms':fixture_validation_ms"in driver and "list(fixture)==[1,1,98816]"in driver
assert driver.index('wall=(time.perf_counter()-tick)*1000')<driver.index('ck(lib.glm_routed_ceiling_validate_r9046')
assert 'glm_routed_ceiling_policy_r9046(-1)'in driver and "a['fixed_fixture_x_and_ordered_ids_exact']"in controller
assert "report['profile']==0"in driver and 'V(decode_out.data_ptr()),0)'in driver
receipt=r/'results/GLM-routed_ceiling-layer-r9046-frontend.json';assert not receipt.exists();receipt.write_text(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'build':False,'frontend_manifest_and_counter_schema':'PASS','DSO':m['candidate_DSO_SHA256'],'fixed_fixture_validation':'posttimer originalowner x and orderedindices98816B; zero D2D alias'}))
exec(compile((i/'launch_GLM_routed_ceiling_layer_r9046.py').read_bytes(),'launch_GLM_routed_ceiling_layer_r9046.py','exec'))
