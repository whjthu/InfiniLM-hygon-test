import pathlib,subprocess,json,ast,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=json.loads((i/'GLM-M16-cohort-layer-manifest-r9056.json').read_bytes())
assert subprocess.run(['bash','-n',str(i/'GLM_M16_cohort_layer_r9056_inside.sh')]).returncode==0
for n,h in m['sources_SHA256'].items():
 p=i/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 if n.endswith('.py'):compile(p.read_bytes(),n,'exec')
d=(i/'GLM_M16_cohort_layer_r9056.py').read_text();c=(i/'controller_GLM_M16_cohort_layer_r9056.py').read_text()
for node in ast.walk(ast.parse(d)):
 if isinstance(node,ast.Constant)and isinstance(node.value,str)and node.value.startswith('incoming/')and'manifest'in node.value:assert(r/node.value).exists(),node.value
assert "base=r/'candidates/glm-M16-cohort-r9056b'"in d
assert 'glm_attention_cohort_policy_r9054(rank,mode)'in d and 'glm_attention_cohort_counts_r9054(rank,v,4)'in d
assert 'prefill_desc=[(Z*24)(*[v for row in range(start,start+8)for v in [row,0,512]])for start in [0,8]]'in d
assert 'for cohort in range(2):ck(lib.glm_layer_forward_r5670(ctx,4096,8,prefill_desc[cohort]'in d
assert 'V(x[cohort*4096].data_ptr()),V(pos[cohort*4096].data_ptr()),V(out[cohort*4096].data_ptr())'in d
assert d.count('glm_layer_reset_r5670(ctx)')==2
assert 'ctx,16,16,decode_desc' in d and "report['N']==16 and report['B']==16"in d
assert 'grouped_delta==[2,0,0,2,0]'in d and 'vector_delta==[0,1,0,1,0,0,0,0]'in d
assert "entry_delta[3:]==([0,1,0]if mode else[1,0,0])"in d
assert "q['final_reduced_stats']==[6,5,0]"in d and "d['final_reduced_stats']==[6,5,0]"in c
assert "report['profile']==0"in d and 'V(decode_out.data_ptr()),0)'in d
out=r/'results/GLM-M16-cohort-layer-r9056-frontend.json';assert not out.exists();out.write_text(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'build':False,'actualdriver_frontend':'PASS','prefill_stable_rows':[list(range(8)),list(range(8,16))],'no_reset_between_prefix_cohorts':True,'DSO':m['candidate_DSO_SHA256'],'logical_request_checks_per_TP_replica':176,'expected_actual_R128_original_fallback':True}))
exec(compile((i/'launch_GLM_M16_cohort_layer_r9056.py').read_bytes(),'launch_GLM_M16_cohort_layer_r9056.py','exec'))
