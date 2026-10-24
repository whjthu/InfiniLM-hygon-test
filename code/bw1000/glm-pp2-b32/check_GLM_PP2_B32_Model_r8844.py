import ast,json,pathlib,hashlib
r=pathlib.Path('.');d=r/'GLM-PP2-B32-Model-private-r8844';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((d/'GLM-CROSSREQUEST-Model-source-manifest-r8844.json').read_text());g=json.loads((d/'GLM477-frozen-gold-r8844.json').read_text(encoding='utf-8-sig'))
for n,h in m['source_SHA256'].items():
 assert sha(d/n)==h,n
 if n.endswith('.py'):ast.parse((d/n).read_text(encoding='utf-8-sig'),filename=n)
ns={};exec(compile((d/'glm_crossrequest_contract_r8844.py').read_text(),'contract','exec'),ns)
q={'completed':True,'padding_requested':False,'CUDA_initialized':False,'total_effective_input_tokens':16384,'capacity':640,'max_output_tokens':128,'cases':[{'logical_row':i,'input_ids':ids,'input_ids_SHA256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()}for i,ids in enumerate(g['input_ids'])]}
p=ns['validate_fixture'](q);assert len(p)==32 and len(set(map(tuple,p)))==32
for n in(16,32):
 ids=g['generated_ids'][:n];e=g['stopped_at_EOS'][:n]
 assert ns['validate_reference']('I512','R',p[:n],ids,e,128,{},p,{},g)[0]
 wrong=[list(x)for x in ids];wrong[-1][-1]=(wrong[-1][-1]+1)%154880
 assert not ns['validate_reference']('I512','R',p[:n],wrong,e,128,{},p,{},g)[0]
 assert not ns['validate_reference']('I512','R',p[:n],ids,[not e[0]]+e[1:],128,{},p,{},g)[0]
 try:ns['validate_reference']('I512','R',p[:n],ids,e,128,{},p,{},None)
 except AssertionError:pass
 else:raise AssertionError('rebaseline allowed')
probe=(d/'glm_prefill_micro_probe_r8844.py').read_text();tree=ast.parse(probe)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');assert run.args.args[-1].arg=='phase_marker'
assign=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='labels'for t in n.targets));assert ast.literal_eval(assign.value)==m['perf_labels']
assert "len(i512_inputs)==32"in probe and "i512_inputs[:16] if '-B16-'in label else i512_inputs"in probe
base=(d/'glm_crossrequest_resident_base_r8844.py').read_text();assert 'self.chosen=None'in base and '[32,640,8,256]'in base and 'BANK_BYTES=256*1024**2'in base and 'FLOOR=512*1024**2'in base
launch=(d/'launch_GLM_prefill_Model_r8844.py').read_text();inside=(d/'glm_prefill_micro_Model_r8844_inside.sh').read_text()
assert "nodes=['a02r4n12','a02r4n13']"in launch and "'a02r4n12,a02r4n13'"in launch and "'00:35:00'"in launch and '2000 python'in inside
assert 'GLM_CACHE_ROWS_R3627=32'in launch and 'GLM_PP_SIZE_R3603=2'in launch
modern=(d/'modern_glm_prefill_micro_r8844.py').read_text();assert 'candidates/glm-PP2-B32-r8843/CPU-build.json'in modern and "fixture['cases']]*4"not in modern
fb=(d/'glm_continuous_feedback_r8844.py').read_text();assert 'cards==16'in fb and 'ITL_P95_seconds_per_request'in fb and 'ITL_P99_seconds_per_request'in fb and 'maximum_gap_seconds_per_request'in fb
receipt={'CPU_only':True,'GPU':False,'AST_source_SHA256':'PASS','frozen477_32unique_512input':True,'B16subset_and_B32_fullIDs_EOS_positive':True,'token_EOS_negatives_rejected':True,'no_rebaseline':True,'singleload_twoarms':m['perf_labels'],'constructor32_commonmask':True,'newkerneloptsOFF':True,'resources35min2000n12n13':True,'remaining_GPU_gate':'RootreadyGO/sourceclaimpush/SHA/freshidle/uniquewatch','manifest_SHA256':sha(d/'GLM-CROSSREQUEST-Model-source-manifest-r8844.json')}
(r/'GLM-PP2-B32-Model-CPUchecks-r8844.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt))
