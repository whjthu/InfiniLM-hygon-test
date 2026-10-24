import pathlib,json,hashlib,ast
r=pathlib.Path('.');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,d:p.write_text(json.dumps(d,indent=2),encoding='utf-8',newline='\n')
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r8701.json').read_text());sources={}
for n,h in m['source_SHA256'].items():
 if n in ['glm_prefill_window_observer_r8701.cpp','glm_prefill_window_model_r8701.cpp','build_GLM_prefill_window_r8701.py','GLM_prefill_window_CPU_r8701_inside.sh','GLM-prefill-window-CPU-r8701-manifest.json']:continue
 target=n.replace('8701','8708')
 if target==n:assert sha(r/n)==h,n
 if target.endswith('.py'):ast.parse((r/target).read_text())
 sources[target]=sha(r/target)
sources['dispatch_GLM_prefix_Model_r8708.py']=sha(r/'dispatch_GLM_prefix_Model_r8708.py')
side=json.loads((r/'GLM-transient-Model-observer-manifest-r8701.json').read_text());side['source_SHA256']={n.replace('8701','8708'):sha(r/n.replace('8701','8708'))for n in side['source_SHA256']};put(r/'GLM-transient-Model-observer-manifest-r8708.json',side)
m['DSO_SHA256'].pop('candidates/glm-prefill-window-r8701/libglm_prefill_window_r8701.so');m['DSO_SHA256']['candidates/glm-prefix-bridge-r8705/libglm_prefix_bridge_r8705.so']='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
m.update(source_SHA256=sources,model_SHA256='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52',root_GO='ROOT_CONTINUOUS_FINITE_MODEL_GO_AFTER519_COMPONENT',observer_manifest_SHA256=sha(r/'GLM-transient-Model-observer-manifest-r8708.json'),CPU_native_receipt={'path':'GLM-prefill-prefix-CPU-observed-r8705-a.json','SHA256':sha(r/'GLM-prefill-prefix-CPU-observed-r8705-a.json')},maximum_full128_arms=4,perf_labels=['perf-I512-proof-O-r0','perf-I512-proof-N-r0','perf-I512-proof-N-r1','perf-I512-proof-O-r1'],scope='One load genuineTP8PP2/B16M8/I512O128 proofOFF/ON ordinaryprofileOFF; initialON full128 then optionalNO full128 byactualremainingtime>2max(pairwall)+240sQAreserve. Samegold/EOS/QA/fullIDs/originalwatchauditfinallypost. n14+n15/c64each/240000Meach/60minhard/3500inside.',math='CPU169 DSO derived from8703 proof2TU+identitybridge1TU; serving Model math unchanged; onlyCPUplan provenance. Existing1af0 helper/SDK dependencies frozen.',CPU_contracts={'actualCPU':'910407.169 onebridgeTU/27reusedobjects over successful8703','Model_constructor':'genuineTP8PP2 stages38/40; componentbridge unused','ordinary_profileOFF':True},prefill_micro_candidates=[8],fixed_ordinary_arm_count='2 mandatory, +2 conditional budget',diagnostic_budget='NO observer window/profile; actual remaining processbudget after firstON pair requires >2max(pairwall)+240 forNO repeat',scope_hit='Record prefix counters eachModelowner/arm; N>0/O0; no mid-bit/fullnetwork gate.')
put(r/'GLM-CROSSREQUEST-Model-source-manifest-r8708.json',m)
print(sha(r/'GLM-CROSSREQUEST-Model-source-manifest-r8708.json'),len(sources))
