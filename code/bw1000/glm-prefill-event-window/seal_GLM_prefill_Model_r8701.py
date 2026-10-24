import ast,hashlib,json,pathlib
r=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=lambda p,q:p.write_bytes((json.dumps(q,indent=2)+'\n').encode())
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r5848.json').read_text());old=m['source_SHA256'];sources={}
for n,h in old.items():
 new=n.replace('5848','8701')
 if new==n:assert sha(r/n)==h,('shared_source_changed',n)
 if new.endswith('.py'):ast.parse((r/new).read_text())
 sources[new]=sha(r/new)
for n in ['glm_prefill_window_observer_r8701.cpp','glm_prefill_window_model_r8701.cpp','build_GLM_prefill_window_r8701.py','GLM_prefill_window_CPU_r8701_inside.sh','GLM-prefill-window-CPU-r8701-manifest.json']:sources[n]=sha(r/n)
sources['launch_GLM_prefill_Model_r8701.py']=sha(r/'launch_GLM_prefill_Model_r8701.py')
side=json.loads((r/'GLM-transient-Model-observer-manifest-r5848.json').read_text());side['source_SHA256']={n.replace('5848','8701'):sha(r/n.replace('5848','8701'))for n in side['source_SHA256']};write(r/'GLM-transient-Model-observer-manifest-r8701.json',side)
m.update(source_SHA256=sources,model_SHA256='fe179cf20aba957b2b6dbd53349dd10dbf650ef9c2ef7a0e70fb843cfc36ef6f',observer_manifest_SHA256=sha(r/'GLM-transient-Model-observer-manifest-r8701.json'),root_GO='PENDING_ROOT_REVIEW_NO_MODEL_GPU',scope='OneModelload/2TUeventobserver only. O8B16M8full128baseline then sameO8full128 diagnostic: stage0cohort0layer0/cohort1layer1; fullgold/QA/rawCPUargmax/all16normal/watchpost. NoGPUbusy or speedadoption. Node14/15 c64/240000Meach hard60min; goal45minfirstdecision notguaranteed.',CPU_native_receipt={'path':'GLM-prefill-window-CPU-observed-r8701-a.json','SHA256':sha(r/'GLM-prefill-window-CPU-observed-r8701-a.json')},maximum_loads=1,maximum_full128_arms=2,perf_labels=['perf-I512-O8-baseline','prof-I512-O8-diag'],diagnostic_budget='Allowner vote beforediag requires actualremaining3500sprocessbudget >2xmeasuredbaselinewall+180sQAreserve. Else skipdiag normallyclose/QA/originalpost; never qualifiesobservationwithout16scopes.',math='Sameparent1af0/26objects kept, only2window/modelscopeTUs replaced; headerABI/SDK/freshness/stream/sync unchanged.',scope_hit='CPUcompile isnotproof. GPU requires8stage0owners×2actualscopes withstage/layer/mode/row/past/count/matched255 andevents<=512; stage1zero. Auditoutputsactualmaxadjacent stream/host/CPUinterval andspecificcallchaincandidate.')
oldpath='candidates/glm-shared-validation-model-r5817/libglm_shared_validation_model_r5817.so';m['DSO_SHA256'].pop(oldpath);m['DSO_SHA256']['candidates/glm-prefill-window-r8701/libglm_prefill_window_r8701.so']=m['model_SHA256']
write(r/'GLM-CROSSREQUEST-Model-source-manifest-r8701.json',m)
print(json.dumps({'source_files':len(sources),'Model_manifest_SHA256':sha(r/'GLM-CROSSREQUEST-Model-source-manifest-r8701.json'),'observer_manifest_SHA256':m['observer_manifest_SHA256']}))
