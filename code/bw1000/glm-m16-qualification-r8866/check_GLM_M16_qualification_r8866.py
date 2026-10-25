import pathlib,json,ast,hashlib
r=pathlib.Path(__file__).resolve().parent;d=r/'GLM-M16-Model-private-r8866';old=r/'GLM-M16-Model-private-r8862';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8866.json';m=json.loads(mp.read_bytes());p=json.loads((old/'GLM-CROSSREQUEST-Model-source-manifest-r8862.json').read_bytes())
for n,h in m['source_SHA256'].items():
 assert sha(d/n)==h,n
 if n.endswith('.py'):ast.parse((d/n).read_bytes(),filename=n)
assert m['DSO_SHA256']==p['DSO_SHA256']and m['model_SHA256']==p['model_SHA256']=='68bdbddc918063dee48b4da21bb3ca24a55abdc4a4a3c0d7c415e3e167de100e'
assert m['cpu_receipt_path']==p['cpu_receipt_path']and m['CPU_native_receipt']==p['CPU_native_receipt']
changed=[]
for n in m['source_SHA256']:
 orig=n.replace('r8866','r8862')
 if (d/n).read_bytes().replace(b'r8866',b'r8862')!=(old/orig).read_bytes():changed.append(n)
assert set(changed)=={'glm_crossrequest_control_r8866.py','glm_prefill_micro_probe_r8866.py','modern_glm_prefill_micro_r8866.py','GLM-transient-Model-observer-manifest-r8866.json'}
entry=d/'dispatch_GLM_resident_exact_Model_r8866.py';t=ast.parse(entry.read_bytes());assignment=next(x for x in t.body if isinstance(x,ast.Assign));ns={'sys':type('S',(),{})()};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-entry-argv','exec'),ns);assert ns['sys'].argv==['launch_GLM_prefill_Model_r8866.py',sha(mp)]
launcher=ast.parse((d/'launch_GLM_prefill_Model_r8866.py').read_bytes());a=next(x for x in launcher.body if isinstance(x,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='args'for v in x.targets));env={};exec(compile(ast.Module(body=[a],type_ignores=[]),'actual-launcher-resource-list','exec'),env);args=env['args'];assert all(x in args for x in('--jobid=910376','-N2','-n2','-c64','a02r4n12,a02r4n13','--gres=dcu:8','--mem=240000M','00:45:00','GLM_CACHE_ROWS_R3627=32','GLM_CACHE_CAP_R3627=640'))
assert b'timeout --kill-after=20s 2600 python -u /refactor/incoming/modern_glm_prefill_micro_r8866.py'in(d/'glm_prefill_micro_Model_r8866_inside.sh').read_bytes()
pt=ast.parse((d/'glm_prefill_micro_probe_r8866.py').read_bytes());run=next(x for x in pt.body if isinstance(x,ast.FunctionDef)and x.name=='run');lab=next(x for x in run.body if isinstance(x,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='labels'for v in x.targets));env={};exec(compile(ast.Module(body=[lab],type_ignores=[]),'actual-labels','exec'),env);assert env['labels']==m['perf_labels']==['perf-I512-B32-M8-N-r0','perf-I512-B32-M16-N-r0'];assert m['maximum_loads']==1 and m['maximum_full128_arms']==2
for k,v in [('resident_physical_bytes_per_layer',160*1024**2),('reserve_bytes',6*1024**3),('allowance_bytes',64*1024**2),('runtime_floor_bytes',512*1024**2)]:assert m[k]==v
checks=json.loads((r/'GLM564-consumer-repair-CPUchecks-r8866.json').read_bytes());assert checks['passed']and checks['actual_override_not_base_only']
out={'CPU_only':True,'GPU':False,'dlopen':False,'nativecompile':False,'passed':True,'manifest_SHA256':sha(mp),'entry_SHA256':sha(entry),'DSO_SHA256':m['model_SHA256'],'actual_override_CPUcheck_SHA256':sha(r/'GLM564-consumer-repair-CPUchecks-r8866.json'),'semantic_changes_only':['actual finish includes vector16','existing durable generation row refresh/atomic fsync before finish','success-only completion/QA scope'],'other_changes':'private namespace hashes/observer and worker scope text','old564_unchanged':True,'entry_actual_argv_manifest_match':True,'actual_launcher_resources':args,'full128_arms':env['labels'],'budget':'45minhard2600inner; original564 ended2108s after both generations;492s historical remainder for QA/shutdown;varianceUNKNOWN;no repeat/retry/extension','sourceclaim_and_root_final_GPU_GO':'PENDING'}
(r/'GLM-M16-qualification-CPUchecks-r8866.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in out.items()if k!='actual_launcher_resources'}))
