import ast,datetime,hashlib,json,pathlib
r=pathlib.Path('.');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['glm_PP3_constructor_r8781.py','GLM_PP3_constructor_r8781_inside.sh','audit_GLM_PP3_constructor_r8781.py','watch_GLM_PP3_constructor_r8781.py','launch_GLM_PP3_constructor_r8781.py','verify_gpu_idle_r5738.py','verify_gpu_idle_r5819.py']
for name in names:
    p=r/name;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=name)
tree=ast.parse((r/names[0]).read_text())
for node in ast.walk(tree):
    if isinstance(node,ast.Attribute):
        assert node.attr not in ['modern_glm_model_forward','modern_glm_model_finish','modern_glm_model_load_batch','modern_glm_model_resident_mask_r5640']
    if isinstance(node,ast.Name):assert node.id not in ['safe_open','AutoTokenizer']
m={'card':'ROOT_FINITE_PP3_CONSTRUCTOR_ONLY_AFTER8780_STOP','GPU_scope':'One24owner unloadedconstructor; no checkpoint tensor I/O/finish/forward/resident. No performance result.',
   'source_SHA256':{n:sha(r/n) for n in names},'DSO_SHA256':{'candidates/glm-PP3-B64-r8780/libglm_PP3_B64_r8780.so':'99b99b3ab56de735cbff177df532af0a7ce4836d6af25ce5109dfc5c997d4862'},
   'resources':{'allocation':910376,'nodes':['a02r4n12','a02r4n13','a02r4n14'],'cards_per_node':8,'CPU_per_node':64,'hostmem_M_per_node':240000,'hard_seconds':300,'inside_seconds':260},
   'frozen_CPU_receipt':'GLM-PP3-CPU-observed-r8780-a.json','CPU_handle':'910407.182','bounds':[0,26,54,78],'rows':64,'capacity':640,
   'omitted_runtime_guard':'Resident maskCAPI remainsPP2-only and READY-only; not called. Requires mechanicalrepair before any futurePP3 Model, not silently bypassed.',
   'admission_screen':'Afteractualconstruct+trim+tinyCCL, compareactualfree withestimatedresident512MiB*localLayerCount+6GiB+64MiB. This doesnotallocate resident or proveprefill scratchpeak.',
   'close':'All24normal destroy+shutdown; originalunique watcher waits actualSlurmterminal, validates24reports and executesfinallyunique3nodeKFDpost evenauditfails.',
   'no_replay':True,'Model_or_throughput_adoption':False}
path=r/'GLM-PP3-constructor-manifest-r8781.json';path.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode())
h=sha(path)
dispatch="import pathlib,sys\nsys.argv=['launch_GLM_PP3_constructor_r8781.py',%r]\nexec(compile((pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/launch_GLM_PP3_constructor_r8781.py').read_text(),'launch_GLM_PP3_constructor_r8781.py','exec'))\n"%h
(r/'dispatch_GLM_PP3_constructor_r8781.py').write_bytes(dispatch.encode())
claim={'owner':'bw1000-agent','execution_owner':'/root/moe_decode_arithmetic_audit','work_item':'measure-glm-pp3-unloaded-capacity','state':'ROOT_CONTINUOUS_CONSTRUCTOR_ONLY_GO_NOT_SUBMITTED','updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'source':'code/bw1000/glm-pp3-constructor/GLM-PP3-constructor-manifest-r8781.json','manifest_SHA256':h,'entry':'dispatch_GLM_PP3_constructor_r8781.py','entry_SHA256':sha(r/'dispatch_GLM_PP3_constructor_r8781.py'),
       'CPU':'910407.182 COMPLETED0/48s;6privatehostTU+22reusedobjects, noSDK/GPUcompile','DSO_SHA256':m['DSO_SHA256'],
       'scope':m['GPU_scope'],'resources':m['resources'],'qualification':m['close'],'Model_adoption':False,'prior_card_STOP':'GLM-PP3-B64-CPU-closure-r8780.json'}
(r/'GLM-PP3-constructor-claim-r8781.json').write_bytes((json.dumps(claim,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({'AST':'PASS_noGPUinit','manifest_SHA256':h,'entry_SHA256':claim['entry_SHA256'],'files':names+['GLM-PP3-constructor-manifest-r8781.json','dispatch_GLM_PP3_constructor_r8781.py']}))
