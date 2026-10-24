import ast,hashlib,json,pathlib
r=pathlib.Path('.');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for old,new in [('launch_GLM_PP3_constructor_r8781.py','launch_GLM_comm_only_r8783.py'),('watch_GLM_PP3_constructor_r8781.py','watch_GLM_comm_only_r8783.py'),('GLM_PP3_constructor_r8781_inside.sh','GLM_comm_only_r8783_inside.sh')]:
 s=(r/old).read_text().replace('glm_PP3_constructor_r8781.py','GLM_comm_only_r8783.py').replace('GLM_PP3_CONSTRUCTOR_NORMAL','GLM_COMM_ONLY_NORMAL_R8783').replace('GLM-PP3-constructor','GLM-comm-only').replace('GLM_PP3_constructor','GLM_comm_only').replace('r8781','r8783').replace('constructor-8781-','comm-only-8783-')
 if new.startswith('watch_'):
  s=s.replace(";state['capacity_screen']=data['all24_estimated_resident_plus6GiB_capacity_screen']",'').replace('qualified_constructor_only','qualified_comm_only')
  s=s.replace("state['actual_handle']=jobs[0][0]","state['actual_handle']=jobs[0][0];state['GPU_submission']=True")
 (r/new).write_bytes(s.replace('\r\n','\n').encode())
names=['GLM_comm_only_r8783.py','GLM_comm_only_r8783_inside.sh','audit_GLM_comm_only_r8783.py','watch_GLM_comm_only_r8783.py','launch_GLM_comm_only_r8783.py','verify_gpu_idle_r5738.py','verify_gpu_idle_r5819.py']
for n in names:
 p=r/n;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 if p.suffix=='.py':ast.parse(p.read_text(),filename=n)
tree=ast.parse((r/names[0]).read_text())
for n in ast.walk(tree):
 if isinstance(n,ast.Attribute):assert n.attr not in ['modern_glm_model_create','modern_glm_model_forward','modern_glm_model_finish','modern_glm_model_load_batch','modern_glm_model_resident_mask_r5640']
cpu=json.loads((r/'GLM-comm-only-CPU-observed-r8784-a.json').read_text(encoding='utf-8-sig'));assert cpu['build']['completed'] and cpu['rows'][0][2:4]==['COMPLETED','0:0']
m={'card':'GLM53-EXACT-COMM-ONLY-PHASE-CONTROL','scope':'Onlyworld24->TP8->originalcontrol init, beforeTextModel, puretinyCCLcontrol+originaldestroy/shutdown. NoModelparams/weights/resident/forward; exactnativephasejournals.',
 'source_SHA256':{n:sha(r/n) for n in names},'DSO_SHA256':{cpu['build']['library'].replace('/refactor/','',1):cpu['build']['library_SHA256']},'CPU':'910407.184 COMPLETED0/9s','CPU_native_receipt':'GLM-comm-only-CPU-observed-r8784-a.json',
 'resources':{'allocation':910376,'nodes':['a02r4n12','a02r4n13','a02r4n14'],'cards':24,'CPU_per_node':64,'host_memory_M_per_node':240000,'hard_seconds':300,'inside_seconds':260},
 'acceptance':'All24 nativejournal9phases withstage/rank/device/PID/native-monotonic/IDfingerprints; completedtinyTP8/world24SUM; normaldestroy+shutdown;Slurm0/audit/uniqueKFDpost.',
 'failure':'Anyunknownphase/timeout/partialreturnSTOP withoutflaggrid/extension/replay. Notcapacity orperformance.',
 'conditional_constructor':'Notinvokedbythispackage. Onlyafterall24normal mayroot/53 select samecap phaseinstrumented constructor; original528neverreplayed.'}
p=r/'GLM-comm-only-manifest-r8783.json';p.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode());h=sha(p)
dispatch="import pathlib,sys\nsys.argv=['launch_GLM_comm_only_r8783.py',%r]\nexec(compile((pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/launch_GLM_comm_only_r8783.py').read_text(),'launch_GLM_comm_only_r8783.py','exec'))\n"%h
(r/'dispatch_GLM_comm_only_r8783.py').write_bytes(dispatch.encode())
print(json.dumps({'AST_no_Model_calls':'PASS','manifest_SHA256':h,'entry_SHA256':sha(r/'dispatch_GLM_comm_only_r8783.py'),'library_SHA256':cpu['build']['library_SHA256'],'files':names+[p.name,'dispatch_GLM_comm_only_r8783.py']}))
