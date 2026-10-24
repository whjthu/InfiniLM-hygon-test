import hashlib,json,os,pathlib,shlex,subprocess,time
r=pathlib.Path('/refactor');b=r/'candidates/glm-PP3-Model-r8801';assert not b.exists();b.mkdir();begin=time.monotonic();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(args,name):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=240);(b/(name+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((name,p.returncode,p.stderr[-6000:]))
 return p.stdout
try:
 parent=json.loads((r/'candidates/glm-constructor-only-r8791/CPU-build.json').read_text());assert parent['completed'] and sha(parent['library'])==parent['library_SHA256']=='59d65034a92bf4065d3d1b979835935e54d059f61742e73f5127a11712ed8ca1'
 source=r/'incoming/modern_glm_PP3_Model_bridge_r8800.cpp';meta=json.loads((r/'incoming/GLM-PP3-Model-bridge-source-r8800.json').read_text());assert sha(source)==meta['source_SHA256']
 original=r/'candidates/glm-constructor-only-r8791/modern_glm_constructor_bridge_r8790.cpp';assert sha(original)==meta['base_SHA256'] and source.read_text()==original.read_text().replace('c->pp!=2','c->pp!=3').replace('5640 ready/PP2/cap640/trace','5640 ready/PP3/cap640/trace')
 old=r/'candidates/glm-constructor-only-r8791/bridge.o';cmd=next(list(v) for v in parent['commands'] if '-c' in v and '-o' in v and v[v.index('-o')+1]==str(old))
 link=next(list(v) for v in parent['commands'] if '-shared' in v);objects={v:sha(v) for v in link if v.endswith('.o')};assert len(objects)==28 and str(old) in objects
 src=b/'modern_glm_PP3_Model_bridge_r8800.cpp';src.write_bytes(source.read_bytes());obj=b/'bridge.o';dep=b/'bridge.d'
 cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(obj)
 if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(dep)
 run(cmd,'compile-bridge');lib=b/'libglm_PP3_Model_r8801.so';link=[str(obj) if v==str(old) else v for v in link];link[link.index('-o')+1]=str(lib);run(link,'link')
 symbols=run(['nm','-D','--defined-only',str(lib)],'exports')
 assert all(name in symbols for name in ['modern_glm_model_create','modern_glm_model_destroy','modern_glm_comm_only_create_r8783','modern_glm_comm_only_destroy_r8783','modern_glm_constructor_only_create_r8790','modern_glm_constructor_only_destroy_r8790'])
 for path,h in objects.items():assert sha(path)==h
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source_SHA256=sha(src),object_SHA256=sha(obj),parent_library_SHA256=parent['library_SHA256'],replaced_object=str(old),reused_objects_SHA256={p:h for p,h in objects.items() if p!=str(old)},scope='Onlyonebridge hostTU+27originalobjects; originalcreate preserved; noSDK/GPU/nativekernelcompile, noModelparams/weights/forward')
except BaseException as exc:q['error']=repr(exc);raise
finally:q['seconds']=time.monotonic()-begin;(b/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({k:q.get(k) for k in ['completed','producer','seconds','library_SHA256','error']}))
