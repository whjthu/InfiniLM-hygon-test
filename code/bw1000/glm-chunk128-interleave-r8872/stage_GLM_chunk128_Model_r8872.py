import pathlib,json,hashlib,ast,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8872.json';assert sha(mp)=='b715ba53edde30192fdd107f94d6b325aafa534ff2729c1e99052f7db9eb73b1';m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():
 p=r/'incoming'/n;assert sha(p)==h,('source',n,sha(p),h)
 if n.endswith('.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for n,h in m['DSO_SHA256'].items():assert sha(r/n)==h,('DSO',n)
sys.path.insert(0,str(r/'incoming'));from glm_model_library_metadata_r8872 import read_model_library_metadata
meta,c=read_model_library_metadata(r,m);assert len(c['compiled'])==7 and len(c['reused_objects_SHA256'])==21
e=r/'incoming/dispatch_GLM_resident_exact_Model_r8872.py';assert sha(e)=='75a4858765fb8507299676aab171d1d7e446e4fd12a7a308916b89d365318a8e'
assert m['resident_physical_bytes_per_layer']==160*1024**2 and m['maximum_loads']==1 and m['maximum_full128_arms']==2 and m['performance_micro']==[8]
print(json.dumps({'staged':True,'GPU':False,'dlopen':False,'manifest_SHA256':sha(mp),'entry_SHA256':sha(e),'actual_metadata':meta,'sources':len(m['source_SHA256'])}))

launcher=r/'incoming/launch_GLM_prefill_Model_r8872.py';tree=ast.parse(launcher.read_text());a=next(x.value for x in ast.walk(tree)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='args'for t in x.targets));argv=ast.literal_eval(a);assert argv[argv.index('-t')+1]=='00:45:00' and argv[argv.index('--mem=240000M')]=='--mem=240000M' and argv[argv.index('-w')+1]=='a02r4n12,a02r4n13';assert 'WAVE_INSIDE_R2323=glm_prefill_micro_Model_r8872_inside.sh'in argv;inside=(r/'incoming/glm_prefill_micro_Model_r8872_inside.sh').read_text();assert '2600' in inside and 'modern_glm_prefill_micro_r8872.py'in inside;assert 'from glm_prefill_micro_probe_r8872 import run' in(r/'incoming/modern_glm_prefill_micro_r8872.py').read_text();assert m['active']==16 and m['maximum_full128_arms']==2 and m['prefill_token_chunks']==[512,128];print(json.dumps({'actual_launch_inside_consumer_binding':True,'GPU':False,'dlopen':False}))
