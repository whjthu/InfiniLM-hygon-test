import pathlib,json,hashlib,ast,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8868.json';assert sha(mp)=='aea1012248ef32f8e80784e5bdc240ffa683617264f86e003714e16fb6156018';m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():
 p=r/'incoming'/n;assert sha(p)==h,('source',n,sha(p),h)
 if n.endswith('.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for n,h in m['DSO_SHA256'].items():assert sha(r/n)==h,('DSO',n)
sys.path.insert(0,str(r/'incoming'));from glm_model_library_metadata_r8868 import read_model_library_metadata
meta,c=read_model_library_metadata(r,m);assert len(c['compiled'])==7 and len(c['reused_objects_SHA256'])==21
e=r/'incoming/dispatch_GLM_resident_exact_Model_r8868.py';assert sha(e)=='64bc6f34ffb6a17734ac3ef486379330d6e45496dc53f3a4f4bc5e7f069a6fb4'
entry=ast.parse(e.read_bytes());assignment=next(x for x in entry.body if isinstance(x,ast.Assign));binding={'sys':type('Args',(),{})()};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-sealed-entry-argv','exec'),binding)
assert binding['sys'].argv==['launch_GLM_prefill_Model_r8868.py',sha(mp)]
assert (r/'incoming'/binding['sys'].argv[0]).is_file()and sha(r/'incoming'/binding['sys'].argv[0])==m['source_SHA256'][binding['sys'].argv[0]]
assert m['resident_physical_bytes_per_layer']==160*1024**2 and m['maximum_loads']==2 and m['maximum_loads_per_replica']==1 and m['maximum_full128_arms_per_replica']==2 and m['decode_micro_candidates']==[8,16]
print(json.dumps({'staged':True,'GPU':False,'dlopen':False,'manifest_SHA256':sha(mp),'entry_SHA256':sha(e),'actual_metadata':meta,'sources':len(m['source_SHA256'])}))
