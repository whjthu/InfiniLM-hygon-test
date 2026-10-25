import pathlib,json,hashlib,ast,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8866.json';assert sha(mp)=='57f0e164d8daa6dac15c36afc85ddf620c52faccdb55da7fcb31e5c8e72b5432';m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():
 p=r/'incoming'/n;assert sha(p)==h,('source',n,sha(p),h)
 if n.endswith('.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for n,h in m['DSO_SHA256'].items():assert sha(r/n)==h,('DSO',n)
sys.path.insert(0,str(r/'incoming'));from glm_model_library_metadata_r8866 import read_model_library_metadata
meta,c=read_model_library_metadata(r,m);assert len(c['compiled'])==7 and len(c['reused_objects_SHA256'])==21
e=r/'incoming/dispatch_GLM_resident_exact_Model_r8866.py';assert sha(e)=='d3da03487ab1eac39a9e18931089caf1230f762e692bfaf5fc6327442883b8c3'
assert m['resident_physical_bytes_per_layer']==160*1024**2 and m['maximum_loads']==1 and m['maximum_full128_arms']==2 and m['performance_micro']==[8,16]
print(json.dumps({'staged':True,'GPU':False,'dlopen':False,'manifest_SHA256':sha(mp),'entry_SHA256':sha(e),'actual_metadata':meta,'sources':len(m['source_SHA256'])}))
