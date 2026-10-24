import pathlib,json,hashlib,sys,subprocess,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sys.path.insert(0,str(r/'incoming'));from glm_model_library_metadata_r8855 import read_model_library_metadata
mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8855.json';m=json.loads(mp.read_text());meta,cpu=read_model_library_metadata(r,m);assert len(cpu['compiled'])==1 and len(cpu['reused_objects_SHA256'])==27
q=json.loads((r/'incoming/GLM-exact-budget-Model-CPUchecks-r8855.json').read_text());symbols={};wanted=set(q['actual_consumer_CAPI_names'])
for rel,h in m['DSO_SHA256'].items():
 p=r/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 output=subprocess.check_output(['nm','-D','--defined-only',str(p)],text=True,timeout=30)
 found={s.split()[-1]for s in output.splitlines()if s.split()}
 for name in wanted&found:symbols.setdefault(name,[]).append({'provider':rel,'SHA256':h})
assert set(symbols)==wanted,sorted(wanted-set(symbols))
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'dlopen':False,'actual_metadata':meta,'actual_consumer_symbols':symbols,'symbol_count':len(symbols),'manifest_SHA256':hashlib.sha256(mp.read_bytes()).hexdigest(),'compiled_TUs':1,'reused_objects':27}))
