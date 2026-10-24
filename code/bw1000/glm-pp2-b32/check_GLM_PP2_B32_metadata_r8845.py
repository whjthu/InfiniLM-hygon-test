import pathlib,json,hashlib,re,sys,copy
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sys.path.insert(0,str(r/'incoming'))
from glm_model_library_metadata_r8845 import read_model_library_metadata
mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8845.json';m=json.loads(mp.read_text());meta,c=read_model_library_metadata(r,m)
old=(r/'incoming/modern_glm_prefill_micro_r8844.py').read_text();values=set(re.findall(r"assert report\['library_sha256'\]=='([0-9a-f]{64})'",old));assert len(values)==1
wrong=copy.deepcopy(m);wrong['model_SHA256']=values.pop()
try:read_model_library_metadata(r,wrong)
except AssertionError as e:assert str(e)=='CPU_receipt_manifest_ModelSHA_mismatch';negative=str(e)
else:raise AssertionError('old Model hash accepted')
assert len(c['compiled'])==6 and len(c['reused_objects_SHA256'])==22
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'dlopen':False,'manifest_SHA256':hashlib.sha256(mp.read_bytes()).hexdigest(),'actual_metadata':meta,'actual_CPU_receipt_keys':sorted(c),'CPU_compiled':len(c['compiled']),'CPU_reused':len(c['reused_objects_SHA256']),'old_hash_negative':negative,'scope':'same pure metadata function used by CPU preflight before preengine CUDA and Modelcreate; actual host receipt/ELF bytes read, not AST-only'}))
