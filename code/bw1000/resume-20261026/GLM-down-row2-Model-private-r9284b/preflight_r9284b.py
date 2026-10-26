import pathlib,json,hashlib,sys
sys.path.insert(1,'/refactor/incoming');r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read_private_metadata():
 cp=r/'candidates/GLM-down-row2-Model-r9284b/CPU-build.json';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and c['HOST_TUs_compiled']==1 and c['GPU_TUs_compiled']==0 and c['Frame_ABI']==8 and c['irregular739_NOTlinked']and c['parent_SHA256']=='a549b9b3951b51b52d3f8a96f0fe585da6a92cb88e77054970fdbef3ad483078';assert sha(c['library'])==c['library_SHA256'];assert all(sha(f)==h for f,h in {**c['reused_objects_SHA256'],**c['reused_qualified_GPU_object_SHA256']}.items());assert all(sha(f['object'])==f['object_SHA256']and sha(f['source'])==f['source_SHA256']for f in c['compiled'])
 return {'cpu_receipt_actual_path':str(cp),'cpu_receipt_SHA256':sha(cp),'library_actual_path':c['library'],'library_declared_path':c['library'],'library_SHA256':c['library_SHA256'],'GPU':False,'dlopen':False,'parent648':c['parent_SHA256']}
def validate():
 from GLM_transient_preflight_r9059 import validate as parentvalidate
 q=parentvalidate();q['parent648_model_metadata']=q['model_library_metadata'];q['model_library_metadata']=read_private_metadata();return q
if __name__=='__main__':print(json.dumps(validate()))
