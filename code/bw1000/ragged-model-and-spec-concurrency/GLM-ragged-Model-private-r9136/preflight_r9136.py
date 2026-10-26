import pathlib,json,hashlib,sys
sys.path.insert(1,'/refactor/incoming')
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read_private_metadata():
 cp=r/'candidates/GLM-ragged-Model-r9136/CPU-build.json';assert sha(cp)=='f842a71ebb81a9dca6e4e985218646098ac3734fa4bb56a5fd09a47ba74b692b';c=json.loads(cp.read_text());assert c['completed']and c['CPU_only']and c['HOST_TUs_compiled']==2 and c['GPU_TUs_compiled']==0 and len(c['reused_objects_SHA256'])==26 and c['parent_SHA256']=='fbd07490dd9887868e4a6f5303170969dfa188e9cbcb39f7dca708b3a303741a';assert sha(c['library'])==c['library_SHA256']=='ffdcaedce904efa8b7cb4dd1f8ba143898369b765418b2ebd036bb66475573e0';assert all(sha(f)==h for f,h in c['reused_objects_SHA256'].items());assert c['slot_header_SHA256']=='80098ca2c6805d4fb4afe1104e2a4da65c1ceb8f98b58556908a5545eadae40a'
 return {'cpu_receipt_actual_path':str(cp),'cpu_receipt_SHA256':sha(cp),'library_actual_path':c['library'],'library_declared_path':c['library'],'library_SHA256':c['library_SHA256'],'GPU':False,'dlopen':False,'parent648':c['parent_SHA256']}
def validate():
 from GLM_transient_preflight_r9059 import validate as parentvalidate
 q=parentvalidate();q['parent648_model_metadata']=q['model_library_metadata'];q['model_library_metadata']=read_private_metadata();return q
if __name__=='__main__':print(json.dumps(validate()))
