import pathlib,json,hashlib,sys
sys.path.insert(1,'/refactor/incoming');r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
CPU_SHA='83bd99026decb9f7b38acd4301506854288bfe2887c3265e32b3051f3813c009';DSO_SHA='1760cd0782deed03bee608dda9c7b62690c7f2180ca1038300760596f29a9e63'
def read_private_metadata():
 cp=r/'candidates/GLM-M15-oproj-Model-r125/CPU-build.json';assert sha(cp)==CPU_SHA;c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU_initialized']and not c['dlopen']and c['HOST_TUs_compiled']==3 and c['GPU_TUs_compiled']==0 and c['ABI']==1 and not c['Context_layout_changed']and c['default_OFF']and c['stats_count']==17;assert c['parent_native_SHA256']=='c6506898f7d718e7257253b51bf720ff8fc84ed733d9b7306edd989308d37514';assert sha(c['library'])==c['library_SHA256']==DSO_SHA
 for n,h in c['reused_objects_SHA256'].items():assert sha(n)==h,n
 for f in c['compiled']:assert sha(f['object'])==f['object_SHA256']and sha(f['source'])==f['source_SHA256']and f['fresh_M_equal_MD']
 for n,h in c['actual_binary_SHA256'].items():assert sha(n)==h,n
 return {'cpu_receipt_actual_path':str(cp),'cpu_receipt_SHA256':CPU_SHA,'library_actual_path':c['library'],'library_declared_path':c['library'],'library_SHA256':DSO_SHA,'GPU':False,'dlopen':False,'parent648':c['parent_native_SHA256'],'M15_default_OFF':True,'M15_stats_count':17}
def validate():
 from GLM_transient_preflight_r9059 import validate as parentvalidate
 q=parentvalidate();q['parent648_model_metadata']=q['model_library_metadata'];q['model_library_metadata']=read_private_metadata();return q
if __name__=='__main__':print(json.dumps(validate()))
