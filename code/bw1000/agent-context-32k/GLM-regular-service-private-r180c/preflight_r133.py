import pathlib,json,hashlib,os,sys
sys.path.insert(1,'/refactor/incoming');r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read_private_metadata():
 cp=r/'candidates/GLM-critical-observer-r131/CPU-build.json';pin='8c478b86c887d8f235c3eaa8ee9cd369d053112bc5fb437eb5bd0a8bb9f4a88d';assert sha(cp)==pin;c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU_initialized']and not c['dlopen']and c['HOST_TUs_compiled']==4 and c['GPU_TUs_compiled']==0 and c['ABI']==1 and not c['Context_layout_changed']and c['default_OFF']and c['events_per_owner']==512 and c['new_synchronization_calls']==c['math_changes']==c['schedule_changes']==0;assert c['parent_native_SHA256']=='1760cd0782deed03bee608dda9c7b62690c7f2180ca1038300760596f29a9e63';assert sha(c['library'])==c['library_SHA256']=='ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'
 for n,h in c['reused_objects_SHA256'].items():assert sha(n)==h,n
 for f in c['compiled']:assert sha(f['object'])==f['object_SHA256']and sha(f['source'])==f['source_SHA256']and f['fresh_M_equal_MD']and f['full_semantic_argv_replayed']
 for n,h in c['actual_binary_SHA256'].items():assert sha(n)==h,n
 return {'cpu_receipt_actual_path':str(cp),'cpu_receipt_SHA256':pin,'library_actual_path':c['library'],'library_declared_path':c['library'],'library_SHA256':c['library_SHA256'],'GPU':False,'dlopen':False,'parent648':c['parent_native_SHA256'],'M15_default_OFF':True,'M15_stats_count':17,'observer_default_OFF':True}
def validate():
 from GLM_transient_preflight_r9059 import validate as parentvalidate
 q=parentvalidate();q['parent648_model_metadata']=q['model_library_metadata'];q['model_library_metadata']=read_private_metadata();return q
if __name__=='__main__':print(json.dumps(validate()))
