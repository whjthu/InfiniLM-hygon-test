"""Read-only census of the same existing ctx, before original8GiB admission."""
from pathlib import Path
import ctypes,hashlib,json,os
from .cache_policy_r452 import from_environment,validate_census
BOUNDS=[0,38,78]
LIBRARY='/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/libGLM_V_on_demand_Model_r403.so'
NATIVE_SHA='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
def actual_providers(lib):
 class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.restype=ctypes.c_int
 expected=Path(LIBRARY).resolve();assert hashlib.sha256(expected.read_bytes()).hexdigest()==NATIVE_SHA
 result={}
 for name in ['modern_glm_model_create','modern_glm_model_load_batch','modern_glm_model_finish','modern_glm_model_forward','modern_glm_model_reset','modern_glm_model_destroy','modern_glm_model_shutdown','modern_glm_constructor_census_r4158']:
  i=Info();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(i));got=Path(i.fname.decode()).resolve();assert got==expected,(name,str(got));result[name]=str(got)
 maps=sorted({l.split()[-1]for l in Path('/proc/self/maps').read_text().splitlines()if l.split()[-1].startswith('/')and Path(l.split()[-1]).name==expected.name})
 assert maps and all(Path(p).resolve()==expected for p in maps)
 return dict(library_SHA256=NATIVE_SHA,CAPI_providers=result,actual_mapped_Native=maps)
def constructor_snapshot(lib,ctx,stage,rank,report,report_path=None,provider_reader=actual_providers):
 policy=from_environment(os.environ,stage)
 if report['cache_policy']!=policy.metadata():raise RuntimeError('Actual owner cache policy differs before census')
 binding=provider_reader(lib);assert binding['library_SHA256']==NATIVE_SHA
 report['actual_r403_Native_binding']=binding;report['library_sha256']=NATIVE_SHA;report['PID']=os.getpid();report['parent_PID']=os.getppid()
 report.update(actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=os.environ['GLM_SOURCE_MANIFEST_SHA'],run=os.environ['GLM_RUN_R5040'])
 fn=lib.modern_glm_constructor_census_r4158;fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_char_p
 raw=fn(ctx)
 if not raw:raise RuntimeError('Original unloaded tensor census returned null')
 c=json.loads(raw)
 validate_census(c,policy)
 report.update(constructor_cache_census=c,constructor_cache_verified=True,constructor_noV_verified=bool(policy.noV_flag),Native_loaded_by_census=False,constructor_calls_expected=1)
 if report_path is None:
  report_path=Path('/refactor/results')/('GLM-r404-constructor-'+os.environ['GLM_RUN_R5040']+'-stage%d-rank%d.json'%(stage,rank))
 p=Path(report_path);payload=dict(run=os.environ['GLM_RUN_R5040'],actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],stage=stage,rank=rank,Native_SHA256=report['library_sha256'],actual_binding=binding,source_manifest_SHA256=os.environ['GLM_SOURCE_MANIFEST_SHA'],context_policy=policy.policy,cache_policy=policy.metadata(),census=c,loaded=False)
 with p.open('x')as out:json.dump(payload,out,indent=2);out.write('\n')
 report['constructor_snapshot_path']=str(p);report['constructor_snapshot_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
 return c
