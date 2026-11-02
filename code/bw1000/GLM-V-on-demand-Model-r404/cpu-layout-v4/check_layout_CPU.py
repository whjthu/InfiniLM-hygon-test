"""Consume the real production purepreengine and eight-argument create wrapper.

Only the subsequent Torch import and actual CABI create are explicit leaves.
No preengine module substitution, no fake device/Native qualification.
"""
from pathlib import Path
import ast,base64,builtins,ctypes,hashlib,importlib.util,json,os,sys,types
from unittest.mock import patch
D=Path(__file__).resolve().parent;O=D.parent/'owner_v2';sys.path.insert(0,str(O))
from topology_r404 import read_layout,create_context,validate_layout
import preengine_r133 as production_preengine
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 cfgpath=Path('/old/models/ZhipuAI-GLM-5.3/config.json');raw=cfgpath.read_bytes();cfg=json.loads(raw);records=[];device_boundaries=[]
 model=ast.parse((O/'model_entry_r290.py').read_text());worker=next(n for n in model.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
 calls=[n for n in ast.walk(worker)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='create_context'];assert len(calls)==1
 assert [ast.unparse(x)for x in calls[0].args]==['lib','rank','stage','ROWS','CAPACITY','cfg','payload']
 assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_glm_model_create'for n in ast.walk(worker))
 entry_lines=(O/'model_entry_r290.py').read_text().splitlines();assert next(i for i,x in enumerate(entry_lines)if '_layout_before_device = 'in x)<next(i for i,x in enumerate(entry_lines)if '_protocol_preweight = 'in x)
 old=ast.parse((D.parent/'owner/preengine_r133.py').read_text());old_assert=next(n for n in ast.walk(old)if isinstance(n,ast.Assert)and "os.environ['GLM_PP_SIZE_R3603'] == '4'"in ast.unparse(n.test))
 old_code=compile(ast.fix_missing_locations(ast.Module(body=[old_assert],type_ignores=[])),'<original-PP4-production-assert>','exec')
 payload={k:base64.b64encode(v).decode()for k,v in [('world',b'world'*20),('control',b'control'*20)]};payload['tp']=[base64.b64encode(b'TP0'*40).decode(),base64.b64encode(b'TP1'*40).decode()]
 base=dict(GLM_PP_SIZE_R3603='2',GLM_CACHE_ROWS_R3627='1',GLM_CACHE_CAP_R3627='32768',GLM_V_ON_DEMAND_R403='1',GLM_BOUNDED_ROPE_R4279='1',GLM_SHARED_CONTROL_R4279='1',GLM_SHARED_PREFIX_VALIDATE_R5814='0',GLM_ACTUAL_CAPTURE_R5164='0',USE_WEIGHT_CACHE_R3604='0',LD_PRELOAD='/refactor/candidates/blas-policy-r271/libblas_policy_r271.so',LD_LIBRARY_PATH='/refactor/install/hygon-glm-5f91ad7-r3672/lib:/other')
 original_import=builtins.__import__
 class DeviceBoundary(Exception):pass
 def no_device_import(name,*args,**kwargs):
  if name=='torch':raise DeviceBoundary('Original production Torch import; device execution intentionally not consumed')
  return original_import(name,*args,**kwargs)
 for stage,node in enumerate(['a02r4n12','a02r4n13']):
  env=dict(base,GLM_STAGE_R5040=str(stage),SLURM_PROCID=str(stage));layout=read_layout(env,cfgpath,node);assert layout['stage']==stage and layout['bounds']==[0,38,78]
  with patch.dict(os.environ,env):
   try:exec(old_code,dict(os=os))
   except AssertionError:old_rejected=True
   else:raise AssertionError('Original PP4 mismatch was not reproduced')
  with patch('builtins.__import__',no_device_import):
   try:production_preengine.validate(topology_environment=env,topology_config_path=cfgpath,topology_hostname=node)
   except DeviceBoundary as e:device_boundaries.append(dict(stage=stage,actual_preengine_prefix_consumed=True,original_Torch_leaf_reached=True,message=str(e)))
   else:raise AssertionError('Unexpected device path execution')
  for rank in range(8):
   def leaf(*args):
    assert len(args)==8;device,st,rows,capacity,config_bytes,*buffers=args;assert(device,st,rows,capacity)==(rank,stage,1,32768);assert config_bytes==json.dumps(cfg).encode()
    expected=[base64.b64decode(payload[k]if k!='tp'else payload[k][stage])for k in ['world','tp','control']];assert [b.raw[:-1]for b in buffers]==expected
    records.append(dict(stage=stage,rank=rank,argument_count=8,rows=rows,capacity=capacity,config_bytes_SHA256=hashlib.sha256(config_bytes).hexdigest(),group_buffer_nbytes=[len(v)for v in expected],group_buffer_SHA256=[hashlib.sha256(v).hexdigest()for v in expected],Native_executed=False));return object()
   lib=types.SimpleNamespace(modern_glm_model_create=leaf);assert create_context(lib,rank,stage,1,32768,cfg,payload,env,node)is not None
  for key,value in [('GLM_PP_SIZE_R3603','4'),('GLM_CACHE_ROWS_R3627','2'),('GLM_CACHE_CAP_R3627','16384'),('GLM_V_ON_DEMAND_R403','0'),('GLM_STAGE_R5040',str(1-stage))]:
   bad=dict(env);bad[key]=value
   try:validate_layout(bad,cfg,node)
   except(AssertionError,KeyError,ValueError):pass
   else:raise AssertionError('Invalid current layout accepted:'+key)
 assert len(records)==16 and len(device_boundaries)==2 and 'torch'not in sys.modules
 return dict(completed=True,passed=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],actual_config_path=str(cfgpath),actual_config_SHA256=sha(cfgpath),actual_production_preengine_module=str(production_preengine.__file__),preengine_SHA256=sha(O/'preengine_r133.py'),topology_SHA256=sha(O/'topology_r404.py'),owner_manifest_SHA256=sha(O/'SOURCE_MANIFEST.json'),original_PP4_assert_negative_roles=2,current_layout_negatives=10,actual_preengine_prefix=device_boundaries,create_wrapper_records=records,actual_create_CABI_argument_count=8,CPU=1,GPU=0,Native_dlopen=False,Native_create=False,weights=False,Model=False,remaining_device_contract='Original Torch CUDA initialization, DSO/CABI audits, group manufacture and actual Native context/create/layout remain unexecuted in this CPU. Original same preengine readonly/CABI branch before old topologyassert was observed in847; this receipt does not qualify PP2 Model fit/math/quality.')
if __name__=='__main__':
 out=Path('/refactor/candidates/GLM-V-on-demand-Model-r404-CPU-layout-v4');out.mkdir(exist_ok=False);path=out/'CPU-result.json'
 try:result=check()
 except BaseException as e:
  import traceback;result=dict(completed=True,passed=False,error=repr(e),traceback=traceback.format_exc(),GPU=0,Native_dlopen=False,Model=False)
 path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(path=str(path),SHA256=sha(path),passed=result['passed'])));raise SystemExit(0 if result['passed']else 1)
