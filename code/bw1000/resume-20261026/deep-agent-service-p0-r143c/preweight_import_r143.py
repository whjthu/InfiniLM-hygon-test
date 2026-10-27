"""Real eight-GPU import/provider smoke before opening any checkpoint payload."""
import pathlib,json,hashlib,os,ctypes,sys
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run():
 tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];p=R/'results'/('deep-agent-p0-r143c-'+tag+'-preweight.json');assert not p.exists();q={'passed':False,'GPU':True,'weights_payload_opened':False,'model_constructed':False,'actual_step':tag}
 try:
  m=json.loads((D/'source-manifest-r143.json').read_bytes())
  for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
  c=json.loads((R/'candidates/Deep-agent-p0-r143c/CPU-preflight.json').read_bytes());assert c['completed']and c['CPU_only']and c['static_only']
  for n,h in c['baseline_files_SHA256'].items():assert sha(n)==h,n
  for n,h in c['overlay_files_SHA256'].items():assert sha(n)==h,n
  import torch
  assert torch.cuda.device_count()==8;torch.cuda.set_device(0);q['visible_GPU_count']=8
  native=pathlib.Path(c['Native_path']);assert sha(native)==m['Native_SHA256'];lib=ctypes.CDLL(str(native))
  class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
  dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.restype=ctypes.c_int
  providers={}
  for symbol in ['modern_native_engine_create','modern_native_engine_keys','modern_native_engine_load','modern_native_engine_finish','modern_native_engine_reset','modern_native_engine_forward','modern_native_engine_destroy']:
   info=Info();assert dl(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info));actual=pathlib.Path(info.fname.decode()).resolve();assert actual==native.resolve()and sha(actual)==m['Native_SHA256'];providers[symbol]={'path':str(actual),'SHA256':sha(actual)}
  from infinilm.server.inference_server import InferenceServer
  from infinilm.llm import FinishReason
  assert pathlib.Path(sys.modules[InferenceServer.__module__].__file__).resolve().is_relative_to(R/'candidates/Deep-agent-p0-r143c/python')
  receipt=json.loads((R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json').read_bytes());maps=pathlib.Path('/proc/self/maps').read_text().splitlines();actual_paths={pathlib.Path(line.split()[-1]).resolve()for line in maps if len(line.split())>=6 and line.split()[-1].startswith('/')};bindings=[]
  for name,proof in receipt['actual_DTK_bindings'].items():
   expected=pathlib.Path(name).resolve();stem=pathlib.Path(name).name.split('.so')[0];found={x for x in actual_paths if stem in x.name};assert found=={expected},(stem,[str(x)for x in found],str(expected));assert sha(expected)==proof['SHA256'];bindings.append({'path':str(expected),'SHA256':sha(expected)})
  q.update(passed=True,standard_server_imported=True,Native_CAPI_providers=providers,actual_required_DSO_bindings=bindings,Core_stack_substitution=False)
 except BaseException as e:q['error']=repr(e)
 p.write_text(json.dumps(q,indent=2)+'\n');print('DEEP_AGENT_P0_R143C_PREWEIGHT_IMPORT',json.dumps({k:q.get(k)for k in ['passed','visible_GPU_count','error']}),flush=True);assert q['passed']
