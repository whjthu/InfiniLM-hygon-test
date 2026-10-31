"""Actual latest root/extension import and injected app, no Model construction."""
import pathlib,os,sys,json,hashlib,traceback,socket,datetime,asyncio
R=pathlib.Path('/refactor');family=sys.argv[1];B=R/'candidates/latest-family-service-r270'/family;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'family':family,'passed':False,'CPU_only':True,'GPU_calls':0,'weight_payload_opened':False,'Model_constructed':False,'synthetic_root_core_llm':False,'actual_compiled_extension_import':False,'positive_Model_requests':0}
def forbidden_network(*a,**k):raise RuntimeError('CPU entry probe forbids network')
socket.create_connection=forbidden_network;socket.socket.connect=forbidden_network
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_MODULES_CACHE=str(B/'hf-modules'),HF_HOME=str(B/'hf-home'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def audit(event,args):
 if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weight payload read forbidden')
sys.addaudithook(audit)
def kfd():
 rows=[]
 for p in pathlib.Path('/proc/self/fd').iterdir():
  try:
   dest=str(p.resolve())
   if '/dev/kfd'in dest or '/dev/dri/'in dest:rows.append(dest)
  except FileNotFoundError:pass
 return rows
try:
 expected=json.loads((B/'installation.json').read_bytes());assert not kfd()
 import torch
 torch.set_num_threads(1)
 def guard(name):
  def unique(*a,**k):raise RuntimeError('GPU API forbidden: '+name)
  return unique
 for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,guard(name))
 torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 # No synthetic InfiniLM packages. Import the actual compiled extension through
 # the complete actual public root, then guard unneeded Model constructors.
 import infinilm,_infinilm
 assert pathlib.Path(infinilm.__file__).resolve()==(B/'python/infinilm/__init__.py').resolve()
 assert sha(_infinilm.__file__)==expected['extension_SHA256']
 for cls in [infinilm.LLM,infinilm.AsyncLLMEngine]:cls.__init__=guard(cls.__name__+' constructor')
 q.update(actual_compiled_extension_import=True,root_SHA256=sha(infinilm.__file__),extension_SHA256=sha(_infinilm.__file__),extension_path=str(pathlib.Path(_infinilm.__file__).resolve()),Model_constructors_guarded=True)
 from prepared_family_entry_r268 import prepare_app
 model='/old/models/ZhipuAI-GLM-5.3'if family=='GLM'else'/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8';metadata=None if family=='GLM'else'/refactor/results/latest-three-processors-r260/Deep-model-metadata'
 app=prepare_app(family,model,processor_path=metadata);q['prepared_identity']=app['identity']
 async def check():
  import httpx
  async with app['app'].router.lifespan_context(app['app']):
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app['app']),base_url='http://cpu-only')as c:
    a=await c.get('/health');b=await c.get('/v1/models');assert a.status_code==200 and b.status_code==200
    return {'health':a.status_code,'models':b.status_code,'network':False,'Model_inference':False}
 q['routes']=asyncio.run(check());assert not torch.cuda.is_initialized()and not kfd();q['torch_cuda_initialized']=False;q['GPU_device_fds']=[]
 selected={}
 for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
  if len(line.split())<6:continue
  p=pathlib.Path(line.split()[-1]);name=p.name
  if p.is_absolute()and ('_infinilm.cpython'in name or name in ['libinfinilm_runtime.so','libinfiniops.so','libinfiniccl.so','libinfinirt.so']):
   p=p.resolve();selected[str(p)]=sha(p)
 assert sum('_infinilm.cpython'in pathlib.Path(p).name for p in selected)==1
 assert selected.get(expected['extension_real_path'])==expected['extension_SHA256']
 q['actual_loaded_family_DSO_SHA256']=selected;q['passed']=True
except BaseException as exc:q.update(error_type=type(exc).__name__,error=str(exc)[:1600],failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(exc.__traceback__)])
finally:
 q['completed']=True;(B/'CPU-actual-entry-r271.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['family','completed','passed','error_type','error']}))
assert q['passed']
