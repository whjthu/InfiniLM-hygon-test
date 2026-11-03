"""Changed actual public/provider/actor/loader/forward/decode path, recording leaves."""
import ast,asyncio,ctypes,enum,hashlib,io,json,logging,os,queue,tempfile,threading,time,weakref
from contextlib import asynccontextmanager,redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from public_deep_native import actors,worker,native_context,checkpoint_loader
from public_deep_native.cli import build_server,validate_config

class Reason(str,enum.Enum):
 CANCELED='canceled';ERROR='error';TIMEOUT='timeout';EOS_TOKEN='eos_token';LENGTH='length'

class ProcessorLeaf:
 def get_tokenizer(self):return self
 def apply_chat_template(self,conversation,tokenize=False,add_generation_prompt=False,**kw):
  text=json.dumps(conversation,ensure_ascii=False,separators=(',',':'))
  return text+('<｜Assistant｜><think>'if add_generation_prompt else'')
 def __call__(self,prompt,return_tensors=None,**kw):return {'input_ids':np.array([[ord(c)for c in prompt]],np.int64)}
 def decode(self,ids,**kw):return ''.join(chr(i)for i in ids)

class FunctionLeaf:
 def __init__(self,f):self.f=f
 def __call__(self,*args):return self.f(*args)

class NativeLeaf:
 def __init__(self):
  self.functions={};self.loaded=0;self.batch_sizes=[];self.destroyed=0;self.sampling=0;self.calls=0;self.fail_forward=False;self.block_next=False;self.entered=threading.Event();self.release=threading.Event();self.counts={};self.memory=weakref.WeakValueDictionary()
 def __getattr__(self,name):
  if name not in self.functions:self.functions[name]=FunctionLeaf(lambda *a:self.call(name,*a))
  return self.functions[name]
 def call(self,name,*a):
  if name=='modern_native_engine_error':return b'recorded Native error'
  if name=='modern_native_engine_create':assert a[1:]==(1,32768);return 123
  if name=='modern_native_engine_keys':return json.dumps(['weight.%05d'%i for i in range(67569)]).encode()
  if name=='modern_native_engine_load':
   rows=json.loads(a[1]);assert all(x['pointer']in self.memory for x in rows),'CPU tensor borrow died before Native load returned';assert all(x['dtype']in [0,1,2,3]for x in rows);self.loaded+=len(rows);self.batch_sizes.append(len(rows));return 0
  if name=='modern_native_engine_destroy':self.destroyed+=1;return 0
  if name=='modern_native_engine_reset':self.sampling=0;return 0
  if name=='modern_native_engine_finish':assert self.loaded==67569;return 0
  if name=='modern_native_engine_forward':
   ctx,T,B,*p=a;assert ctx==123 and B==1
   if self.block_next:self.block_next=False;self.entered.set();assert self.release.wait(5)
   self.calls+=1
   if self.fail_forward:return 1
   raw=np.ctypeslib.as_array(ctypes.cast(p[6],ctypes.POINTER(ctypes.c_uint16)),shape=(129280,));raw.fill(0x3f80)
   sampled=ctypes.cast(p[7],ctypes.POINTER(ctypes.c_int64));sampled[0]=[11,12,101,102,1][min(self.sampling,4)];self.sampling+=1
   past=ctypes.cast(p[1],ctypes.POINTER(ctypes.c_int32))[0]
   for d in range(8):
    hc=0 if T==512 else 2;self.counts[('deep_hc_canonical_calls_r7130',d,hc)]=self.counts.get(('deep_hc_canonical_calls_r7130',d,hc),0)+86
    self.counts[('modern_deep_layout_calls_r7052',d,3)]=self.counts.get(('modern_deep_layout_calls_r7052',d,3),0)+43
    k=3 if T==512 and past==0 else 4;self.counts[('deep_moe_tail_calls_r7211',d,k)]=self.counts.get(('deep_moe_tail_calls_r7211',d,k),0)+43
   return 0
  if name.endswith('_get_r7130')or name.endswith('_get_r7052')or name.endswith('_get_r7211')or name.endswith('_get_r5920'):return 0
  if name in ['modern_deep_c4_tile_policy_r5920','modern_deep_allselected_set_r813']:return 0
  if 'calls_'in name:return self.counts.get((name,*a),0)
  raise AssertionError(('unknown Native leaf',name))

def public_server_class():
 p=Path(__file__).parents[1]/'public-GLM-native-provider-r425/public-python/infinilm/server/inference_server.py'
 cls=next(n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.ClassDef)and n.name=='InferenceServer');cls.body=[n for n in cls.body if getattr(n,'name',None)in ['__init__','_create_app','_build_sampling_params']]
 class AppLeaf:
  def __init__(self,*,lifespan):self.lifespan=lifespan
 def generic(**kw):raise AssertionError('Generic engine must not be selected by explicit provider')
 ns=dict(os=os,Optional=__import__('typing').Optional,KVTransferConfig=object,TOOL_CALL_FORMATS={'deepseek_v4':None},AsyncLLMEngine=generic,FastAPI=AppLeaf,asynccontextmanager=asynccontextmanager,logger=logging.getLogger('Deep427CPU'),SamplingParams=SimpleNamespace,DEFAULT_STREAM_TIMEOUT=100.,DEFAULT_REQUEST_TIMEOUT=1000.)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(p),'exec'),ns);Server=ns['InferenceServer'];Server._register_routes=lambda self,app:None;return Server

def case(failure=False):
 root=Path(__file__).parent;lib=NativeLeaf();torch=SimpleNamespace(float32='F32',bfloat16='BF16',int8='I8',int64='I64',cuda=SimpleNamespace(mem_get_info=lambda rank:(30*1024**3,80*1024**3)))
 class TensorLeaf:
  def __init__(self,name):
   index=int(name.split('.')[-1]);self.dtype=['F32','BF16','I8','I64'][index%4];self.shape=(140000000,)if index==0 else(1,);lib.memory[id(self)]=self
  def contiguous(self):return self
  def data_ptr(self):return id(self)
  def numel(self):return self.shape[0]
  def element_size(self):return {'F32':4,'BF16':2,'I8':1,'I64':8}[self.dtype]
 class ReaderLeaf:
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def get_tensor(self,name):return TensorLeaf(name)
 affinity={'actual_owner_report':{'owners':[dict(device=i,changes=1,verified_mask=','.join(str(v)for v in range(i*4,i*4+4)))for i in range(8)]}}
 policy=dict(failures=0,owners=[dict(tid=i,created=1,verified=1,caller='libinfiniops.so')for i in range(8)])
 cfg_native=dict(num_hidden_layers=43,num_nextn_predict_layers=1,compress_ratios=[0,0]+[4,128]*21,vocab_size=129280)
 mapping={'weight.%05d'%i:'recorded.safetensors'for i in range(67569)}
 with tempfile.TemporaryDirectory()as tmp:
  path=Path(tmp);model=path/'model';model.mkdir();(model/'generation_config.json').write_text(json.dumps(dict(eos_token_id=[1])))
  config=json.loads((root/'config.template.json').read_bytes());validate_config(config)
  for key,value in [('HC_mode',1),('Native_SHA256','976f'+'0'*60),('model_type','fake_native_deep')]:
   bad=dict(config);bad[key]=value
   try:validate_config(bad)
   except AssertionError:pass
   else:raise AssertionError('Unqualified default accepted: '+key)
  config.update(model_path=str(model),processor_path=str(model),runtime_directory=str(path),run='cpu-'+str(failure),actual_step='CPU.recording',source_manifest_SHA256='recording-not-device',provider_config_SHA256='recording-config',gpu_start_epoch=time.time())
  server,provider,prepared=build_server(config,ProcessorLeaf(),Reason,public_server_class());engine=prepared['engine'];app=server._create_app()
  req=engine.add_chat_request(messages=[dict(role='user',content='x'*2050)],sampling_params=SimpleNamespace(temperature=0.,max_tokens=128,stop=None),request_id='full-input',request_data={'max_tokens':128,'_agent_schemas':{},'_agent_choice':'none'})
  binding=checkpoint_loader.declare(lib);lib.fail_forward=failure
  leafs=[patch.object(worker,'configure_process',return_value=None),patch.object(worker,'verify_native_chain',return_value=({},dict(library_SHA256=config['Native_SHA256']),{})),patch.object(worker,'torch_setup',return_value=torch),patch.object(worker,'safe_open_dependency',return_value=lambda *a,**k:ReaderLeaf()),patch.object(worker,'read_model_metadata',return_value=(cfg_native,mapping)),patch.object(worker,'bind_library',return_value=binding),patch.object(native_context,'verify_affinity',return_value=affinity),patch.object(native_context,'providers',return_value={'recorded_external_ELF':True}),patch.object(native_context,'observe_threads',return_value=dict(status='RECORDING_NATIVE_LEAF',owners=[])),patch.object(native_context,'blas_snapshot',return_value=policy)]
  from contextlib import ExitStack
  with ExitStack()as stack:
   for leaf in leafs:stack.enter_context(leaf)
   async def run():
    try:
     async with app.lifespan(app):
      assert provider.is_healthy()and provider.group.owner_thread.is_alive()
      until=time.monotonic()+10
      while not req.is_finished():
       assert time.monotonic()<until;await asyncio.sleep(.01)
      if failure:
       assert req.finish_reason==Reason.ERROR
       return
      assert req.generated_token_ids==[101,102,1]and req.visible_token_ids==[101,102]and req.finish_reason==Reason.EOS_TOKEN
      record=json.loads((provider.group.run.path/'owner.json').read_bytes())['requests'][0]
      assert record['actual_forward_attempts']==record['actual_forward_completed']==5 and record['Native_forward_host_summary']['prefill_calls']==3 and record['Native_forward_host_summary']['decode_calls']==2
      assert sum(v['tokens']for v in record['Native_continuation']if v['kind']=='prefill')==len(req.prompt_token_ids)>2050
      # Stop intent must cancel an active request without finishing it before
      # the recorded Native forward is allowed to return/reset/drain.
      lib.block_next=True
      active=engine.add_chat_request(messages=[dict(role='user',content='stop active')],sampling_params=SimpleNamespace(temperature=0.,max_tokens=128,stop=None),request_id='active',request_data={'max_tokens':128,'_agent_schemas':{},'_agent_choice':'none'})
      assert await asyncio.to_thread(lib.entered.wait,3)
      def release_after_cancel():
       assert provider.group.stop_intent.wait(3)
       with engine._admission_lock:assert active.cancel_event.is_set()and not active.is_finished()
       lib.release.set()
      watcher=threading.Thread(target=release_after_cancel);watcher.start()
     if not failure:watcher.join(timeout=3);assert active.finish_reason==Reason.CANCELED
    except BaseException:
     if not failure:raise
     assert provider.group.failed.is_set()
   asyncio.run(run())
  assert lib.loaded==67569 and lib.destroyed==1 and not provider.group.owner_thread.is_alive()
  assert lib.batch_sizes[0]==1 and max(lib.batch_sizes)==1024
  report=json.loads((provider.group.run.path/'owner.json').read_bytes())
  if failure:assert report['passed']is False and report['requests'][0]['actual_forward_completed']==0
  return dict(case='Native_nonzero_failstop_cleanup'if failure else'actual_public_lifespan_rawloader_fullprefill_Native_sample_EOS_active_stop',passed=True,real_source_functions=True,Native_keys_recorded=lib.loaded,raw_batches=len(lib.batch_sizes),destroy_calls=lib.destroyed,Native_worker_threads_are_recording_not_device=True)

def source_cli_case():
 from public_deep_native import cli
 root=Path(__file__).resolve().parent
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 package=root.parent/'public-GLM-native-provider-r425';manifest=package/'SOURCE_MANIFEST.json'
 assert digest(manifest)==cli.PUBLIC_SHA
 for name,h in json.loads(manifest.read_bytes())['files_SHA256'].items():assert digest(package/name)==h,name
 output=io.StringIO()
 with redirect_stdout(output):
  cli.main(['--config',str(root/'config.template.json'),'--config-sha',digest(root/'config.template.json'),'--source-sha',digest(root/'SOURCE_MANIFEST.json'),'--run','recording-source-cli','--actual-step','recording.source','--gpu-start-epoch',str(time.time())])
 assert json.loads(output.getvalue())==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 return dict(case='actual_cli_source_validation_and_frozen_public_dependency',passed=True,public_package_SHA256=cli.PUBLIC_SHA)

if __name__=='__main__':
 print(json.dumps(dict(passed=True,cases=[source_cli_case(),case(False),case(True)],GPU=False,Native_dlopen=False,weights_payload_opened=False,new_compile=False,external_jobs=0,recording_leaves=['Native CAPI and provider ELF/proc identities','weight tensors/safetensors/config metadata','Torch/device/process environment','official processor fixture','FastAPI app/routes construction; actual public lifespan methods executed'],new_Model_qualification=False)))
