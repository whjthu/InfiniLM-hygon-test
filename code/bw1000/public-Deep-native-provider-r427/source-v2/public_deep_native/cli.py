"""Ordinary public Deep InferenceServer, backed by the concrete afc6 provider."""
import argparse,hashlib,json,os,socket,sys,threading,time
from pathlib import Path,PurePosixPath
from .checkpoint_loader import NATIVE_SHA

PUBLIC_SHA='122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_package(expected):
 root=Path(__file__).resolve().parents[1];p=root/'SOURCE_MANIFEST.json';assert sha(p)==expected
 for name,h in json.loads(p.read_bytes())['files_SHA256'].items():
  item=root/name;assert item.resolve().is_relative_to(root.resolve())and sha(item)==h,name
 return root

def validate_config(config):
 assert config['backend']=='qualified-native-deep-hc0'and config['model_type']=='deepseek_v4'
 assert config['Native_SHA256']==NATIVE_SHA and(config['TP'],config['PP'],config['rows'],config['context'],config['HC_mode'])==(8,1,1,32768,0)
 assert config['Native_seconds']>config['service_seconds']>120
 assert config['public_package_SHA256']==PUBLIC_SHA
 for key in ['model_path','processor_path','Native_library','Common_extension','Common_python_root','runtime_directory','public_package_root']:
  value=config[key]
  if type(value)is not str or'\\'in value or not PurePosixPath(value).is_absolute():raise ValueError('POSIX absolute deployment path required: '+key)
 return config

def prepare_public(config):
 # Generic public route/engine_factory implementation is reused byte-for-byte,
 # independently pinned to the frozen425 package; its GLM provider is not imported.
 package=Path(config['public_package_root']);assert sha(package/'SOURCE_MANIFEST.json')==PUBLIC_SHA
 manifest=json.loads((package/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in manifest['files_SHA256'].items():
  p=package/name;assert p.resolve().is_relative_to(package.resolve())and sha(p)==h
 public=package/'public-python'
 from .worker import torch_setup
 torch_setup();sys.path.insert(0,config['Common_python_root']);sys.path.insert(0,str(public))
 import infinilm,_infinilm
 assert Path(infinilm.__file__).resolve()==(public/'infinilm/__init__.py').resolve()
 assert Path(_infinilm.__file__).resolve()==Path(config['Common_extension']).resolve()and sha(_infinilm.__file__)==config['Common_extension_SHA256']
 from infinilm.processors import AutoInfinilmProcessor
 from infinilm.llm import FinishReason
 from infinilm.server.inference_server import InferenceServer
 return AutoInfinilmProcessor.from_pretrained(config['processor_path']),FinishReason,InferenceServer

def build_server(config,processor,Reason,Server):
 from .prepared_queue import prepare_engine,bind_public_sampling
 from .provider import DeepProvider
 prepared=prepare_engine('Deep',config['model_path'],processor=processor,finish_reasons=Reason,processor_path=config['processor_path'])
 assert prepared['identity']['eos_token_ids']==[1]
 provider=DeepProvider(config,prepared)
 server=Server(model_path=config['model_path'],device='cuda',dtype='bfloat16',tensor_parallel_size=8,pipeline_parallel_size=1,pipeline_parallel_stage=0,cache_type='static',max_batch_size=1,max_cache_len=32768,max_tokens=4096,temperature=0.,top_p=1.,top_k=1,enable_graph=False,skip_load=False,enable_prefix_caching=False,host=config['host'],port=config['port'],tool_call_parser=prepared['parser'],engine_factory=provider,stream_timeout_s=1800,request_timeout_s=7200)
 bind_public_sampling(server,prepared['engine']);return server,provider,prepared

def watch_public_shutdown(provider,http,done,service_deadline):
 # Same actual run/identity/peer contract as the frozen public425 watcher.
 while not done.wait(.2):
  try:
   stop=provider.group.run.stop_requested()
   if stop or provider.group.failed.is_set()or time.time()>=service_deadline:
    http.should_exit=True;return
  except BaseException as error:
   try:provider.group._trip(error)
   finally:http.should_exit=True
   return

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--run',required=True);p.add_argument('--actual-step',required=True);p.add_argument('--gpu-start-epoch',type=float,required=True);p.add_argument('--serve',action='store_true');a=p.parse_args(argv)
 verify_package(a.source_sha);assert sha(a.config)==a.config_sha;config=validate_config(json.loads(a.config.read_bytes()))
 config.update(run=a.run,actual_step=a.actual_step,gpu_start_epoch=a.gpu_start_epoch,source_manifest_SHA256=a.source_sha,provider_config_SHA256=a.config_sha)
 if not a.serve:
  print(json.dumps(dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)));return
 assert config['execute_authorized']is True and socket.gethostname().split('.')[0]==config['node']
 from .worker import configure_process
 from .native_binding import binding
 from .protocol import Run
 configure_process(config);_,receipt,_=binding();assert receipt['library_SHA256']==NATIVE_SHA
 run=Run(config);run.write('FRONTEND.json',dict(PID=os.getpid(),NativeRequest_serialized=False));provider=None
 try:
  processor,Reason,Server=prepare_public(config);server,provider,prepared=build_server(config,processor,Reason,Server)
  run.write('PREPARED.json',dict(prepared_identity=prepared['identity']))
  import uvicorn
  http=uvicorn.Server(uvicorn.Config(server._create_app(),host=config['host'],port=config['port'],log_level='info'));done=threading.Event()
  watcher=threading.Thread(target=watch_public_shutdown,args=(provider,http,done,config['gpu_start_epoch']+config['service_seconds']),name='Deep-public-HTTP-lifecycle',daemon=False);watcher.start();primary=None
  try:http.run()
  except BaseException as error:primary=error;raise
  finally:
   done.set();watcher.join(timeout=2)
   try:provider.stop()
   except BaseException as cleanup:
    if primary is None:raise
    if hasattr(primary,'add_note'):primary.add_note('Deep public stop: '+repr(cleanup))
  if provider.group.failed.is_set():raise provider.group.failure
 except BaseException as error:
  try:run.fail(error,stage=0)
  except BaseException as persistence:
   if hasattr(error,'add_note'):error.add_note('Deep frontend failure record: '+repr(persistence))
  raise

if __name__=='__main__':main()
