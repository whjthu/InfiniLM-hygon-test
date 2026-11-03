"""Actual four-stage ordinary bootstrap, exact source/loader/head/32owner floor."""
from pathlib import Path
import ctypes,json,os,socket,sys,time,traceback,signal,importlib
from contract import CUTS,GEOMETRY,CAP,sha,layout,memory_gate,validate_config,frozen_fixtures
from lifecycle import Gate,guarded_safe_open
from native_binding import bind

def import_exact(name,path):
 value=importlib.import_module(name)
 assert Path(value.__file__).resolve()==Path(path).resolve(),name
 return value

def verify_bundle(directory,digest):
 D=Path(directory);p=D/'SOURCE_MANIFEST.json';assert sha(p)==digest
 m=json.loads(p.read_bytes())
 for n,h in m['files_SHA256'].items():
  assert not Path(n).is_absolute()and'..'not in Path(n).parts and sha(D/n)==h,n
 return m

class Settings:
 def __init__(self,env,root=Path('/refactor')):
  import math,re
  assert env.get('KIMI410_ORDINARY_RUNTIME')=='1'
  self.node=int(env['SLURM_PROCID']);assert self.node in range(4)
  self.run=env['KIMI410_RUN'];assert re.fullmatch('[a-z0-9][a-z0-9-]{7,39}',self.run)
  self.nonce=env['KIMI410_NONCE'];assert len(self.nonce)>=32
  self.head=env['KIMI_ENGINE_HEAD'];self.port=int(env['GLOBAL_PORT_R2354']);assert 1024<=self.port<=58535
  self.start=float(env['KIMI410_GPU_START_UNIX']);self.deadline=float(env['KIMI410_NATIVE_DEADLINE_UNIX'])
  self.native_s=int(env['KIMI410_NATIVE_SECONDS']);assert 1<=self.native_s<=7200
  assert math.isfinite(self.start)and self.start>0 and self.deadline==self.start+self.native_s
  self.root=Path(root);self.cancel=Path(env['KIMI410_CANCEL_FILE']);assert self.cancel.parent==self.root/'control'and self.run in self.cancel.name

class RealDependencies:
 """CPU tests replace actual CABI/device/tensor/transport leaves, never this class."""
 def __init__(self,directory,providers=None):
  self.D=Path(directory);self.providers=providers or{};self.control=self.sampler=self.lib=None
 def leaf(self,name,default):return self.providers.get(name,default)
 def prepare(self,settings,gate):
  self.settings=settings;self.gate=gate;D=self.D
  pins=json.loads((D/'bindings.json').read_bytes());self.pins=pins
  # Original source manifests and receipts are checked before Torch/device load.
  original=json.loads((D/'baseline-manifest-r252.json').read_bytes());assert sha(D/'baseline-manifest-r252.json')==pins['baseline_manifest_SHA256']
  for p,h in original['source_SHA256'].items():assert sha(p)==h,p
  self.model=Path(pins['model_view']['path'])
  assert sha(self.model/'config.json')==pins['model_view']['config_SHA256']and sha(self.model/'model.safetensors.index.json')==pins['model_view']['index_SHA256']
  cp=Path(pins['candidate_receipt']);assert sha(cp)==pins['candidate_receipt_SHA256'];candidate=json.loads(cp.read_bytes())
  assert candidate['completed']and candidate['passed']and candidate['CPU_only']and not candidate['GPU']and not candidate['Native_dlopen']
  assert candidate['native_SHA256']==pins['candidate_Native_SHA256']and sha(candidate['native'])==pins['candidate_Native_SHA256']
  assert len(candidate['all_linked_object_SHA256'])==33
  for p,h in candidate['all_linked_object_SHA256'].items():assert sha(p)==h,p
  assert candidate['actual_dependency_library_SHA256']==pins['actual_dependency_library_SHA256']
  for v in candidate['actual_dependency_library_SHA256'].values():assert sha(v['path'])==v['SHA256']
  fixturepins=json.loads((D/'FIXTURE_PINS.json').read_bytes());self.fixtures=frozen_fixtures(fixturepins['frozen_inputs_path'],fixturepins)
  for v in pins['external_packages'].values():verify_bundle(v['path'],v['manifest_SHA256'])
  self.modules={}
  for name in ['parameter_loader_r9143','shard_head_contract_r9183','weight_key_guard_r9156','memory_guard_r9144','model_affinity_check_r2628','stage_control_r2386','xtml_codec_r143','long_codec_r179']:
   self.modules[name]=import_exact(name,D/(name+'.py'))
  for name,p in pins['external_Python_modules'].items():
   assert sha(p['path'])==p['SHA256'];sys.path.insert(0,str(Path(p['path']).parent));self.modules[name]=import_exact(name,p['path'])
  for k,v in pins['required_environment'].items():assert os.environ.get(k)==v,(k,'production shell policy')
  assert self.leaf('hostname',lambda:socket.gethostname().split('.')[0])()==pins['nodes'][settings.node]
  gate.check('Torch-import-device-inventory')
  def torch_provider():
   import torch
   return torch
  self.torch=self.leaf('torch',torch_provider)();self.torch.set_num_threads(1);assert self.torch.cuda.device_count()==8
  for d in range(8):
   p=self.torch.cuda.get_device_properties(d);assert('gfx936'in p.name or'gfx936'in getattr(p,'gcnArchName',''))and p.total_memory>=63<<30 and p.multi_processor_count==80
  def open_provider():
   from safetensors import safe_open
   return safe_open
  self.safe_open=guarded_safe_open(self.leaf('safe_open',open_provider)(),gate)
  self.control=self.leaf('control',self.modules['stage_control_r2386'].StageControl)(settings.node,settings.head,settings.port+7000,settings.nonce,4,max(1,min(7200,gate.remaining())))
  runtime_SHA=sha(D/'SOURCE_MANIFEST.json')
  rows=self.control.collect('r410-admission',dict(node=settings.node,Native_SHA256=candidate['native_SHA256'],runtime_source_manifest_SHA256=runtime_SHA,original_GPU_start_unix=settings.start,Native_deadline_unix=settings.deadline,nonce_SHA256=__import__('hashlib').sha256(settings.nonce.encode()).hexdigest()))
  assert len(rows)==4 and {r['node']for r in rows}==set(range(4))and all(r['Native_SHA256']==candidate['native_SHA256']and r['runtime_source_manifest_SHA256']==runtime_SHA and r['original_GPU_start_unix']==settings.start and r['Native_deadline_unix']==settings.deadline and r['nonce_SHA256']==rows[0]['nonce_SHA256']for r in rows)
  gate.check('Native-dlopen-binding');self.lib,self.actual_providers=self.leaf('bind',bind)(candidate['native'],gate)
  self.loaded_maps=self.leaf('maps',self.verify_maps)(candidate['actual_dependency_library_SHA256'])
  self.sampler=self.leaf('sampler',self.modules['memory_guard_r9144'].MemorySampler)(self.torch,interval=.01)
  return dict(candidate_receipt_SHA256=pins['candidate_receipt_SHA256'],Native_SHA256=candidate['native_SHA256'],actual_CAPI_providers=self.actual_providers,actual_loaded_libraries=self.loaded_maps,all33_object_hashes_checked=True,fixture_source_checked_before_Torch=True)
 @staticmethod
 def verify_maps(expected):
  lines=Path('/proc/self/maps').read_text().splitlines();actual={}
  for name,v in expected.items():
   paths={line.split()[-1]for line in lines if Path(line.split()[-1]).name==name};assert len(paths)==1
   p=next(iter(paths));assert sha(p)==v['SHA256'];actual[name]=dict(path=p,SHA256=sha(p))
  return actual
 def check(self,rc):
  if rc:raise RuntimeError(self.lib.modern_kimi_engine_error_r2475().decode())
 def memory(self):return[list(self.torch.cuda.mem_get_info(d))for d in range(8)]
 def guard(self,label,report,save):
  self.gate.check('memory:'+label);snapshot=self.sampler.snapshot();assert not snapshot['sampler_error']
  packets=self.control.collect('r410-memory:'+label,dict(node=self.settings.node,snapshot=snapshot));attempt=dict(label=label,owners=packets,required_free_bytes=4<<30)
  report['last_memory_guard_attempt']=attempt;save();attempt['minimum_free_bytes']=memory_gate(packets);report['last_memory_guard']=attempt;save();return attempt
 def run_requests(self,ctx,report,save):
  from ordinary import Native,run
  codec=self.modules['long_codec_r179'].KimiLongCodec(self.model)
  def grade(f,ids,finish):return self.modules['grade_output'].grade_output(f,ids,finish,codec.decode)
  def record(result):report['ordinary_report']=result;save()
  return run(Native(self.lib,ctx,self.control,self.settings.node,self.gate,report,save),self.fixtures,grade,record)

def execute(settings,gate,deps,report,save):
 ctx=None;normal=False;primary=None;cleanup=[]
 report.update(completed=False,Model_qualified=False,node=settings.node,target_cards=32,draft_cards=0,HTTP=False,phase='preengine',topology_package_not_network_only=True)
 try:
  gate.check('entry');report.update(deps.prepare(settings,gate));save();lib=deps.lib
  ranges=[]
  for stage in range(4):
   a=(ctypes.c_size_t*2)();deps.check(lib.modern_kimi_engine_partition_r2475(stage,4,a));ranges.append(list(a))
  assert ranges==CUTS;report['PP_ranges']=ranges
  cfg=json.loads((deps.model/'config.json').read_bytes());text=validate_config(cfg)
  report['memory_before_create']=deps.memory();assert min(v[0]for v in report['memory_before_create'])>58<<30
  deps.control.collect('r410-before-create',dict(node=settings.node));gate.check('create')
  tick=time.monotonic();ctx=lib.modern_kimi_engine_create_ordinary_tp8_pp4_r410(json.dumps(cfg).encode(),settings.node,settings.head.encode(),settings.port)
  if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
  report['create_seconds']=time.monotonic()-tick
  actual_layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert actual_layout==layout(settings.node);report['layout']=actual_layout
  geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert geometry==GEOMETRY;report['cache_geometry']=geometry
  capability=json.loads(lib.modern_kimi_ordinary_capabilities_r410(ctx));assert capability==CAP;report['ordinary_capabilities']=capability
  head=deps.modules['shard_head_contract_r9183'];report['head_before_weights']=head.read_head(lib,settings.node,4,False)
  # Actual current hook is lazy: RuntimeWorkspaceHandle binds on first Ops submit,
  # not on Runtime/model construction. Record the pre-work state without treating
  # an unbound owner as a violation; the unchanged strict verifier runs below
  # immediately after first successful forward, then again after requests.
  report['owner_affinity_before_weight_load']=dict(actual_report=json.loads(lib.modern_owner_affinity_report_r2392()),parent_CPU_allowed_mask=sorted(os.sched_getaffinity(0)),stage='pre-work observation only; first Ops submit has not necessarily occurred',qualified=False)
  save()
  keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));mapping=json.loads((deps.model/'model.safetensors.index.json').read_bytes())['weight_map'];keycheck=deps.modules['weight_key_guard_r9156']
  summary=keycheck.summarize_keys(keys,mapping,settings.node,ranges);report['actual_engine_weight_keys']=summary
  report['all93_stage_keys']=keycheck.validate_all_stages(deps.control.collect('r410-keys',dict(node=settings.node,keys=summary)),ranges)
  deps.guard('before-load',report,save);report['phase']='loading_weights';save()
  report.update(deps.modules['parameter_loader_r9143'].load_parameters(lib,ctx,deps.model,text,keys,deps.safe_open,deps.torch,deps.check,lambda value:(report.update(value),save())))
  report['head_after_finish']=head.read_head(lib,settings.node,4,True);deps.guard('after-finish',report,save)
  report['phase']='requests';save();deps.control.collect('r410-weights-ready',dict(node=settings.node,loaded_keys=len(keys)))
  report['ordinary_report']=deps.run_requests(ctx,report,save)
  assert report['ordinary_report']['completed']and len(report['ordinary_report']['requests'])==2 and all(r['completed']and r['normal_reset']and r['quality']['passed']and r['finish']=='eos'for r in report['ordinary_report']['requests'])
  report['head_after_requests']=head.read_head(lib,settings.node,4,True);deps.guard('after-requests',report,save)
  report['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));report['owner_affinity']=deps.modules['model_affinity_check_r2628'].verify(lib)
  deps.control.collect('r410-before-destroy',dict(node=settings.node));normal=True
 except BaseException as e:
  primary=e;report.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  if deps.sampler is not None:
   try:deps.sampler.close();assert not deps.sampler.thread.is_alive();report['sampler_CPU_cleanup_completed']=True
   except BaseException as e:cleanup.append(dict(phase='sampler',error=repr(e)))
  if ctx:
   try:deps.check(deps.lib.modern_kimi_engine_destroy_r2475(ctx));report['all8_owned_workers_shutdown_completed']=True
   except BaseException as e:cleanup.append(dict(phase='Native-destroy',error=repr(e)))
  if deps.control is not None:
   try:
    if normal and not cleanup:deps.control.collect('r410-after-destroy',dict(node=settings.node,normal=True))
    deps.control.close();report['control_CPU_cleanup_completed']=True
   except BaseException as e:cleanup.append(dict(phase='control-close',error=repr(e)))
  report.update(completed=bool(normal and primary is None and not cleanup),cleanup_errors=cleanup,deadline_checks=gate.counts,phase='completed'if normal and primary is None and not cleanup else'failed')
  try:save()
  except BaseException as e:
   if primary is None:primary=e
   else:report['secondary_save_error']=repr(e)
 if primary is not None:raise primary
 if cleanup:raise RuntimeError('Owned ordinary cleanup failed')
 return report

def main(env=None,providers=None,root=Path('/refactor')):
 env=os.environ if env is None else env;D=Path(__file__).resolve().parent;settings=Settings(env,root);gate=Gate(settings)
 (providers or{}).get('install_signal_handler',lambda f:signal.signal(signal.SIGTERM,f))(lambda *_:setattr(gate,'signal_cancel',True))
 digest=env['KIMI410_RUNTIME_SOURCE_SHA256'];verify_bundle(D,digest)
 # A later published explicit Model claim is necessary; this source/CPU package is OFF.
 claim=Path(env['KIMI410_PUBLISHED_MODEL_CLAIM']);assert sha(claim)==env['KIMI410_PUBLISHED_MODEL_CLAIM_SHA256'];c=json.loads(claim.read_bytes())
 assert c['Model_GO']is True and c['runtime_source_manifest_SHA256']==digest and c['target_cards']==32 and c['draft_cards']==0 and c['Native_seconds']==settings.native_s
 path=settings.root/'results'/('Kimi-ordinary-r410-'+settings.run+'-node'+str(settings.node)+'.json');assert not path.exists();report=dict(source_manifest_SHA256=digest,actual_step=env['SLURM_JOB_ID']+'.'+env['SLURM_STEP_ID'])
 def save():
  temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8');temporary.replace(path)
 save();execute(settings,gate,RealDependencies(D,providers),report,save)
 print('KIMI_ORDINARY_R410_NORMAL '+str(settings.node),flush=True)
if __name__=='__main__':main()
