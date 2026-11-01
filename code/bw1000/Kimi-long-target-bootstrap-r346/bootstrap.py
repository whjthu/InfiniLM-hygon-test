"""One actual32-owner target residency and immutable337 four-request driver."""
from pathlib import Path
import ctypes,json,os,socket,sys,time,traceback,signal
from audit import verify_artifacts,verify_bundle,sha
from lifecycle import Settings,Gate,GuardedLibrary,guarded_safe_open,connect_draft,NATIVE_SHA
from first_capture import validate as validate_first_capture

class RealDependencies:
 def __init__(self,directory):self.directory=Path(directory);self.control=self.sampler=self.rawlib=self.lib=self.native=None
 def prepare(self,settings,gate):
  self.settings=settings;self.gate=gate;self.bound=verify_artifacts(self.directory,gate);self.m=self.bound['manifest'];self.modules=self.bound['modules'];self.model=self.bound['model']
  for k,v in self.m['required_environment'].items():assert os.environ.get(k)==v,k
  assert socket.gethostname().split('.')[0]==self.m['nodes'][settings.node]
  gate.check('Torch-import-and-device-inventory')
  import torch
  from safetensors import safe_open
  self.torch=torch;self.safe_open=guarded_safe_open(safe_open,gate);torch.set_num_threads(1);assert torch.cuda.device_count()==8
  for d in range(8):
   p=torch.cuda.get_device_properties(d);assert('gfx936'in p.name or'gfx936'in getattr(p,'gcnArchName',''))and p.total_memory>=63*1024**3 and p.multi_processor_count==80
  gate.check('StageControl-construction')
  self.control=self.modules['stage_control_r2386'].StageControl(settings.node,settings.head,settings.port+7000,settings.nonce,4,max(1,min(7200,gate.remaining())))
  admission=self.control.collect('r346-source-admission',dict(node=settings.node,Native_SHA256=NATIVE_SHA,session_SHA256=settings.identity['nonce_SHA256']))
  assert len(admission)==4 and {v['node']for v in admission}==set(range(4))and all(v['Native_SHA256']==NATIVE_SHA and v['session_SHA256']==settings.identity['nonce_SHA256']for v in admission)
  gate.check('Native-dlopen-binding')
  from native_binding_r346 import bind,verify_maps
  self.library,self.lib,self.providers=bind(self.m,self.bound['binding_input'],gate);self.rawlib=self.lib.raw;self.verify_maps=verify_maps
  # Additive327 entry points resolve to the same exact candidate DSO.
  class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
  dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int
  for name in ['modern_kimi_long_features_begin_r305','modern_kimi_long_features_capabilities_r305']:
   value=Dl();assert dl(ctypes.cast(getattr(self.rawlib,name),ctypes.c_void_p),ctypes.byref(value))and Path(value.fname.decode()).resolve()==self.library.resolve();self.providers[name]=str(self.library)
  self.maps=verify_maps(self.m)
  self.sampler=self.modules['memory_guard_r9144'].MemorySampler(torch,interval=.01)
  return dict(artifact_audit=self.bound['audit'],actual_CAPI_providers=self.providers,actual_loaded_libraries=self.maps,python_providers={'torch':{'path':torch.__file__,'SHA256':sha(torch.__file__)},'safetensors_safe_open':str(safe_open.__module__)})
 def check(self,code):
  if code:raise RuntimeError(self.lib.modern_kimi_engine_error_r2475().decode())
 def memory(self):return[list(self.torch.cuda.mem_get_info(d))for d in range(8)]
 def memory_guard(self,label,report,save):
  self.gate.check('memory-guard:'+label);snap=self.sampler.snapshot();assert not snap['sampler_error'];packets=self.control.collect('r346-memory:'+label,snap)
  minimum=min(pair[0]for p in packets for pair in p['memory']);attempt=dict(label=label,minimum_free_bytes=minimum,required_free_bytes=4<<30,owners=packets)
  report['last_memory_guard_attempt']=attempt;save()
  self.modules['entry_contract_r9143'].memory_gate(packets) if 'entry_contract_r9143'in self.modules else self.memory_floor(packets)
  report['last_memory_guard']=attempt;return attempt
 @staticmethod
 def memory_floor(packets):
  values=[pair[0]for p in packets for pair in p['memory']];assert len(values)==32 and min(values)>4<<30;return min(values)
 def read_config(self):return json.loads((self.model/'config.json').read_bytes())
 def mapping(self):return json.loads((self.model/'model.safetensors.index.json').read_bytes())['weight_map']
 def read_head(self,node,loaded):return self.modules['shard_head_contract_r9183'].read_head(self.lib,node,2,loaded)
 def keys(self,keys,mapping,stage,ranges):return self.modules['weight_key_guard_r9156'].summarize_keys(keys,mapping,stage,ranges)
 def all_keys(self,packets,ranges):return self.modules['weight_key_guard_r9156'].validate_all_stages(packets,ranges)
 def load(self,ctx,text,keys,progress):
  self.gate.check('parameter-loader-start')
  return self.modules['parameter_loader_r9143'].load_parameters(self.lib,ctx,self.model,text,keys,self.safe_open,self.torch,self.check,progress)
 def run_pair(self,g,settings,gate,save):
  codec=self.modules['long_codec_r179'].KimiLongCodec(self.model);frozen=json.loads((self.bound['driver']/'frozen_inputs.json').read_bytes())
  assert codec.identity['tokenizer_config_SHA256']==frozen['tokenizer']['tokenizer_config_SHA256']and codec.identity['tiktoken_model_SHA256']==frozen['tokenizer']['tiktoken_model_SHA256']
  self.native=self.modules['long_capture_native'].Native(g,enabled=True,draft_cards=1)
  original_harvest=self.native.harvest
  def harvest_with_first_validation(ids,past,width):
   result=original_harvest(ids,past,width)
   if 'first_capture_validation'not in g['report']:
    assert ids==[1]and past==[0]and width==[512],'First capture must be third-arm first512 seed'
    n=ctypes.c_size_t();self.check(self.lib.modern_kimi_batch_taps_count_r9122(self.native.ctx,ctypes.byref(n)));layers=[]
    for index in range(n.value):
     meta=(ctypes.c_int64*8)();self.check(self.lib.modern_kimi_batch_taps_info_r9122(self.native.ctx,index,meta,8));assert list(meta)==[self.native.epoch,int(meta[1]),512,7168,1,1,0,512];layers.append(int(meta[1]))
    expected={0:[19,37],1:[],2:[66,78,90],3:[]};assert sorted(layers)==expected[self.native.node]
    if self.native.node==0:assert set(result)=={1}and len(result[1])==512*35840*2
    packet=dict(node=self.native.node,epoch=self.native.epoch,layerIDs=sorted(layers),rows=512,local_bytes=len(layers)*512*7168*2,finite_checked_by_immutable_harvest=True)
    packets=self.control.collect('r346-first-capture-validation',packet)
    g['report']['first_capture_validation']=validate_first_capture(packets,self.native.epoch);save()
   return result
  self.native.harvest=harvest_with_first_validation
  gate.check('capture-transport-establish')
  self.modules['capture_transport'].establish(self.native,settings.head,settings.capture_port,settings.nonce,max(1,min(7200,gate.remaining())))
  def connect_once():return connect_draft(settings,gate,self.control,self.modules['publisher'].DraftClient,lambda value:(g['report'].update(value),save()))
  def grade(fixture,ids,finish):return self.modules['grade_output'].grade_output(fixture,ids,finish,codec.decode)
  def pair_save(value):g['report']['pair_report']=value;save()
  gate.check('start-forward-driver337')
  return self.modules['pair_driver'].run_pair(self.native,frozen,connect_once,grade,deadline=gate.deadline,cancelled=gate.cancelled,save=pair_save,enabled=True)
 def affinity(self):return self.modules['model_affinity_check_r2628'].verify(self.lib)
 def close_capture(self):
  if self.native is not None:self.modules['capture_transport'].close(self.native)
 def close_sampler(self):
  if self.sampler is not None:
   self.sampler.close();assert not self.sampler.thread.is_alive(),'Owned CPU memory sampler did not exit'

def execute(settings,gate,deps,report,save):
 """Entire entry state machine; CPU tests replace only real device/file providers."""
 ctx=None;normal=False;failure=None;failure_tb=None;cleanup=[];report.update(completed=False,Native_SHA256=NATIVE_SHA,node=settings.node,target_cards=32,service_cards=33,actual_step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),qualification_scope='Four new same327 diagnostic arms; qualified8df/252508 is the code parent. No latest99/ec full Model qualification inherited.')
 report['runtime_budget']=dict(original_GPU_start_unix=settings.start,Native_deadline_unix=settings.deadline,Native_seconds=28800,GPU_seconds=29400,clock_mapping='original epoch mapped once to monotonic; never restarted')
 try:
  gate.check('entry-before-imports');report.update(deps.prepare(settings,gate));save();lib=deps.lib;control=deps.control
  def check(code):deps.check(code)
  ranges=[]
  for stage in range(2):
   a=(ctypes.c_size_t*2)();check(lib.modern_kimi_engine_partition_r2475(stage,2,a));ranges.append(list(a))
  assert ranges==deps.m['static_memory']['PP_ranges']==[[0,46],[46,93]];report['PP_ranges']=ranges
  cfg=deps.read_config();text=cfg['text_config'];assert text['num_hidden_layers']==93 and text['hidden_size']==7168 and text['vocab_size']==163840 and text['linear_attn_config']['num_heads']==96 and text['linear_attn_config']['head_dim']==128 and text['kv_lora_rank']==512 and text['qk_rope_head_dim']==64
  report['memory_before_create']=deps.memory();assert min(v[0]for v in report['memory_before_create'])>58<<30
  gate.check('before-create');control.collect('r346-before-create',dict(node=settings.node,Native_SHA256=NATIVE_SHA));gate.check('Native-create')
  tick=time.monotonic();ctx=lib.modern_kimi_engine_create_long_agent_r179(json.dumps(cfg).encode(),settings.node,2,2,settings.head.encode(),settings.port)
  if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
  report['create_seconds']=time.monotonic()-tick;report['head_before_weights']=deps.read_head(settings.node,False)
  layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));expected=dict(node=settings.node,local_workers=8,TP=16,TP_node_rank=settings.node%2,PP=2,PP_stage=settings.node//2,sampling_node=2,public_graph_enabled=False,owned_metadata_enabled=False);assert layout==expected;report['layout']=layout
  geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert geometry=={'max_rows':2,'pages':260,'page_size':256,'KDA_pool_rows':3,'pages_per_request':128,'max_total_tokens':32768};report['cache_geometry']=geometry
  cap=json.loads(lib.modern_kimi_batch_capabilities_r9122(ctx));assert cap['logical_capacity']==1 and cap['implemented_max_flatten_tokens']==64 and cap['pool_rows']==3 and cap['flags']==3
  cap=json.loads(lib.modern_kimi_long_agent_capabilities_r179(ctx));assert cap['scoped_long_agent']and cap['max_total_tokens']==32768 and cap['state_pool_rows']==3
  keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));summary=deps.keys(keys,deps.mapping(),settings.node//2,ranges);report['actual_engine_weight_keys']=summary
  report['actual_allstage_weight_keys']=deps.all_keys(control.collect('r346-keys-before-load',dict(node=settings.node,keys=summary)),ranges)
  report['before_load_memory_guard']=deps.memory_guard('before-load',report,save);gate.check('before-safe-open-and-load')
  report['phase']='loading_weights';save()
  def progress(value):report.update(value);save()
  report.update(deps.load(ctx,text,keys,progress));report['head_after_finish']=deps.read_head(settings.node,True)
  report['after_finish_memory_guard']=deps.memory_guard('after-finish',report,save);report['phase']='weights_ready';save();gate.check('before-start-forward')
  control.collect('r346-weights-ready',dict(node=settings.node,loaded_keys=len(keys)))
  g=dict(lib=lib,ctx=ctx,control=control,node=settings.node,tail=settings.node//2==1,emitter=2,vocab=163840,pp=2,tp_nodes=2,report=report,memory_guard=lambda label,reset=False:deps.memory_guard(label,report,save))
  report['pair_report']=deps.run_pair(g,settings,gate,save)
  assert report['pair_report']['completed']and report['pair_report']['normal_draft_protocol_close']and len(report['pair_report']['requests'])==4
  assert all(r['completed']and r['normal_reset']and r['quality']['passed']and r['finish']=='eos'for r in report['pair_report']['requests'])
  report['head_after_pair']=deps.read_head(settings.node,True);report['owner_affinity']=deps.affinity();report['after_pair_memory_guard']=deps.memory_guard('after-pair',report,save)
  report['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));gate.check('before-normal-destroy');control.collect('r346-before-destroy',dict(node=settings.node));normal=True
 except BaseException as e:
  failure=e;failure_tb=e.__traceback__;report.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  for label,action in [('capture',deps.close_capture),('sampler',deps.close_sampler)]:
   try:action();report[label+'_CPU_cleanup_completed']=True
   except BaseException as e:cleanup.append(dict(phase=label,error=repr(e)))
  try:
   if ctx:
    deps.check(deps.lib.modern_kimi_engine_destroy_r2475(ctx));report['all8_owned_workers_shutdown_completed']=True
  except BaseException as e:cleanup.append(dict(phase='Native-destroy',error=repr(e)))
  try:
   if normal and not cleanup:deps.control.collect('r346-after-destroy',dict(node=settings.node,normal=True))
  except BaseException as e:cleanup.append(dict(phase='four-leader-destroy-ack',error=repr(e)))
  try:
   if deps.control is not None:deps.control.close();report['control_CPU_cleanup_completed']=True
  except BaseException as e:cleanup.append(dict(phase='control-close',error=repr(e)))
  report.update(completed=bool(normal and failure is None and not cleanup),cleanup_errors=cleanup,deadline_checks=gate.counts,phase='completed'if normal and failure is None and not cleanup else'failed');save()
 if failure is not None:raise failure.with_traceback(failure_tb)
 if cleanup:raise RuntimeError('Actual owned cleanup failed')
 return report

def main():
 D=Path(__file__).resolve().parent;settings=Settings(os.environ);gate=Gate(settings);gate.check('bootstrap-source')
 # Do not raise between a successful create return and storing its Context.
 # The next explicit boundary handles TERM; the347 outer deadline still kills hangs.
 signal.signal(signal.SIGTERM,lambda *_:setattr(gate,'signal_cancel',True))
 digest=os.environ['KIMI_LONG_TARGET_MANIFEST_SHA_R347'];verify_bundle(D,digest)
 path=settings.root/'results'/('Kimi-long-pair-r346-'+settings.run+'-node'+str(settings.node)+'.json');assert not path.exists()
 report=dict(source_manifest_SHA256=digest,completed=False,HTTP=False,client_SLO_qualified=False)
 def save():
  temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
 save();execute(settings,gate,RealDependencies(D),report,save)
 print('KIMI_LONG_PAIR_R346_NORMAL '+str(settings.node),flush=True)

if __name__=='__main__':main()
