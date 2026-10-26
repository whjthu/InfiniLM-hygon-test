"""Manifest-driven private modern batch Model entry; no historical chat/tokenidentity gate."""
import os,pathlib,json,time,ctypes,sys,traceback,socket
sys.path.append('/refactor/incoming')
from entry_contract_r9143 import validate,verify_inputs,geometry,memory_gate,sha
from native_binding_r9143 import bind,verify_maps
from parameter_loader_r9143 import load_parameters
from stage_control_r2386 import StageControl
from model_affinity_check_r2628 import verify as verify_owner_affinity
assert os.environ.get('KIMI_ENTRY_R9143')=='1','defaultOFF'
assert os.environ.get('KIMI_CAPACITY_ONLY_R9145','0')in ['0','1'],'capacitymode0or1'
r=pathlib.Path('/refactor');manifest_path=pathlib.Path(os.environ['KIMI_ENTRY_MANIFEST_R9143']);manifest=json.loads(manifest_path.read_text());t=validate(manifest);model,b=verify_inputs(manifest);frontend=pathlib.Path(manifest['frontend_directory']).resolve();sys.path.insert(0,str(frontend));import target_protocol;assert pathlib.Path(target_protocol.__file__).resolve()==frontend/'target_protocol.py';tp_nodes=t['tp_nodes'];pp=t['pp'];capacity=t['logical_capacity'];node=int(os.environ['SLURM_PROCID'])+int(os.environ.get('GLOBAL_OFFSET_R2354','0'));assert 0<=node<len(manifest['nodes'])and socket.gethostname()==manifest['nodes'][node];stage=node//tp_nodes;tail=stage==pp-1;emitter=(pp-1)*tp_nodes;head=os.environ['KIMI_ENGINE_HEAD'];port=int(os.environ['GLOBAL_PORT_R2354']);tag=os.environ['GLOBAL_RUN_R2354'];experiment_started_r9046=time.monotonic()
assert os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'and os.environ['INFINILM_KIMI_SPEC_R3632']=='1'and os.environ['INFINILM_KIMI_PAIR_TP16_R9006']=='1'and os.environ['INFINILM_KIMI_SPEC_MATH_R3676']=='0'
# Batchprovider ON switch fixed beforecase; percase GROUPED/EXEC decides OFF/ON.
os.environ['INFINILM_KIMI_BATCH_I192_R9125']='1'
import torch
from safetensors import safe_open
import infinilm
torch.set_num_threads(1);assert torch.cuda.device_count()==8
for d in range(8):
 p=torch.cuda.get_device_properties(d);assert ('gfx936'in p.name or'gfx936'in getattr(p,'gcnArchName',''))and p.total_memory>=63*1024**3 and p.multi_processor_count==80
 with torch.cuda.device(d):assert torch.arange(2048,device='cuda',dtype=torch.float32).sum().item()==2047*2048//2
control=StageControl(node,head,port+7000,os.environ['KIMI_ENGINE_NONCE'],len(manifest['nodes']),7200);path=r/'results'/('Kimi-modern-PP4-r9145-'+tag+'-node%d.json'%node);assert not path.exists();report={'node':node,'run':tag,'TP':16,'PP':pp,'PP_stage':stage,'sampling_node':emitter,'logical_capacity':capacity,'manifest_SHA256':sha(manifest_path),'completed':False,'GPU_Model_qualification_pending':True};ctx=None;sampler=None

def save():
 p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(path)
def memory():return [list(torch.cuda.mem_get_info(d))for d in range(8)]
try:
 for k,v in manifest['required_environment'].items():assert os.environ.get(k)==v,k
 from memory_guard_r9144 import MemorySampler
 sampler=MemorySampler(torch)
 def memory_guard(label,reset=False):
  if reset:sampler.reset()
  snap=sampler.snapshot();assert not snap['sampler_error'];packets=control.collect('memory-guard:'+label,snap);minimum=memory_gate(packets);result={'label':label,'minimum_free_bytes':minimum,'required_free_bytes':4*1024**3,'owners':packets};report['last_memory_guard']=result;return result
 report['memory_before_create']=memory();assert min(v[0]for v in report['memory_before_create'])>58*1024**3
 participants=control.collect('entry-before-create',{'manifest':sha(manifest_path),'node':node});assert all(v['manifest']==sha(manifest_path)for v in participants)
 library,lib,providers=bind(manifest,b);report['actual_CAPI_providers']=providers;report['actual_loaded_libraries_before_create']=verify_maps(manifest)
 def check(code):
  if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 ranges=[]
 for p in range(pp):
  a=(ctypes.c_size_t*2)();check(lib.modern_kimi_engine_partition_r2475(p,pp,a));ranges.append(list(a))
 assert ranges==manifest['static_memory']['PP_ranges'];report['PP_ranges']=ranges;cfg=json.loads((model/'config.json').read_text());text=cfg['text_config'];assert text['num_hidden_layers']==93;assert text['hidden_size']==7168 and text['linear_attn_config']['num_heads']==96 and text['linear_attn_config']['head_dim']==128 and text['kv_lora_rank']==512 and text['qk_rope_head_dim']==64;vocab=text['vocab_size']
 report['phase']='creating';save();tick=time.perf_counter();ctx=lib.modern_kimi_engine_create_batch_r9122c(json.dumps(cfg).encode(),node,tp_nodes,pp,head.encode(),port,capacity)
 if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 report['create_seconds']=time.perf_counter()-tick;layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert layout=={'node':node,'local_workers':8,'TP':16,'TP_node_rank':node%2,'PP':pp,'PP_stage':stage,'sampling_node':emitter,'public_graph_enabled':False,'owned_metadata_enabled':False};report['layout']=layout
 actual=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert actual==geometry(capacity);report['cache_geometry']=actual;cap=json.loads(lib.modern_kimi_batch_capabilities_r9122(ctx));assert cap['logical_capacity']==capacity and cap['implemented_max_flatten_tokens']==64 and cap['pool_rows']==2*capacity+1 and cap['flags']==3;report['actual_batch_capabilities']=cap
 report['memory_after_create']=memory();packets=control.collect('allowner-memory-floor',{'memory':report['memory_after_create']});report['preload_memory_gate']={'minimum_free_bytes':memory_gate(packets),'required_free_bytes':4*1024**3};report['capacity_gate_passed']=True;report['capacity_only_mode']=os.environ.get('KIMI_CAPACITY_ONLY_R9145','0')=='1';report['weight_payload_opened']=False;report['forward_calls_before_loading']=0;report['phase']='capacity_validated';save();control.collect('allowners-capacity-validated',{'node':node,'capacity':capacity,'layout':layout,'geometry':actual,'minimum_free_bytes':report['preload_memory_gate']['minimum_free_bytes']})
 projection={'schema':1,'run':tag,'node':node,'globalnode':node,'host':socket.gethostname(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'manifest_SHA256':sha(manifest_path),'native_DSO_SHA256':manifest['native_DSO_SHA256'],'capacity_gate_passed':True,'capacity_only_mode':report['capacity_only_mode'],'phase':'capacity_probe_closing'if report['capacity_only_mode']else'loading_weights','weight_payload_opened_at_publish':False,'inference_forward_calls_at_publish':0,'layout':layout,'geometry':actual,'PP_ranges':ranges,'all64_min_free_bytes':report['preload_memory_gate']['minimum_free_bytes'],'required_free_bytes':4*1024**3,'GPU_Model_qualified':False}
 cp=r/'results'/('Kimi-PP4-capacity-r9145-'+tag+'-node%d.json'%node);assert not cp.exists();tmp=cp.with_suffix('.tmp');tmp.write_text(json.dumps(projection,indent=2));tmp.replace(cp)

 if report['capacity_only_mode']:
  report['capacity_only_passed']=True;report['capacity_probe_before_destroy_passed']=True;report['scope']='PREWEIGHT_CAPACITY_DIAGNOSTIC_NO_FORWARD_NO_MODEL_PERF';control.collect('capacity-before-destroy',{})
 else:
  report['phase']='loading_weights';report['weight_payload_opened']=True;save()
  def progress(x):report.update(x);save()
  keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));report.update(load_parameters(lib,ctx,model,text,keys,safe_open,torch,check,progress));report['memory_after_finish']=memory();report['after_finish_memory_gate']=memory_guard('after-finish');report['actual_loaded_libraries']=verify_maps(manifest);report['library_sha256']=sha(library);report['phase']='weights_ready';save();control.collect('weights-ready',{'memory':report['memory_after_finish'],'loaded_keys':len(keys)});control.set_timeout(900)
  target_protocol.run(globals());assert report.get('serving_protocol_complete');report['owner_affinity']=verify_owner_affinity(lib);report['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));report['protocol_before_destroy_passed']=True;control.collect('before_destroy',{})
except BaseException as e:report['error']=repr(e);report['traceback']=traceback.format_exc();save();raise
finally:
 try:
  if sampler is not None:sampler.close()
  if ctx:check(lib.modern_kimi_engine_destroy_r2475(ctx));report['all8_owned_workers_shutdown_completed']=True
  if report.get('protocol_before_destroy_passed')or report.get('capacity_probe_before_destroy_passed'):control.collect('after_destroy',{})
  report['completed']=bool(report.get('protocol_before_destroy_passed')or report.get('capacity_probe_before_destroy_passed'));report['phase']='completed'if report['completed']else'failed'
 except BaseException as e:report['shutdown_error']=repr(e);report['completed']=False;raise
 finally:save();control.close()
print('KIMI_MODERN_BATCH_R9143_NORMAL',node,flush=True)
