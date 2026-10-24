import pathlib as _manifest_pathlib,json as _manifest_json,hashlib as _manifest_hashlib
_manifest_root=_manifest_pathlib.Path('/refactor')
_manifest=_manifest_json.loads((_manifest_root/'incoming/Kimi-TP16PP2-I192-manifest-r8669za.json').read_text())
for _name,_hash in _manifest['source_sha256'].items():
 assert _manifest_hashlib.sha256((_manifest_root/'incoming'/_name).read_bytes()).hexdigest()==_hash,_name
for _name,_hash in _manifest['remote_evidence_sha256'].items():
 assert _manifest_hashlib.sha256((_manifest_root/_name).read_bytes()).hexdigest()==_hash,_name
from Kimi_load_cgroup_probe_r8470 import probe as _probe_cgroup
_cgroup=_probe_cgroup()
_cgroup_name='Kimi-TP16PP2-I192-cgroup-r8669za-'+__import__('os').environ['GLOBAL_RUN_R2354']+'-node'+__import__('os').environ['SLURM_PROCID']+'.json'
with (_manifest_root/'results'/_cgroup_name).open('x') as _cgroup_file:_manifest_json.dump(_cgroup,_cgroup_file,indent=2)
import os as _os,sys as _sys
assert _os.environ.get('INFINILM_OWNER_CPU_WIDTH_R2392')=='4',('owner_CPU_width_early',_os.environ.get('INFINILM_OWNER_CPU_WIDTH_R2392'))
assert _os.environ['PYTHONPYCACHEPREFIX'].startswith('/local/')
_sys.pycache_prefix=_os.environ['PYTHONPYCACHEPREFIX']+'/'+str(_os.getpid())
from kimi_TP16PP2_I192_protocol_r8669za import run as _preimported_reduce_probe
from model_affinity_check_r2628 import verify as verify_owner_affinity
import collections,ctypes,hashlib,json,os,pathlib,time,threading,signal
import numpy as np
import torch
from Kimi_memory_domains_r8152 import snapshot as memory_domains
from safetensors import safe_open
from stage_control_r2386 import StageControl
import infinilm

torch.set_num_threads(1)
r=pathlib.Path('/refactor');model=pathlib.Path(os.environ.get('KIMI_MODEL_VIEW_R3679','/shared/Kimi-K3'));node=int(os.environ['SLURM_PROCID'])+int(os.environ['GLOBAL_OFFSET_R2354']);tp_nodes=int(os.environ['KIMI_TP_NODES']);pp=int(os.environ['KIMI_PP_SIZE']);participants=tp_nodes*pp;assert(tp_nodes,pp)in[(1,4),(2,2),(4,1),(4,2),(6,1)]
stage=node//tp_nodes;tail=stage==pp-1;emitter=(pp-1)*tp_nodes;tag=os.environ['GLOBAL_RUN_R2354'];head=os.environ['KIMI_ENGINE_HEAD'];port=int(os.environ['GLOBAL_PORT_R2354'])
control=StageControl(node,head,port+7000,os.environ['KIMI_ENGINE_NONCE'],participants,7200);control.collect('before_create',{'node':node,'TP':8*tp_nodes,'PP':pp})
path=r/'results'/f'kimi-TP16PP2-I192-r8669za-{tag}-node{node}.json';assert not path.exists();report={'node':node,'step':tag,'TP':8*tp_nodes,'PP':pp,'PP_stage':stage,'nodes':participants,'all_serving_cards':participants*8,'sampling_node':emitter,'completed':False,'target_achieved':False,'cases':[],'chat_cases':[],'natural_cases':[],'scope':'Fresh93ordinary dualvocabhead OFF/ON; B1M1 matched-only/B16M4 historical+QA, completeIDs/EOS required; no inherited TPS.'}
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def memory():return [list(torch.cuda.mem_get_info(d))for d in range(8)]
report.update(numerical_variant='1429_q_a_q_b_prefill_F32_then_BF16',old_precision_raw_compatibility_claimed=False)
report['LM_source_commit']='5f91ad738583db8f5617aeae53b76523f85d1ec6'
report['loading_control_policy']={'initial_timeout_seconds':7200,'after_weights_ready_seconds':900,'prior_run':'r2361 failed waiting for uneven weight loading, no QA or performance qualification','math_ownership_and_QA_guards_unchanged':True}
owned_policy=2
tile_policy=0
report['dense_prefill_optin']=os.environ.get('INFINILM_DENSE_PREFILL_F32_R2475')=='1';report['candidate_source']='CPU209privateI192native11b41ca9-defaultOFF';report['scope']='TP16PP2 B16M16 fullModel I192OFF/ON; nohistoricalcrossTPgold. Sameprocessoriginalfallback full128 reference, candidate finalIDs/EOS+QA first; ordinaryONNO thenrestoreQA. Two512prompts repeated, noModelTPSuntilaudit.'
assert os.environ.get('INFINILM_DENSE_PREFILL_F32_R2475')in ['0','1']
report['SDK_tile_optin']=os.environ.get('INFINILM_MXFP4_TILE_TP_R2559')=='1'
report['loader_transaction_policy']={'max_parameter_keys':16384,'max_source_bytes':2*1024**3,'baseline_max_keys':2048,'baseline_max_bytes':512*1024**2,'owner_LoadBatch_capacity':256,'checkpoint_bytes_dtype_shape_and_fullQA_unchanged':True,'CPU_copy_build_receipt':'common-latest-r3673-Kimi-build.json'}
report['scope']='TP16PP2 B16M16 fullModel I192OFF/ON; nohistoricalcrossTPgold. Sameprocessoriginalfallback full128 reference, candidate finalIDs/EOS+QA first; ordinaryONNO thenrestoreQA. Two512prompts repeated, noModelTPSuntilaudit.'
ctx=None;feedback=None;feedback_hub=None;save()
try:
 report['phase']='hardware_preflight';report['hardware']=[];report['memory_before_create']=memory();assert all(free>58*1024**3 for free,total in report['memory_before_create'])
 for d in range(8):
  props=torch.cuda.get_device_properties(d);assert ('gfx936'in props.name or'gfx936'in getattr(props,'gcnArchName',''))and props.total_memory>=63*1024**3 and props.multi_processor_count==80
  with torch.cuda.device(d):
   x=torch.arange(2048,device='cuda',dtype=torch.float32);assert x.sum().item()==2047*2048//2;torch.cuda.synchronize();del x
  report['hardware'].append({'device':d,'name':props.name,'memory':props.total_memory,'CU':props.multi_processor_count,'small_compute_passed':True})
 preflight=json.loads((r/'results/kimi-resident-r2493-preflight.json').read_text());choice=next(v for v in preflight['topologies']if v['TP']==8*tp_nodes and v['PP']==pp);assert choice['all_stages_parameter_plus_6GiB_reserve_fits_64GiB'];assert choice==_manifest['static_memory']['topology'];report['static_memory_scope']='ExactsameTP16PP2retainedheaderbudget d70d62,derivedmetadata notnewruntimefit';report['planned_resident_memory']=choice['stages'][stage]
 manifest=json.loads((r/'incoming/Kimi-head-Model-manifest-r8235.json').read_text())
 for name,h in manifest['source_sha256'].items():assert sha(r/'incoming'/name)==h,name
 preparation=json.loads((r/'incoming/r5034-Kimi-packed-preparation.json').read_text())
 for name,h in preparation['source_sha256'].items():assert sha(r/'incoming'/name)==h,name
 assert sha(r/'results/kimi-packed-W2-Model-r5049-build.json')==preparation['build_record_remote_SHA256']
 build=json.loads((r/'results/kimi-packed-W2-Model-r5049-build.json').read_text())
 assert build['completed']and build['CPU_contract']['valid_metadata_cases']==108 and build['CPU_contract']['invalid_metadata_cases']==397
 parent_library=pathlib.Path(build['library']);assert sha(parent_library)==preparation['model_sha256']==build['library_SHA256'];torch.cuda.set_device(0)
 from Kimi_TP16PP2_I192_binding_r8669za import load as load_N4_Model
 library,lib,n4_counter_r8167,N4_binary_gate=load_N4_Model(r);report['N4_binary_gate']=N4_binary_gate
 metadata_gate_path=r/'incoming'/preparation['component_gate'];assert sha(metadata_gate_path)==preparation['component_gate_sha256']
 metadata_helper_path=r/'candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so';assert sha(metadata_helper_path)==preparation['helper_sha256']=='85a54291b8aedd1e59556013ab38d15cc077940e70dd8dfd70acbe50b78c25e4'
 metadata_helper=ctypes.CDLL(str(metadata_helper_path));metadata_counter=metadata_helper.modern_grouped_metadata_counts_r5858;metadata_counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];metadata_counter.restype=ctypes.c_int
 bounded_counter=metadata_helper.modern_bounded_counts_r6054;bounded_counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];bounded_counter.restype=ctypes.c_int
 actual_helper_paths={line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libkimi_grouped_prefill.so'in line};assert actual_helper_paths=={str(metadata_helper_path)}
 assert os.environ.get('INFINILM_KIMI_METADATA_PACK_R5858')=='0'and os.environ.get('INFINILM_KIMI_BOUNDED_BF16_R6054')=='0'
 assert os.environ.get('INFINILM_KIMI_STATIC_DIV_R6020','0')=='0'
 packed_counter=lib.modern_kimi_packed_counts_r6136;packed_counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];packed_counter.restype=ctypes.c_int
 for device in range(8):
  values=(ctypes.c_uint64*8)();assert packed_counter(device,values,8)==0 and list(values)==[0]*8
 for dev,n in[(-1,8),(8,8),(0,0),(0,7),(0,9)]:
  values=(ctypes.c_uint64*9)(*([0xfeed]*9));assert packed_counter(dev,values,n)!=0 and list(values)==[0xfeed]*9
 assert packed_counter(0,None,8)!=0
 assert os.environ.get('INFINILM_KIMI_PACKED_AUTO_R6136')=='0' and os.environ.get('INFINILM_KIMI_PACKED_W2_R6122')=='0'
 packed_gate=r/'incoming'/preparation['packed_component_gate'];assert sha(packed_gate)==preparation['packed_component_gate_sha256']
 report['packed_preengine']={'counter_width':8,'all8_initial_zero':True,'invalid_no_write':True,'gate_SHA256':sha(packed_gate),'actual_GPU_branch_readback_requested':False,'ordinary_only':True}
 report['helper_preengine']={'path':str(metadata_helper_path),'sha256':sha(metadata_helper_path),'component_gate_sha256':sha(metadata_gate_path),'metadata_fixedOFF':True,'bounded_BF16_initial_OFF':True}
 save()
 assert os.environ.get('INFINILM_KIMI_MOE_WORKSPACE_R4665')=='1' and os.environ.get('INFINILM_KIMI_MOE_EXECUTION_R4665')=='0'
 assert b'INFINILM_KIMI_MOE_WORKSPACE_R4665\x00'in library.read_bytes()and b'INFINILM_KIMI_MOE_EXECUTION_R4665\x00'in library.read_bytes()
 report['serving_constructor_contract']={'workspace':1,'execution_initial':0,'workspace_cap':64,'bytes_per_layer':19148544,'library_sha256':sha(library)};save()
 for method in ['error','create','keys','load','finish','forward','rank_logits','reset','destroy','partition','output_metrics','resize_cache','cache_geometry','layout']:assert getattr(lib,'modern_kimi_engine_'+method+'_r2475')
 lib.modern_kimi_engine_error_r2475.restype=ctypes.c_char_p
 def check(code):
  if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 lib.modern_kimi_engine_create_r2475.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_size_t];lib.modern_kimi_engine_create_r2475.restype=ctypes.c_void_p
 for name in ['keys','cache_geometry','output_metrics']:
  f=getattr(lib,'modern_kimi_engine_'+name+'_r2475');f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_char_p
 for name in ['finish','reset','destroy','rank_logits']:getattr(lib,'modern_kimi_engine_'+name+'_r2475').argtypes=[ctypes.c_void_p]
 lib.modern_kimi_engine_load_r2475.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.modern_kimi_engine_forward_r2475.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]+[ctypes.c_void_p]*9
 lib.modern_kimi_owned_stats_r1213.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
 def owned_stats():
  values=[]
  for d in range(8):
   v=(ctypes.c_uint64*12)();check(lib.modern_kimi_owned_stats_r1213(d,v));values.append(list(v))
  return values
 lib.modern_kimi_tiled_calls_r1861.argtypes=[ctypes.c_int];lib.modern_kimi_tiled_calls_r1861.restype=ctypes.c_uint64
 lib.modern_kimi_q_prefill_fp32_calls_r1429.argtypes=[ctypes.c_int];lib.modern_kimi_q_prefill_fp32_calls_r1429.restype=ctypes.c_uint64
 lib.modern_kimi_engine_partition_r2475.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]
 ranges=[]
 for p in range(pp):
  pair=(ctypes.c_size_t*2)();check(lib.modern_kimi_engine_partition_r2475(p,pp,pair));ranges.append(list(pair))
 assert [n for a,b in ranges for n in range(a,b)]==list(range(93));report['PP_ranges']=ranges
 cfg=json.loads((model/'config.json').read_text());text=cfg['text_config'];assert text['hidden_size']==7168 and text['num_attention_heads']==96 and text['qk_nope_head_dim']==128 and text['qk_rope_head_dim']==64 and text['v_head_dim']==128 and text['kv_lora_rank']==512 and text['q_lora_rank']==1536 and text['linear_attn_config']['num_heads']==96 and text['linear_attn_config']['head_dim']==128;vocab=text['vocab_size'];report['phase']='creating';save();begin=time.perf_counter();ctx=lib.modern_kimi_engine_create_r2475(json.dumps(cfg).encode(),node,tp_nodes,pp,head.encode(),port,16)
 if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 report['create_seconds']=time.perf_counter()-begin;lib.modern_kimi_engine_layout_r2475.argtypes=[ctypes.c_void_p];lib.modern_kimi_engine_layout_r2475.restype=ctypes.c_char_p;layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert layout=={'node':node,'local_workers':8,'TP':8*tp_nodes,'TP_node_rank':node%tp_nodes,'PP':pp,'PP_stage':stage,'sampling_node':emitter,'public_graph_enabled':False,'owned_metadata_enabled':False};report['layout']=layout
 geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert geometry=={'max_rows':16,'pages':68,'page_size':256,'KDA_pool_rows':17,'pages_per_request':4,'max_total_tokens':1024};report['cache_geometry']=geometry;report['memory_after_create_domains']=memory_domains(torch,'after-create');report['memory_after_create']=memory();assert all(free>3*1024**3 for free,total in report['memory_after_create']),('insufficient_workspace_reserve_after_create',report['memory_after_create'])
 keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert keys<=set(mapping);report['parameter_keys']=len(keys);report['phase']='loading_weights';save();groups=collections.defaultdict(list)
 for key in keys:groups[mapping[key]].append(key)
 pending=[];pending_bytes=loaded=flushes=0;start=time.perf_counter()
 def flush():
  global pending_bytes,loaded,flushes
  if not pending:return
  descriptors=[{'name':name,'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]}for name,w in pending];check(lib.modern_kimi_engine_load_r2475(ctx,json.dumps(descriptors).encode()));loaded+=len(pending);pending.clear();pending_bytes=0;flushes+=1
  if flushes%16==0:report.update(loaded_keys=loaded,loading_elapsed_seconds=time.perf_counter()-start);save();print('KIMI_GLOBAL_R1482_LOAD',node,loaded,len(keys),round(time.perf_counter()-start,3),flush=True)
 for filename,names in sorted(groups.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as shard:
   for name in sorted(names):
    w=shard.get_tensor(name)
    if name.endswith('.self_attn.A_log'):w=w[:text['linear_attn_config']['num_heads']]
    w=w.contiguous();pending.append((name,w));pending_bytes+=w.numel()*w.element_size()
    if len(pending)>=16384 or pending_bytes>=2*1024*1024*1024:flush()
   flush()
 check(lib.modern_kimi_engine_finish_r2475(ctx));assert loaded==len(keys);report.update(loaded_keys=loaded,loading_seconds=time.perf_counter()-start,memory_after_finish=memory())
 maps=pathlib.Path('/proc/self/maps').read_text().splitlines()
 for part,expected in [('libinfiniops.so',json.loads((r/'results/latest-sdk-r3672-Kimi-build.json').read_text())['library_sha256']),('libkimi_grouped_prefill.so','85a54291b8aedd1e59556013ab38d15cc077940e70dd8dfd70acbe50b78c25e4')]:
  paths={line.split()[-1]for line in maps if part in line};assert len(paths)==1;actual=sha(pathlib.Path(next(iter(paths))));assert actual==expected;report[part+'_sha256']=actual
 report['memory_after_finish_domains']=memory_domains(torch,'after-finish');report['library_sha256']=sha(library);report['phase']='weights_ready';save();control.collect('weights_ready',{'node':node,'layout':layout,'memory':report['memory_after_finish']});control.set_timeout(900);print('KIMI_GLOBAL_R1482_WEIGHTS_READY',node,report['loading_seconds'],flush=True)
 mappings={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfinilm_runtime.so'in v or '_infinilm.cpython'in v};assert mappings and all('/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/'in p for p in mappings);report['actual_LM_native_libraries']={p:sha(pathlib.Path(p))for p in mappings};save()
 rtproof=json.loads((r/'results/common-latest-r3673-Kimi-build.json').read_text());assert rtproof['completed'];assert[sha(pathlib.Path(p))for p in mappings if'libinfinilm_runtime.so'in p]==[rtproof['runtime_sha256']];cpylib=ctypes.CDLL(next(p for p in mappings if'libinfinilm_runtime.so'in p));cpylib.modern_cpu_copy_stats_r2515.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];values=(ctypes.c_uint64*5)();assert cpylib.modern_cpu_copy_stats_r2515(values,5)==0;report['CPU_copy_stats_after_loading']=list(values);assert values[0]>0 and values[1]>0;report['loader_flushes']=flushes;report['source_parameter_keys_loaded']=loaded;save()
 sdk=ctypes.CDLL(str(r/'install/hygon-kimi-5f91ad7-r3672/lib/libinfiniops.so'));sdk.modern_mxfp4_tile_tp_counts_r2559.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def sdk_tile_stats():
  values=(ctypes.c_uint64*8)();assert sdk.modern_mxfp4_tile_tp_counts_r2559(values,8)==0;return list(values)
 report['SDK_tile_initial_counts']=sdk_tile_stats();assert report['SDK_tile_initial_counts']==[0]*8
 from kimi_TP16PP2_I192_protocol_r8669za import run
 run(globals())
 assert report.get('serving_protocol_complete')
 report['owner_affinity']=verify_owner_affinity(lib);report['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));report['protocol_before_destroy_passed']=True;control.collect('before_destroy',{})

except BaseException as failure:
 report['error']=repr(failure);save();raise
finally:
 try:
  if ctx:check(lib.modern_kimi_engine_destroy_r2475(ctx));report['all8_owned_workers_shutdown_completed']=True
  if report.get('protocol_before_destroy_passed'):control.collect('after_destroy',{})
  report['completed']=bool(report.get('protocol_before_destroy_passed'));report['phase']='completed'if report['completed']else'failed'
 except BaseException as failure:report['shutdown_error']=repr(failure);report['completed']=False;raise
 finally:
  try:
   if feedback:feedback.close()
   if feedback_hub:feedback_hub.close()
  finally:save();control.close()
print('KIMI_SPEC_R3634_COMPLETE',node,flush=True)
