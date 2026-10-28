"""Actual new32K Native constructor, two legal Loading resets, then destroy.
No checkpoint payload, finish, forward, generation, substitute comm or math.
"""
import argparse,ctypes,json,os,pathlib,socket,time,sys
from binding_r183 import bind,maps,sha
from memory_r183 import MemorySampler
from stage_control_r2386 import StageControl
from shard_head_contract_r9183 import read_head
from weight_key_guard_r9156 import summarize_keys,validate_all_stages
D=pathlib.Path(__file__).resolve().parent.parent;R=pathlib.Path('/refactor');NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];FLOOR=4*1024**3

def cache_plan(cfg,stage):
 t=cfg['text_config'];kda={int(i)-1 for i in t['linear_attn_config']['kda_layers']};begin,end=[[0,46],[46,93]][stage];k=sum(begin<=i<end for i in kda);mla=end-begin-k
 return {'scope':'SOURCE_DERIVED_AFTER_REAL_RESET_SUCCESS_NOT_PER_ALLOCATION_READBACK','layers':end-begin,'KDA_layers':k,'MLA_layers':mla,'MLA_each_shape':[2,260,256,6,192],'MLA_each_bytes':306708480,'KDA_conv_each_shape':[3,2304,3],'KDA_conv_each_bytes':41472,'KDA_ssm_each_shape':[3,6,128,128],'KDA_ssm_each_bytes':589824,'cache_payload_bytes_per_owner':mla*306708480+k*(41472+589824),'actual_per_allocation_storage_readback':'UNKNOWN','allocator_and_driver_overhead':'UNKNOWN'}

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);x=a.parse_args();mf=D/'source-manifest-r183.json';assert sha(mf)==x.manifest_sha;m=json.loads(mf.read_bytes())
 for name,h in m['source_SHA256'].items():assert sha(D/name)==h,name
 pins=json.loads((D/'pins-r183.json').read_bytes());assert sha(pins['CPU_receipt'])==pins['CPU_receipt_SHA256'];cpu=json.loads(pathlib.Path(pins['CPU_receipt']).read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU']and not cpu['Native_dlopen']and cpu['new_HOST_TUs']==6 and cpu['new_HIP_TUs']==0 and cpu['native']==pins['Native']and cpu['native_SHA256']==pins['Native_SHA256']
 for p,h in cpu['all_linked_object_SHA256'].items():assert sha(p)==h,p
 model=pathlib.Path(pins['model']);assert sha(model/'config.json')==pins['config_SHA256']and sha(model/'model.safetensors.index.json')==pins['index_SHA256'];cfg=json.loads((model/'config.json').read_bytes());mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
 node=int(os.environ['SLURM_PROCID']);assert socket.gethostname()==NODES[node]and os.environ['SLURM_JOB_ID']=='910380';step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert int(os.environ['SLURM_CPUS_PER_TASK'])==64 and len(os.sched_getaffinity(0))>=64
 assert os.environ['INFINILM_KIMI_SPEC_R3632']=='1'and os.environ['INFINILM_KIMI_PAIR_TP16_R9006']=='1'and os.environ['INFINILM_KIMI_SPEC_MATH_R3676']=='0'and os.environ['NCCL_NET']=='IBext_v8'
 import torch
 import infinilm
 torch.set_num_threads(1);assert torch.cuda.device_count()==8
 for device in range(8):
  prop=torch.cuda.get_device_properties(device);assert prop.total_memory>=63*1024**3 and prop.multi_processor_count==80
  assert 'gfx936'in prop.name or'gfx936'in getattr(prop,'gcnArchName','')
 def drain():
  for device in range(8):torch.cuda.synchronize(device)
 def memory():return [list(torch.cuda.mem_get_info(device))for device in range(8)]
 nonce=os.environ['KIMI_ENGINE_NONCE'];control=StageControl(node,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,nonce,4,840);path=R/'results'/('Kimi-long-capacity-r183-'+step+'-node%d.json'%node);assert not path.exists()
 report={'node':node,'host':socket.gethostname(),'actual_step':step,'CPU_receipt_SHA256':pins['CPU_receipt_SHA256'],'Native_SHA256':pins['Native_SHA256'],'source_manifest_SHA256':x.manifest_sha,'completed':False,'passed':False,'full93_model_parameter_constructor':'UNKNOWN_UNTIL_CREATE_AND_KEYS_PASS','initial_create_cache_allocation':'UNKNOWN_NOT_USED_AS_CACHE_FIT_PROOF','actual_cache_constructed_after_first_reset':False,'weight_payload_opened':False,'finish_calls':0,'model_forward_calls':0,'reset_call_count':0,'reset_weight_state':'Loading_without_payload_or_finish','full_Model_fit':'UNKNOWN_STOP','GPU_Model_qualified':False,'exact_transient_peak':'UNKNOWN','Native_pool_reserved':'UNKNOWN','phases':{}}
 ctx=None;lib=None;sampler=None;destroy_ok=False
 def save():
  temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
 def record_phase(name):
  snap=sampler.snapshot();assert not snap['sampler_error'];snap['observed_floor_passed']=all(v[0]>=FLOOR for v in snap['observed_min_memory']);report['phases'][name]=snap;return snap['observed_floor_passed']
 def ck(code):
  if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 def safe_error(error):return {'type':type(error).__name__,'message':str(error).replace(nonce,'[REDACTED]')[:1500]}
 try:
  lib,providers=bind(pins);report['actual_CAPI_providers']=providers;report['actual_loaded_libraries_before_create']=maps(pins);report['memory_before_create']=memory();assert min(v[0]for v in report['memory_before_create'])>58*1024**3
  control.collect('pre-create-pins',{'node':node,'Native_SHA256':pins['Native_SHA256'],'source_manifest_SHA256':x.manifest_sha})
  sampler=MemorySampler(torch,.01);sampler.reset();report['phase']='actual_long_create';save();local_ok=False
  try:
   started=time.monotonic();ctx=lib.modern_kimi_engine_create_long_agent_r179(json.dumps(cfg).encode(),node,2,2,os.environ['KIMI_ENGINE_HEAD'].encode(),int(os.environ['GLOBAL_PORT_R2354']))
   if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
   drain();report['create_wall_s']=time.monotonic()-started;report['layout']=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert report['layout']=={'node':node,'local_workers':8,'TP':16,'TP_node_rank':node%2,'PP':2,'PP_stage':node//2,'sampling_node':2,'public_graph_enabled':False,'owned_metadata_enabled':False}
   caps=json.loads(lib.modern_kimi_long_agent_capabilities_r179(ctx));assert caps['scoped_long_agent']and caps['logical_capacity']==1 and caps['physical_rows']==2 and caps['state_pool_rows']==3 and caps['max_total_tokens']==32768;report['long_capabilities']=caps
   geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert geometry=={'max_rows':2,'pages':260,'page_size':256,'KDA_pool_rows':3,'pages_per_request':128,'max_total_tokens':32768};report['cache_config_CAPI']=geometry
   report['head_before_weights']=read_head(lib,node,2,False);ranges=[]
   for stage in range(2):out=(ctypes.c_size_t*2)();ck(lib.modern_kimi_engine_partition_r2475(stage,2,out));ranges.append(list(out))
   assert ranges==[[0,46],[46,93]];keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));report['parameter_key_ownership']=summarize_keys(keys,mapping,node//2,ranges);report['full93_model_parameter_constructor']=True
   report['constructor_memory_floor_observed_passed']=record_phase('constructor');local_ok=report['constructor_memory_floor_observed_passed']
  except BaseException as error:
   report['create_error']=safe_error(error)
   try:record_phase('constructor_failed')
   except BaseException as snap_error:report['constructor_snapshot_error']=safe_error(snap_error)
  packets=control.collect('create-status',{'node':node,'ok':local_ok,'keys':report.get('parameter_key_ownership')});all_create_ok=all(q['ok']for q in packets);save()
  if all_create_ok:
   report['all93_key_ownership']=validate_all_stages(packets,[[0,46],[46,93]])
  all_resets_ok=all_create_ok
  for ordinal in [1,2]:
   local_reset_ok=False
   if all_resets_ok:
    sampler.reset();report['phase']='first_real_cache_reset'if ordinal==1 else 'same_shape_reset';save()
    try:
     started=time.monotonic();ck(lib.modern_kimi_batch_reset_r9122(ctx,0));ck(lib.modern_kimi_batch_mode_r9122(ctx,0));drain();report['reset_call_count']+=1
     label='first_reset'if ordinal==1 else 'same_shape_reset';report[label+'_wall_s']=time.monotonic()-started;assert json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx))==report['cache_config_CAPI'];local_reset_ok=record_phase(label)
     metrics=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));assert metrics['engine_calls']==0;report['output_metrics']=metrics
     if ordinal==1:report['actual_cache_constructed_after_first_reset']=True;report['cache_sizes_counts_source_derived']=cache_plan(cfg,node//2)
    except BaseException as error:
     report['reset%d_error'%ordinal]=safe_error(error)
     try:record_phase(('first_reset'if ordinal==1 else 'same_shape_reset')+'_failed')
     except BaseException as snap_error:report['reset_snapshot_error']=safe_error(snap_error)
   packets=control.collect('reset%d-status'%ordinal,{'node':node,'ok':local_reset_ok});all_resets_ok=all(q['ok']for q in packets);save()
  report['actual_loaded_libraries_after_resets']=maps(pins);report['probe_body_passed']=all_resets_ok and report['reset_call_count']==2
 finally:
  report['phase']='destroying';save()
  try:
   if ctx and lib is not None:
    if sampler:sampler.reset()
    started=time.monotonic();ck(lib.modern_kimi_engine_destroy_r2475(ctx));ctx=None;drain();destroy_ok=True;report['destroy_wall_s']=time.monotonic()-started;report['all8_owned_workers_shutdown_completed']=True
    if sampler:report['destroy_observed_floor_passed']=record_phase('after_destroy')
  except BaseException as error:report['destroy_error']=safe_error(error)
  finally:
   if sampler:sampler.close()
   try:
    final=control.collect('destroy-status',{'node':node,'all8_normal':destroy_ok});report['all32_normal_shutdown']=all(q['all8_normal']for q in final)
   except BaseException as error:report['final_barrier_error']=safe_error(error);report['all32_normal_shutdown']=False
   
   try:report['memory_after_destroy']=memory()
   except BaseException as error:report['memory_after_destroy']='UNKNOWN';report['after_destroy_memory_error']=safe_error(error)
   report['destroy_residual_attribution']='UNKNOWN; process exit and post-KFD prove final process cleanup, not immediate Native pool/Torch/driver attribution';report['completed']=bool(report.get('all32_normal_shutdown'));report['passed']=bool(report.get('probe_body_passed')and report['completed']and report.get('destroy_observed_floor_passed'));report['phase']='completed'if report['passed']else'failed_with_cleanup_evidence';save();control.close()
 if report['completed']:print('KIMI_LONG_CAPACITY_R183_NORMAL',node,flush=True)
 return 0 if report['passed']else 1
if __name__=='__main__':raise SystemExit(main())
