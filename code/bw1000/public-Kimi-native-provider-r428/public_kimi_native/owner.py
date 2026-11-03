"""Manifest-driven private modern batch Model entry; no historical chat/tokenidentity gate."""
import os,pathlib,json,time,ctypes,sys,traceback,socket
from .runtime.entry_contract_r9143 import validate,verify_inputs,geometry,memory_gate,sha
from .runtime.native_binding_r9143 import bind,verify_maps
from .runtime.parameter_loader_r9143 import load_parameters
from .runtime.weight_key_guard_r9156 import summarize_keys,validate_all_stages
from .runtime.stage_control_r2386 import StageControl
from .runtime.model_affinity_check_r2628 import verify as verify_owner_affinity
from .runtime.host_observer_r9183 import HostObserver,TimedControl,Reporter
from .runtime.shard_head_contract_r9183 import read_head
from .runtime.guard_snapshot_r9225 import persist_memory_gate

from .request_loop import serve
from .cleanup import close_owner
from .admission import admit


def run_owner(config, prepared, stop, on_ready):
    assert os.environ.get('KIMI_ENTRY_R9143')=='1','defaultOFF'
    assert os.environ.get('KIMI_CAPACITY_ONLY_R9145','0')=='0','capacitymode0or1'
    r=pathlib.Path(config['runtime_directory']);r.mkdir(parents=True,exist_ok=True)
    manifest_path=pathlib.Path(config['model_manifest']);assert sha(manifest_path)==config['model_manifest_SHA256']
    manifest=json.loads(manifest_path.read_text());t=validate(manifest)
    assert t=={'tp_nodes':2,'pp':2,'logical_capacity':1}
    assert manifest['native_DSO_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
    model,b=verify_inputs(manifest);tp_nodes=2;pp=2;capacity=1
    node=config['node'];assert 0<=node<4 and len(set(manifest['nodes']))==4
    assert socket.gethostname().split('.')[0]==manifest['nodes'][node]
    assert int(os.environ['SLURM_PROCID'])+int(os.environ.get('GLOBAL_OFFSET_R2354','0'))==node
    assert os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']==config['actual_step']
    stage=node//2;tail=stage==1;emitter=2;head=config['head'];port=config['native_port'];tag=config['run']
    experiment_started_r9046=time.monotonic();g={};primary=None
    assert os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'and os.environ['INFINILM_KIMI_SPEC_R3632']=='1'and os.environ['INFINILM_KIMI_PAIR_TP16_R9006']=='1'and os.environ['INFINILM_KIMI_SPEC_MATH_R3676']=='0'
    # Batchprovider ON switch fixed beforecase; percase GROUPED/EXEC decides OFF/ON.
    os.environ['INFINILM_KIMI_BATCH_I192_R9125']='1'
    import torch
    from safetensors import safe_open
    import infinilm
    torch.set_num_threads(1);assert torch.cuda.device_count()==8
    for d in range(8):
     p=torch.cuda.get_device_properties(d);assert ('gfx936'in p.name or'gfx936'in getattr(p,'gcnArchName',''))and p.total_memory>=63*1024**3 and p.multi_processor_count==80

    from .runtime import stage_control_r2386 as _private_stage_control_r205
    STAGE_CONTROL_WAIT_SECONDS_R194=7200
    assert 1<=STAGE_CONTROL_WAIT_SECONDS_R194<=7200,'existing StageControl constructor bound; not overall runtime deadline'
    path=r/('Kimi-public-r428-'+tag+'-node%d.json'%node);assert not path.exists()
    control=StageControl(node,head,port+7000,config['nonce'],len(manifest['nodes']),STAGE_CONTROL_WAIT_SECONDS_R194);report={'node':node,'run':tag,'TP':16,'PP':pp,'PP_stage':stage,'sampling_node':emitter,'logical_capacity':capacity,'manifest_SHA256':sha(manifest_path),'public_source_manifest_SHA256':config['source_manifest_SHA256'],'completed':False,'GPU_Model_qualification_pending':True};ctx=None;sampler=None

    gpu_start_unix_r205=float(config['gpu_start_epoch']);native_deadline_unix_r205=gpu_start_unix_r205+config['Native_seconds']
    native_absolute_deadline_monotonic_r205=time.monotonic()+max(0.,native_deadline_unix_r205-time.time())
    report['runtime_budget']={'GPU_start_UTC_unix':gpu_start_unix_r205,'Native_deadline_UTC_unix':native_deadline_unix_r205,'Native_inner_s':config['Native_seconds'],'cleanup_reserve_s':config['Native_seconds']-config['service_seconds'],'clock_mapping':'common keeper-issued GPUstart epoch, mapped once to local monotonic; no per-phase reset'}
    host_observer=HostObserver();control=TimedControl(control,host_observer);reporter=Reporter(path,report,host_observer)
    def save():reporter.full()
    def progress_save(value):reporter.progress(value)
    set_logging_diag=reporter.set_logging_diag
    def memory():return [list(torch.cuda.mem_get_info(d))for d in range(8)]
    try:
     for k,v in manifest['required_environment'].items():assert os.environ.get(k)==v,k
     from .runtime.memory_guard_r9144 import MemorySampler
     sampler=MemorySampler(torch,interval=.01)
     def memory_guard(label,reset=False):
      guard_start=time.perf_counter()
      try:
       if reset:sampler.reset()
       snap=sampler.snapshot();assert not snap['sampler_error'];packets=control.collect('memory-guard:'+label,snap);return persist_memory_gate(label,packets,memory_gate,report,save)
      finally:host_observer.add('memory_guard',time.perf_counter()-guard_start)
     report['memory_before_create']=memory();assert min(v[0]for v in report['memory_before_create'])>58*1024**3
     participants=admit(control,config,sha(manifest_path))
     library,lib,providers=bind(manifest,b);report['actual_CAPI_providers']=providers;report['actual_loaded_libraries_before_create']=verify_maps(manifest);assert (node==0)==(prepared is not None);report['actual_loaded_libraries_after_server_import']=verify_maps(manifest);control.collect('agent-preweight-imports-ok',{'node':node,'passed':True})
     def check(code):
      if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
     ranges=[]
     for p in range(pp):
      a=(ctypes.c_size_t*2)();check(lib.modern_kimi_engine_partition_r2475(p,pp,a));ranges.append(list(a))
     report['actual_partition_CAPI']=ranges;save();assert ranges==manifest['static_memory']['PP_ranges'],('actualDSOpartition_vs_manifest',ranges,manifest['static_memory']['PP_ranges']);report['PP_ranges']=ranges;cfg=json.loads((model/'config.json').read_text());text=cfg['text_config'];assert text['num_hidden_layers']==93,('num_hidden_layers',text['num_hidden_layers']);assert text['hidden_size']==7168,('hidden_size',text['hidden_size']);assert text['linear_attn_config']['num_heads']==96,('KDAheads',text['linear_attn_config']['num_heads']);assert text['linear_attn_config']['head_dim']==128,('KDAhead_dim',text['linear_attn_config']['head_dim']);assert text['kv_lora_rank']==512,('kv_lora_rank',text['kv_lora_rank']);assert text['qk_rope_head_dim']==64,('qk_rope_head_dim',text['qk_rope_head_dim']);vocab=text['vocab_size']
     report['phase']='creating';save();tick=time.perf_counter();ctx=lib.modern_kimi_engine_create_long_agent_r179(json.dumps(cfg).encode(),node,tp_nodes,pp,head.encode(),port)
     if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
     report['create_seconds']=time.perf_counter()-tick;report['head_before_weights']=read_head(lib,node,pp,False);control.collect('allowner-shard-head-geometry',{'node':node,'head':report['head_before_weights']});layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert layout=={'node':node,'local_workers':8,'TP':16,'TP_node_rank':node%2,'PP':pp,'PP_stage':stage,'sampling_node':emitter,'public_graph_enabled':False,'owned_metadata_enabled':False};report['layout']=layout
     actual=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert actual==geometry(capacity);report['cache_geometry']=actual;cap=json.loads(lib.modern_kimi_batch_capabilities_r9122(ctx));assert cap['logical_capacity']==capacity and cap['implemented_max_flatten_tokens']==64 and cap['pool_rows']==2*capacity+1 and cap['flags']==3;report['actual_batch_capabilities']=cap;long_cap=json.loads(lib.modern_kimi_long_agent_capabilities_r179(ctx));assert long_cap['scoped_long_agent']and long_cap['max_total_tokens']==32768 and long_cap['state_pool_rows']==3;report['actual_long_capabilities']=long_cap
     keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];key_summary=summarize_keys(keys,mapping,stage,ranges);report['actual_engine_weight_keys']=key_summary;key_packets=control.collect('allowner-actual-engine-keys-before-load',{'node':node,'keys':key_summary});report['actual_allstage_weight_keys']=validate_all_stages(key_packets,ranges);report['model_weight_keys_validated']=True
     report['memory_after_create']=memory();packets=control.collect('allowner-memory-floor',{'memory':report['memory_after_create']});report['preload_memory_gate']={'minimum_free_bytes':memory_gate(packets),'required_free_bytes':4*1024**3};report['capacity_gate_passed']=True;report['capacity_only_mode']=os.environ.get('KIMI_CAPACITY_ONLY_R9145','0')=='1';report['weight_payload_opened']=False;report['forward_calls_before_loading']=0;report['phase']='capacity_validated';save();control.collect('allowners-capacity-validated',{'node':node,'capacity':capacity,'layout':layout,'geometry':actual,'minimum_free_bytes':report['preload_memory_gate']['minimum_free_bytes']})
     projection={'schema':1,'run':tag,'node':node,'globalnode':node,'host':socket.gethostname(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'manifest_SHA256':sha(manifest_path),'native_DSO_SHA256':manifest['native_DSO_SHA256'],'capacity_gate_passed':True,'capacity_only_mode':report['capacity_only_mode'],'phase':'capacity_probe_closing'if report['capacity_only_mode']else'loading_weights','weight_payload_opened_at_publish':False,'inference_forward_calls_at_publish':0,'layout':layout,'geometry':actual,'PP_ranges':ranges,'all_target_min_free_bytes':report['preload_memory_gate']['minimum_free_bytes'],'required_free_bytes':4*1024**3,'GPU_Model_qualified':False,'actual_target_owners':8*len(manifest['nodes']),'actual_shard_head_geometry_validated':True,'model_weight_keys_validated':report['model_weight_keys_validated'],'actual_engine_key_stage_range':key_summary['actual_range'],'actual_all93_coverage_validated':report['actual_allstage_weight_keys']['all93_layer_ids_exactly_once_across_PPstages']}
     cp=r/('Kimi-public-capacity-r428-'+tag+'-node%d.json'%node);assert not cp.exists();tmp=cp.with_suffix('.tmp');tmp.write_text(json.dumps(projection,indent=2));tmp.replace(cp)

     if report['capacity_only_mode']:
      report['capacity_only_passed']=True;report['capacity_probe_before_destroy_passed']=True;report['scope']='PREWEIGHT_CAPACITY_DIAGNOSTIC_NO_FORWARD_NO_MODEL_PERF';control.collect('capacity-before-destroy',{})
     else:
      report['phase']='loading_weights';report['weight_payload_opened']=True;save()
      def progress(x):report.update(x);save()
      report.update(load_parameters(lib,ctx,model,text,keys,safe_open,torch,check,progress));report['head_after_finish']=read_head(lib,node,pp,True);control.collect('allowner-shard-head-loaded-rows',{'node':node,'head':report['head_after_finish']});report['memory_after_finish']=memory();report['after_finish_memory_gate']=memory_guard('after-finish');report['actual_loaded_libraries']=verify_maps(manifest);report['library_sha256']=sha(library);report['phase']='weights_ready';save();control.collect('weights-ready',{'memory':report['memory_after_finish'],'loaded_keys':len(keys)});control.set_timeout(7200)
      g=locals().copy();g['service_deadline_monotonic']=time.monotonic()+max(0.,gpu_start_unix_r205+config['service_seconds']-time.time());on_ready();serve(g,prepared,stop);assert report.get('serving_protocol_complete');report['head_after_protocol']=read_head(lib,node,pp,True);report['owner_affinity']=verify_owner_affinity(lib);report['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));report['protocol_before_destroy_passed']=True;control.collect('before_destroy',{})
    except BaseException as e:
     primary=e;report['error']=repr(e);report['traceback']=traceback.format_exc()
     try:save()
     except BaseException as persistence:
      if hasattr(e,'add_note'):e.add_note('Primary report persistence: '+repr(persistence))
     raise
    finally:
     close_owner(ctx=ctx,lib=locals().get('lib'),check=locals().get('check'),sampler=sampler,
         control=control,events=g.get('model_event_driver'),report=report,save=save,primary=primary)
    print('KIMI_PUBLIC_R428_NORMAL',node,flush=True)
