import time as _prefix_time
_PREFIX_START=_prefix_time.monotonic()
import os as _cache_os,sys as _cache_sys
_cache_base=_cache_os.environ['PYTHONPYCACHEPREFIX'];assert _cache_base.startswith('/local/infini-pycache-')
_cache_sys.pycache_prefix=_cache_base+'/'+str(_cache_os.getpid());assert len(_cache_os.sched_getaffinity(0))>=32, 'CPU_BIND_before_Torch_and_weights'
from preflight_r133 import validate as _continuous_validate
_continuous_preflight = _continuous_validate()
from preengine_r133 import validate as _preengine_validate
_preengine = _preengine_validate()
from preweight_protocol_r143b import validate as _protocol_validate
_protocol_preweight = _protocol_validate()
from glm_agent_service_probe_r143b import run as _preloaded_probe_r3091
from model_affinity_check_r2628 import verify as verify_owner_affinity
import base64
import ctypes
import hashlib
import json
import os
import pathlib
import time
PROCESS_START_R8708=_PREFIX_START
import traceback
import socket
import re
from glm_large_chat_r3627 import run_chat_checks
os.environ['GLM_DECODE_ROW_DEDUP_R5386']='0';os.environ['GLM_GROUPED_ROUTED_R5123']='0';os.environ['GLM_ASYNC_PROJECTED_HOST_R5011']='0';os.environ['GLM_GROUP_PROJECTED_HOST_R5030']='0';os.environ['GLM_BATCH_ALLSELECTED_R5096']='1'

import torch
import torch.multiprocessing as mp
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_fp16_accumulation=False
assert torch.get_float32_matmul_precision()=='highest' and not any(getattr(torch.backends.cuda.matmul,n)for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation'])


r = pathlib.Path('/refactor')
# Reuse the exact raw-checkpoint loader and CAPI signatures. Do not execute the
# old diagnostic worker or its __main__ block.
baseline = (r / 'incoming/modern_glm_full_model_probe_r2.py').read_text()
baseline=baseline.replace("stage=int(os.environ['SLURM_PROCID'])","stage=int(os.environ['GLM_STAGE_R5040'])").replace("tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']","tag=os.environ['GLM_RUN_R5040']")
MODEL_LIBRARY_METADATA=_continuous_preflight['model_library_metadata']
LIBRARY=pathlib.Path(MODEL_LIBRARY_METADATA['library_actual_path'])
ns = {}
exec(baseline[:baseline.index('def worker(rank,payload):')].replace("r/'build/libmodern_glm_full_model_probe.so'",repr(str(LIBRARY))),ns)
lib, ptr, check, unique, model, stage, tag = (ns[k] for k in ['lib', 'ptr', 'check', 'unique', 'model', 'stage', 'tag'])
ROWS=int(os.environ['GLM_CACHE_ROWS_R3627']);CAPACITY=int(os.environ['GLM_CACHE_CAP_R3627']);assert(ROWS,CAPACITY)==(32,640);PP=int(os.environ['GLM_PP_SIZE_R3603']);assert PP==2;CARDS=8*PP;BOUNDS=[0,38,78]
cache_overlay=pathlib.Path('/dev/shm')/('iloader-r3604-'+str(os.getuid()))/('overlay-glm-pp'+str(PP)+'-s'+str(stage))
if os.environ.get('USE_WEIGHT_CACHE_R3604')=='1':
 assert cache_overlay.is_dir(),str(cache_overlay);model=cache_overlay;ns['model']=model

lib.modern_glm_model_tp_control.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
lib.modern_glm_pipeline_last_profile.restype = ctypes.POINTER(ctypes.c_double)
lib.modern_glm_direct_calls_r601.argtypes=[ctypes.c_int];lib.modern_glm_direct_calls_r601.restype=ctypes.c_uint64
lib.modern_glm_dense_mode_r48_set.argtypes=[ctypes.c_int]
lib.modern_glm_fp8_calls_r780.argtypes=[ctypes.c_int];lib.modern_glm_fp8_calls_r780.restype=ctypes.c_uint64
lib.modern_glm_fp8_runtime_stats_r809.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
lib.modern_glm_fp8_runtime_error_r809.restype=ctypes.c_char_p
def cache_stats809(device):
    v=(ctypes.c_uint64*10)();code=lib.modern_glm_fp8_runtime_stats_r809(device,v,10)
    if code:raise RuntimeError(lib.modern_glm_fp8_runtime_error_r809().decode())
    return dict(zip(['created','destroyed','hits','misses','evictions','active','active_bytes','highwater_bytes','runs','invalid'],list(v)))

def CPU_memory_snapshot():
    """Metadata only. Ancestor charges include other tasks; not solely this Model."""
    result={'time_ns':time.time_ns(),'pid':os.getpid(),'status':{},'cgroup_ancestors':[]}
    for line in pathlib.Path('/proc/self/status').read_text().splitlines():
        if line.split(':')[0]in ['VmRSS','VmHWM','VmLck','VmPin','Cpus_allowed_list','Mems_allowed_list']:result['status'][line.split(':')[0]]=line.split(':',1)[1].strip()
    groups=pathlib.Path('/proc/self/cgroup').read_text().splitlines()
    for line in groups:
        hierarchy,controllers,path=line.split(':',2)
        if hierarchy=='0'and not controllers:base=pathlib.Path('/sys/fs/cgroup');current=base/path.lstrip('/')
        elif 'memory'in controllers.split(','):base=pathlib.Path('/sys/fs/cgroup/memory');current=base/path.lstrip('/')
        else:continue
        for _ in range(9):
            if not current.is_relative_to(base):break
            row={'path':str(current)}
            for name in ['memory.current','memory.max','memory.usage_in_bytes','memory.limit_in_bytes']:
                p=current/name
                if p.exists():row[name]=p.read_text().strip()
            p=current/'memory.stat'
            if p.exists():row['memory.stat']={k:int(v)for k,v in (s.split()for s in p.read_text().splitlines())if k in ['anon','file','shmem','file_mapped','file_dirty','file_writeback','unevictable','pgfault','pgmajfault','rss','cache','mapped_file','total_rss','total_cache','total_mapped_file']}
            result['cgroup_ancestors'].append(row)
            if current==base:break
            current=current.parent
    return result

lib.modern_glm_cache_view_calls_r927.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_cache_view_calls_r927.restype=ctypes.c_uint64
lib.modern_glm_model_indexer_policy_r998.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_model_decode_leases_policy_r1356.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_model_feature_bypass_r1406.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_indexer_metadata_calls_r1406.argtypes=[ctypes.c_int];lib.modern_glm_indexer_metadata_calls_r1406.restype=ctypes.c_uint64
lib.modern_glm_indexer_select_all_calls_r980.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_indexer_select_all_calls_r980.restype=ctypes.c_uint64
lib.modern_glm_model_owned_prefix_policy_r1669.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_prefix_calls_r1669.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_prefix_calls_r1669.restype=ctypes.c_uint64
policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p
adoption=_preengine['receipts']['results/glm-balanced-micro8-r153-lock.json']
mode=adoption['mode'];assert mode in [0,2]

def phase_marker(phase,rank):
    free,total=torch.cuda.mem_get_info(rank)
    row={'phase':phase,'stage':stage,'rank':rank,'PID':os.getpid(),'driver_monotonic_ns':time.monotonic_ns(),'realtime_ns':time.time_ns(),'free':free,'total':total,'is_peak':False}
    path=r/'results'/f'GLM-Model-phase-r9183-{tag}-stage{stage}-rank{rank}.jsonl'
    with path.open('a') as out:out.write(json.dumps(row)+'\n');out.flush()

def worker(rank, payload):
    torch.cuda.set_device(rank)
    assert torch.cuda.mem_get_info(rank)[0]>58*1024**3
    assert mode==2
    assert lib.modern_glm_dense_mode_r48_set(mode)==0
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    cfg = json.loads((model / 'config.json').read_text())
    buffers = [ctypes.create_string_buffer(base64.b64decode(payload[k] if k != 'tp' else payload[k][stage])) for k in ['world', 'tp', 'control']]
    from glm_crossrequest_environment_r9110 import check_hardware
    hardware=check_hardware(torch,rank)
    phase_marker('create.enter',rank)
    ctx = lib.modern_glm_model_create(rank, stage, ROWS, CAPACITY, json.dumps(cfg).encode(), *buffers)
    if not ctx:
        raise RuntimeError(lib.modern_glm_model_error().decode())
    phase_marker('create.return',rank)
    report={'Torch_matmul_supported_modes':{'float32_precision':torch.get_float32_matmul_precision(),**{n:getattr(torch.backends.cuda.matmul,n)for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']}},'TP':8,'PP':PP,'all_serving_cards':CARDS,'weight_source_path':str(model),'cache_capacity':CAPACITY,'cache_rows':ROWS,'mode':mode,'step':tag,'stage':stage,'rank':rank,'cases':[],'completed':False,'scope':'Current3072 fullGLM multi-request fastpath ABBA; startup markers and bounded1200s sharedpayload visibility wait. All original numeric/owner guards retained; failed768 preserved.'}
    report['hardware_gate_before_create']=hardware
    lib.modern_glm_cross_control_r5775.argtypes=[ctypes.c_int]*4
    lib.modern_glm_cross_journal_r5775.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_cross_journal_r5775.restype=ctypes.c_char_p
    identity_error=None;identity=None
    try:
        rc=lib.modern_glm_cross_control_r5775(rank,0,0,-1);assert rc==0
        raw=lib.modern_glm_cross_journal_r5775(rank,1);assert raw
        identity=json.loads(raw);assert sorted(v['layer']for v in identity['identities'])==list(range(BOUNDS[stage],BOUNDS[stage+1]))
        assert all(v['stage']==stage and v['rank']==rank for v in identity['identities'])
    except BaseException as exc:identity_error=repr(exc)
    ivote=torch.tensor([float(identity_error is None)],device='cuda');torch.cuda.synchronize(rank)
    rc=lib.modern_glm_model_control(ctx,1,ctypes.c_void_p(ivote.data_ptr()),stream());assert rc==0 and ivote.item()==CARDS,('all16registrybeforeWeightIO',identity_error)
    report['registry_owner_before_weightIO']={'identities':identity,'all16passed':True,'synchronous_direct_CAPI_owner_thread':True}

    lib.modern_glm_partition_report_r3854.argtypes=[ctypes.c_void_p];lib.modern_glm_partition_report_r3854.restype=ctypes.c_char_p;part=json.loads(lib.modern_glm_partition_report_r3854(ctx));assert[part['begin'],part['end']]==BOUNDS[stage:stage+2]and part['PP']==2;report['actual_partition']=part
    lib.modern_glm_unloaded_trim_r4052.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    constructor_keys_before=lib.modern_glm_model_keys(ctx)
    invalid=(ctypes.c_uint64*6)(*([2**64-1]*6));assert lib.modern_glm_unloaded_trim_r4052(ctx,invalid,5)!=0 and list(invalid)==[2**64-1]*6
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1)as pool:foreign=pool.submit(lib.modern_glm_unloaded_trim_r4052,ctx,invalid,6).result()
    assert foreign!=0 and list(invalid)==[2**64-1]*6;report['constructor_trim_invalid_foreign_owner_rejected']=True
    trim_values=(ctypes.c_uint64*6)();check(lib.modern_glm_unloaded_trim_r4052(ctx,trim_values,6));assert list(trim_values)[3:]==[rank,stage,PP]
    torch.cuda.synchronize(rank);torch.cuda.empty_cache();report['idle_constructor_trim_r5394']=list(trim_values);assert lib.modern_glm_model_keys(ctx)==constructor_keys_before;report['constructor_trim_keys_unchanged']=True
    lib.modern_glm_exact_banks_stats_r4188.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];exact_stats=(ctypes.c_uint64*4)();check(lib.modern_glm_exact_banks_stats_r4188(ctx,exact_stats,4));calls=4*(BOUNDS[stage+1]-BOUNDS[stage]);enabled=int(os.environ['GLM_EXACT_BANKS_R4188']);assert enabled in[0,1]and list(exact_stats)[:3]==[calls,calls*enabled,calls*(1-enabled)];report['exact_bank_dispatch_r4188']=list(exact_stats);report['exact_bank_enabled']=bool(enabled);report['factorial_variant_r5394']=os.environ['GLM_VARIANT_R4363'];report['requested_channel_count_r5394']=int(os.environ['GLM_CHANNELS_R4363'])
    lib.modern_glm_service_memory_stats_r4279.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def service_stats():
        v=(ctypes.c_uint64*9)();check(lib.modern_glm_service_memory_stats_r4279(ctx,v,9));return list(v)
    report['service_memory_after_create_r4279']=service_stats();s4279=report['service_memory_after_create_r4279'];assert s4279[:3]==[1,1,640]and s4279[5]==s4279[6]and s4279[5]>0 and s4279[7:]==[2,163840],s4279
    report['library_sha256']=hashlib.sha256(LIBRARY.read_bytes()).hexdigest();assert report['library_sha256']==MODEL_LIBRARY_METADATA['library_SHA256']
    report['native_library_sha256']=hashlib.sha256((r/'candidates/glm-hybrid-decode-r5386/libnative.so').read_bytes()).hexdigest();assert report['native_library_sha256']==_preengine['receipts']['results/glm-hybrid-decode-model-r5390-build.json']['native_library_sha256']
    lib.modern_glm_projected_host_counts_r4504.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def host_stats():
        a=(ctypes.c_uint64*7)();check(lib.modern_glm_projected_host_counts_r4504(rank,a,7));return list(a)
    host_created=host_stats();host_layers=BOUNDS[stage+1]-BOUNDS[stage];assert host_created==[host_layers,0,2*ROWS*CAPACITY*8*256*2*host_layers,0,0,0,0],host_created
    report['host_bank_after_create_r4504']=host_created
    report['memory_before_communicator_warmup']=torch.cuda.mem_get_info(rank);warm=[]
    for elements in[1,1024,65536,262144]:
        x=torch.ones(elements,device='cuda',dtype=torch.float32);check(lib.modern_glm_model_tp_control(ctx,elements,ptr(x),stream()));assert bool((x==8).all().item())
        y=torch.ones(elements,device='cuda',dtype=torch.float32);check(lib.modern_glm_model_control(ctx,elements,ptr(y),stream()));assert bool((y==CARDS).all().item());warm.append({'elements':elements,'TP8_integer_SUM_exact':True,'world16_integer_SUM_exact':True});del x,y
    torch.cuda.synchronize(rank);torch.cuda.empty_cache();report['preload_communicator_warmup']=warm
    report['service_memory_after_warmup_r4279']=service_stats()
    report['memory_after_create']=torch.cuda.mem_get_info(rank);early=torch.tensor([float(report['memory_after_create'][0]>8*1024**3)],device='cuda');check(lib.modern_glm_model_control(ctx,1,ptr(early),stream()))
    if early.item()!=CARDS:
        report['owner_affinity']=verify_owner_affinity(lib,rank);report.update(capacity_rejected_before_disk_loading=True,completed=False);report['service_memory_before_destroy_r4279']=service_stats();check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown());report['all_contexts_closed_after_memory_rejection']=True;(r/'results'/f'glm-large-preflight-r5394-{tag}-stage{stage}-rank{rank}.json').write_text(json.dumps(report,indent=2));return
    # The existing loader is executed verbatim and updates this report.
    loader = baseline[baseline.index(" sdk=sorted("):baseline.index(' def control(x):')].replace('hygon-runtime-fixes-r1', 'hygon-glm-5f91ad7-r3672')
    assert ' check(lib.modern_glm_model_finish(ctx));' in loader
    loader=loader.replace(' check(lib.modern_glm_model_finish(ctx));'," phase_marker('raw_weight_load.return.before_finish',rank)\n phase_marker('finish.enter',rank)\n check(lib.modern_glm_model_finish(ctx));phase_marker('finish.return',rank);")
    wrapped = 'def load(ctx, report):\n' + loader + '\n return report\n'
    loader_ns = dict(ns)
    loader_ns.update(stage=stage, rank=rank, report=report, ctx=ctx,phase_marker=phase_marker)
    exec(wrapped, loader_ns)
    report['CPU_memory_before_weight_load']=CPU_memory_snapshot();load_begin=time.perf_counter()
    phase_marker('load.enter',rank)
    loader_ns['load'](ctx, report)
    phase_marker('load.return',rank)
    report['actual_owner_weight_load_wall_seconds']=time.perf_counter()-load_begin
    report['CPU_memory_after_weight_load']=CPU_memory_snapshot()
    invalid=(ctypes.c_uint64*6)(*([2**64-1]*6));assert lib.modern_glm_unloaded_trim_r4052(ctx,invalid,6)!=0 and list(invalid)==[2**64-1]*6;report['unloaded_trim_after_loading_rejected']=True
    keys=set(json.loads(lib.modern_glm_model_keys(ctx)))
    layers=sorted({int(k.split('.')[2])for k in keys if k.startswith('model.layers.')});bounds=BOUNDS
    assert layers==list(range(bounds[stage],bounds[stage+1]));report['pipeline_boundaries']=bounds;report['actual_layer_indices']=layers

    sdk_paths=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfiniops.so'in line})
    assert len(sdk_paths)==1 and '/install/hygon-glm-5f91ad7-r3672/lib/' in sdk_paths[0]
    report['actual_sdk_sha256']=hashlib.sha256(pathlib.Path(sdk_paths[0]).read_bytes()).hexdigest()
    assert report['actual_sdk_sha256']==_preengine['receipts']['results/latest-sdk-r3672-GLM-build.json']['library_sha256']
    report['library_sha256'] = hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    assert report['library_sha256']==MODEL_LIBRARY_METADATA['library_SHA256']
    report['LM_source_commit']='5f91ad738583db8f5617aeae53b76523f85d1ec6'
    report['scope']='Actual MMLU random32 zero-shot chat pilot and 8 varied EOS requests per OFF/ON arm; existing618 fuse selector, no output-token equality gate;16cards same load.'
    report['scope']='Actual MMLU random32 zero-shot chat pilot and 8 varied EOS requests per OFF/ON arm; existing618 fuse selector, no output-token equality gate;16cards same load.'
    mappings={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfinilm_runtime.so'in v or '_infinilm.cpython'in v};assert mappings and all('/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/'in p for p in mappings);report['actual_LM_native_libraries']={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()for p in mappings}
    report['scope']='Actual MMLU random32 zero-shot chat pilot and 8 varied EOS requests per OFF/ON arm; existing618 fuse selector, no output-token equality gate;16cards same load.'
    report['scope']='Actual MMLU random32 zero-shot chat pilot and 8 varied EOS requests per OFF/ON arm; existing618 fuse selector, no output-token equality gate;16cards same load.'
    report.update(old_precision_compatibility_qualified=False, numerical_variant='780_per128K_BF16_GEMM_F32dots_Mge32', component_report='glm-fp8-batched-r776-summary.json')
    report['cache_library_sha256']=_preengine['receipts']['results/glm-hybrid-decode-model-r5390-build.json']['fresh_helpers']['helpers/libglm_fp8_cache_r807.so']
    report['same_variant_references']={}
    report['fp8_helper_sha256']=hashlib.sha256((r/'candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so').read_bytes()).hexdigest()
    assert report['fp8_helper_sha256']==_preengine['receipts']['results/model-latest-r3675-GLM-build.json']['fresh_helpers']['helpers/libglm_fp8_batched_r776.so']
    report['lowbatch_manifest_SHA256_r8219'] = _continuous_preflight['manifest_SHA256']
    report_path = r / 'results' / f'glm-agent-service-r143b-{tag}-stage{stage}-rank{rank}.json'

    def control(x):
        check(lib.modern_glm_model_control(ctx, x.numel(), ptr(x), stream()))

    def tp_control(x):
        check(lib.modern_glm_model_tp_control(ctx, x.numel(), ptr(x), stream()))

    lib.modern_glm_model_batch_cache_policy_r3068.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.modern_glm_model_batch_cache_stats_r3068.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def batch_stats3075():
        a=(ctypes.c_uint64*5)();check(lib.modern_glm_model_batch_cache_stats_r3068(ctx,a,5));return list(a)
    lib.modern_glm_model_m15_oproj_policy_r125.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.modern_glm_m15_oproj_stats_r125.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def m15_stats129():
        a=(ctypes.c_uint64*17)();check(lib.modern_glm_m15_oproj_stats_r125(rank,a,17));return list(a)
    initial15=m15_stats129();assert initial15==[0]*17,'M15 defaultOFF/newcacheempty'
    assert lib.modern_glm_model_m15_oproj_policy_r125(ctx,2)!=0 and m15_stats129()==initial15
    sentinel15=(ctypes.c_uint64*17)(*[77]*17);assert lib.modern_glm_m15_oproj_stats_r125(rank,sentinel15,16)!=0 and list(sentinel15)==[77]*17
    import threading
    foreign15=[];thread15=threading.Thread(target=lambda:foreign15.append(lib.modern_glm_model_m15_oproj_policy_r125(ctx,1)));thread15.start();thread15.join(20);assert not thread15.is_alive()and foreign15==[1]and m15_stats129()==initial15
    report.update(M15_default_OFF_verified=True,M15_invalid_wrongthread_untouched=True)
    initial=batch_stats3075();assert initial==[0]*5,('not defaultOFF',initial)
    assert lib.modern_glm_model_batch_cache_policy_r3068(ctx,2)!=0 and batch_stats3075()==initial
    sentinel=(ctypes.c_uint64*5)(77,77,77,77,77)
    assert lib.modern_glm_model_batch_cache_stats_r3068(ctx,sentinel,4)!=0 and list(sentinel)==[77]*5
    import threading
    codes=[]
    wrong=threading.Thread(target=lambda:codes.append(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1)))
    wrong.start();wrong.join(20);assert not wrong.is_alive()and codes==[1]and batch_stats3075()==initial
    report.update(batch_cache_default_OFF_verified=True,batch_cache_wrong_thread_and_invalid_untouched=True)
    helper_maps={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libglm_fp8_'in v};assert len(helper_maps)==2 and all('/candidates/glm-grouped-native-model-r5117/helpers/'in v for v in helper_maps)
    report['actual_FP8_helpers']={v:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest()for v in helper_maps}
    report['Python_bytecode_cache_prefix']=_cache_sys.pycache_prefix;report['probe_preimport_before_CCL_verified']=True
    report['benchmark_program_SHA256']=payload['benchmark_SHA256'];report['natural_cases']=[];report['remaining_process_seconds']=8500-(time.monotonic()-PROCESS_START_R8708)
    vocab=cfg['vocab_size'];barrier=torch.ones(1,device='cuda');control(barrier);assert barrier.item()==CARDS
    report['owner_affinity']=verify_owner_affinity(lib,rank)
    _preloaded_probe_r3091(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,lambda x:phase_marker(x,rank))
    report['BLAS_policy']=json.loads(policy.modern_blas_policy_r243_json());assert report['BLAS_policy']['failures']==0 and all(v['created']==v['verified']for v in report['BLAS_policy']['owners'])
    report['actual_producer']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];report['host_bank_before_destroy_r4504']=host_stats()
    report['cache_before_destroy']=cache_stats809(rank);report['service_memory_before_destroy_r4279']=service_stats();check(lib.modern_glm_model_destroy(ctx));report['cache_after_destroy']=cache_stats809(rank);assert report['cache_after_destroy']['created']==report['cache_after_destroy']['destroyed']and report['cache_after_destroy']['active']==report['cache_after_destroy']['active_bytes']==0
    report['observer_native_destroy_returned0']=True;report['M15_after_destroy']=m15_stats129();released15=report['M15_after_destroy'];assert released15[0]==0 and released15[7]==released15[8]and released15[12]==released15[13]==0 and released15[5]==released15[16]==0
    report['host_bank_after_destroy_r4504']=host_stats()
    # Preserve lifecycle evidence even when a later check fails. Activity may be zero on resident-K/V paths.
    report_path.write_text(json.dumps(report,indent=2))
    assert report['host_bank_after_destroy_r4504'][:3]==[host_layers,host_layers,0], ('host-bank-lifecycle',report['host_bank_after_destroy_r4504'])
    assert all(x>=0 for x in report['host_bank_after_destroy_r4504'][3:]), ('host-bank-activity',report['host_bank_after_destroy_r4504'])
    assert report.get('cohort_phases_completed')or report.get('normal_numeric_rejection')or report.get('resident_candidate_rejected')
    registry=lib.modern_glm_cross_journal_r5775(rank,1);assert registry and not json.loads(registry)['identities'],'registry dtor cleanup';check(lib.modern_glm_model_shutdown());torch.cuda.synchronize(rank);assert torch.isfinite(torch.ones(8,device='cuda')).all().item();torch.cuda.synchronize(rank);report.update(completed=True,worker_shutdown_completed=True,post_close_Torch_liveness=True,serving_target_qualified=False);report_path.write_text(json.dumps(report,indent=2))

def guarded(rank, payload):
    try:
        worker(rank, payload)
    except BaseException:
        traceback.print_exc()
        pp=r/'results'/('GLM-agent-service-error-r143b-'+tag+'-s'+str(stage)+'-r'+str(rank)+'.json');pp.write_text(json.dumps({'error':traceback.format_exc(),'run':tag,'stage':stage,'rank':rank}))
        os._exit(2)


if __name__ == '__main__':
    p=r/'control'/f'glm-agent-service-r143b-control-{tag}-ids.json'
    if stage==0:
        program_path=r/'incoming/GLM-prefill-Model-r9241/program-r9241.json';raw=program_path.read_bytes();program=json.loads(raw);assert len(program['service_cases'])==64 and program['max_context']==640;expected=json.loads((pathlib.Path(__file__).parent/'CPU-result.json').read_text());assert hashlib.sha256(raw).hexdigest()==expected['program_SHA256']
        payload={'world':unique(),'tp':[unique()for _ in range(PP)],'control':unique(),'benchmark':program,'benchmark_SHA256':hashlib.sha256(raw).hexdigest()};assert not p.exists();tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(p)
    begin=time.monotonic()
    while not p.exists():
        if time.monotonic()-begin>1200:raise TimeoutError('GLM benchmark identities missing')
        time.sleep(.1)
    mp.spawn(guarded,args=(json.loads(p.read_bytes()),),nprocs=8,join=True);print('GLM_AGENT_SERVICE_R143_NORMAL',stage,flush=True)
