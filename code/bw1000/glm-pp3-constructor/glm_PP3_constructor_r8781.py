"""One unloaded constructor per owner. No checkpoint tensor I/O or Model forward."""
import base64,ctypes,hashlib,json,os,pathlib,time,traceback
import torch
import torch.multiprocessing as mp
ROOT=pathlib.Path('/refactor');BOUNDS=(0,26,54,78)
BUILD=ROOT/'candidates/glm-PP3-B64-r8780/CPU-build.json'
RECEIPT=json.loads(BUILD.read_text());LIBRARY=pathlib.Path(RECEIPT['library'])
LIBRARY_SHA='99b99b3ab56de735cbff177df532af0a7ce4836d6af25ce5109dfc5c997d4862'
assert RECEIPT['completed'] and RECEIPT['library_SHA256']==LIBRARY_SHA
assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==LIBRARY_SHA
STAGE=int(os.environ['SLURM_PROCID']);RUN=os.environ['GLM_RUN_R5040']
assert STAGE in range(3) and os.environ['GLM_PP_SIZE_R3603']=='3'

def api():
    lib=ctypes.CDLL(str(LIBRARY));P=ctypes.c_void_p;S=ctypes.c_size_t
    lib.modern_glm_model_error.restype=ctypes.c_char_p
    lib.modern_glm_model_id_size.restype=S;lib.modern_glm_model_id.argtypes=[P]
    lib.modern_glm_model_create.argtypes=[ctypes.c_int]*2+[S]*2+[ctypes.c_char_p]+[P]*3
    lib.modern_glm_model_create.restype=P
    for n in ['modern_glm_model_destroy','modern_glm_model_shutdown']:
        getattr(lib,n).argtypes=[P] if n.endswith('destroy') else []
    for n in ['modern_glm_partition_report_r3854','modern_glm_constructor_census_r4158']:
        getattr(lib,n).argtypes=[P];getattr(lib,n).restype=ctypes.c_char_p
    lib.modern_glm_unloaded_trim_r4052.argtypes=[P,ctypes.POINTER(ctypes.c_uint64),S]
    for n in ['modern_glm_model_control','modern_glm_model_tp_control']:
        getattr(lib,n).argtypes=[P,S,P,P]
    return lib

def check(lib,code):
    if code:raise RuntimeError(lib.modern_glm_model_error().decode())

def snapshot(rank):
    torch.cuda.synchronize(rank);free,total=torch.cuda.mem_get_info(rank)
    return {'free':free,'total':total,'device_total_minus_free':total-free,
            'Torch_allocated':torch.cuda.memory_allocated(rank),
            'Torch_reserved':torch.cuda.memory_reserved(rank),
            'native_allocator_allocated_reserved':'UNKNOWN: no such exported counter; device allocation includes native Runtime/CCL',
            'observation_is_peak':False,'driver_monotonic_ns':time.monotonic_ns()}

def worker(rank,payload):
    started=time.monotonic();lib=api();ctx=None
    path=ROOT/'results'/f'GLM-PP3-constructor-r8781-{RUN}-stage{STAGE}-rank{rank}.json'
    assert not path.exists();report={'run':RUN,'stage':STAGE,'rank':rank,'participant':STAGE*8+rank,
        'library_SHA256':LIBRARY_SHA,'scope':'unloaded constructor only; no finish/resident/forward/weights I/O',
        'weights_IO':False,'forward_calls':0,'resident_calls':0,'finished':False,'completed':False,
        'normal_destroy':False,'normal_shutdown':False,'cache_rows':64,'capacity':640,'PP':3,'TP':8}
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    try:
        torch.set_num_threads(1);torch.cuda.set_device(rank)
        properties=torch.cuda.get_device_properties(rank)
        assert str(properties.gcnArchName).split(':')[0]=='gfx936' and properties.multi_processor_count==80
        report['before_create']=snapshot(rank)
        cfg=json.loads((pathlib.Path('/old/models/ZhipuAI-GLM-5.3')/'config.json').read_text())
        assert cfg['num_hidden_layers']==78
        buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k] if k!='tp' else payload[k][STAGE])) for k in ['world','tp','control']]
        before=time.monotonic();ctx=lib.modern_glm_model_create(rank,STAGE,64,640,json.dumps(cfg).encode(),*buffers)
        if not ctx:raise RuntimeError(lib.modern_glm_model_error().decode())
        report['create_seconds']=time.monotonic()-before;report['after_create']=snapshot(rank)
        part=json.loads(lib.modern_glm_partition_report_r3854(ctx));keys=part.pop('keys')
        assert [part['begin'],part['end']]==list(BOUNDS[STAGE:STAGE+2]) and part['PP']==3 and not part['weights_ready']
        report['partition']=part;report['parameter_key_count']=len(keys)
        report['census']=json.loads(lib.modern_glm_constructor_census_r4158(ctx))
        report['census']['scope']='Logical tensor census only; native pool rounding not included. No uninitialized values read.'
        trim=(ctypes.c_uint64*6)();check(lib,lib.modern_glm_unloaded_trim_r4052(ctx,trim,6))
        assert list(trim)[3:]==[rank,STAGE,3];report['constructor_trim']=list(trim)
        stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for name,size in [('modern_glm_model_tp_control',8),('modern_glm_model_control',24)]:
            vote=torch.ones(1,device='cuda',dtype=torch.float32)
            check(lib,getattr(lib,name)(ctx,1,ctypes.c_void_p(vote.data_ptr()),stream))
            assert vote.item()==size;del vote
        torch.cuda.empty_cache();report['after_trim_and_CCL_tiny_control']=snapshot(rank)
        reserve=6*1024**3;bank=(BOUNDS[STAGE+1]-BOUNDS[STAGE])*512*1024**2
        report['planned_resident_physical_estimate']=bank
        report['planned_resident_is_allocated']=False
        report['required_future_reserve']=reserve
        report['free_minus_estimated_future_bank']=report['after_trim_and_CCL_tiny_control']['free']-bank
        report['capacity_screen_only']=report['free_minus_estimated_future_bank']>=reserve+64*1024**2
        report['prefill_scratch_peak']='UNKNOWN; not measured by unloaded constructor'
        report['completed']=True
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        try:
            if ctx:check(lib,lib.modern_glm_model_destroy(ctx));ctx=None;report['normal_destroy']=True
            check(lib,lib.modern_glm_model_shutdown());report['normal_shutdown']=True
            report['after_shutdown']=snapshot(rank)
        except BaseException as exc:
            report.update(cleanup_error=repr(exc),cleanup_traceback=traceback.format_exc());raise
        finally:report['elapsed_seconds']=time.monotonic()-started;save()
    print('GLM_PP3_CONSTRUCTOR_NORMAL',STAGE,rank,flush=True)

if __name__=='__main__':
    path=ROOT/'control'/f'GLM-PP3-constructor-r8781-{RUN}-ids.json'
    if STAGE==0:
        lib=api()
        def unique():
            raw=ctypes.create_string_buffer(lib.modern_glm_model_id_size());check(lib,lib.modern_glm_model_id(raw))
            return base64.b64encode(raw.raw).decode()
        payload={'world':unique(),'tp':[unique() for _ in range(3)],'control':unique()}
        temporary=path.with_suffix('.tmp');assert not path.exists()
        with temporary.open('x') as out:json.dump(payload,out)
        temporary.replace(path)
    begin=time.monotonic()
    while not path.exists():
        if time.monotonic()-begin>75:raise TimeoutError('same-run constructor identities missing')
        time.sleep(.1)
    payload=json.loads(path.read_text());mp.spawn(worker,args=(payload,),nprocs=8,join=True)
    print('GLM_PP3_CONSTRUCTOR_NODE_NORMAL',STAGE,flush=True)
