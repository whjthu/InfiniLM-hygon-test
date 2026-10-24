"""Exact original comm sequence, before any TextModel construction."""
import base64,ctypes,hashlib,json,os,pathlib,time,traceback
import torch
import torch.multiprocessing as mp
ROOT=pathlib.Path('/refactor');STAGE=int(os.environ['SLURM_PROCID']);RUN=os.environ['GLM_RUN_R5040']
BUILD=json.loads((ROOT/'candidates/glm-constructor-only-r8791/CPU-build.json').read_text())
LIBRARY=pathlib.Path(BUILD['library']);SHA='59d65034a92bf4065d3d1b979835935e54d059f61742e73f5127a11712ed8ca1'
assert BUILD['completed'] and BUILD['library_SHA256']==SHA and hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==SHA
assert STAGE in range(3) and os.environ['GLM_PP_SIZE_R3603']=='3'
def api():
    lib=ctypes.CDLL(str(LIBRARY));P=ctypes.c_void_p;S=ctypes.c_size_t
    lib.modern_glm_model_error.restype=ctypes.c_char_p;lib.modern_glm_model_id_size.restype=S;lib.modern_glm_model_id.argtypes=[P]
    lib.modern_glm_constructor_only_create_r8790.argtypes=[ctypes.c_int]*2+[S]*2+[ctypes.c_char_p]+[P]*3+[ctypes.c_char_p];lib.modern_glm_constructor_only_create_r8790.restype=P
    lib.modern_glm_constructor_only_destroy_r8790.argtypes=[P,ctypes.c_char_p];lib.modern_glm_model_shutdown.argtypes=[]
    for name in ['modern_glm_partition_report_r3854','modern_glm_constructor_census_r4158']:
        getattr(lib,name).argtypes=[P];getattr(lib,name).restype=ctypes.c_char_p
    lib.modern_glm_unloaded_trim_r4052.argtypes=[P,ctypes.POINTER(ctypes.c_uint64),S]
    for name in ['modern_glm_model_control','modern_glm_model_tp_control']:getattr(lib,name).argtypes=[P,S,P,P]
    return lib
def check(lib,code):
    if code:raise RuntimeError(lib.modern_glm_model_error().decode())
def worker(rank,payload):
    lib=api();ctx=None;torch.set_num_threads(1);torch.cuda.set_device(rank)
    path=ROOT/'results'/f'GLM-constructor-only-r8790-{RUN}-stage{STAGE}-rank{rank}.json'
    journal=path.with_suffix('.jsonl');assert not path.exists() and not journal.exists()
    journal.touch(exist_ok=False)
    report={'run':RUN,'stage':STAGE,'rank':rank,'participant':STAGE*8+rank,'PID':os.getpid(),'library_SHA256':SHA,
        'scope':'53conditional unloadedTextModelconstructor only; samecommsequence andoriginalsync; actualcensus/free andplannedresidentestimate; nofinish/weights/resident/forward',
        'Model_constructed':False,'weights_IO':False,'finish_calls':0,'resident_calls':0,'forward_calls':0,
        'completed':False,'normal_destroy':False,'normal_shutdown':False,'driver_phases':[],'journal':str(journal)}
    def mark(phase):
        report['driver_phases'].append({'phase':phase,'driver_monotonic_ns':time.monotonic_ns()})
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
        print('GLM_COMM_DRIVER_PHASE',STAGE,rank,phase,flush=True)
    try:
        buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k] if k!='tp' else payload[k][STAGE])) for k in ['world','tp','control']]
        cfg=json.loads((pathlib.Path('/old/models/ZhipuAI-GLM-5.3')/'config.json').read_text());assert cfg['num_hidden_layers']==78
        mark('native_constructor.enter');ctx=lib.modern_glm_constructor_only_create_r8790(rank,STAGE,64,640,json.dumps(cfg).encode(),*buffers,str(journal).encode())
        if not ctx:raise RuntimeError(lib.modern_glm_model_error().decode())
        report['Model_constructed']=True;mark('native_constructor.return.unloaded')
        report['actual_free_total_after_constructor']=list(torch.cuda.mem_get_info(rank));mark('constructor.memory.saved')
        part=json.loads(lib.modern_glm_partition_report_r3854(ctx));keys=part.pop('keys');assert [part['begin'],part['end']]==[(0,26,54)[STAGE],(26,54,78)[STAGE]] and part['PP']==3 and not part['weights_ready']
        report['partition']=part;report['parameter_key_count']=len(keys);report['census']=json.loads(lib.modern_glm_constructor_census_r4158(ctx));mark('logical.census.saved')
        trim=(ctypes.c_uint64*6)();check(lib,lib.modern_glm_unloaded_trim_r4052(ctx,trim,6));assert list(trim)[3:]==[rank,STAGE,3];report['unloaded_trim']=list(trim);mark('constructor.trim.saved')
        stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for name,count in [('modern_glm_model_tp_control',8),('modern_glm_model_control',24)]:
            mark(name+'.enter');vote=torch.ones(1,device='cuda',dtype=torch.float32)
            check(lib,getattr(lib,name)(ctx,1,ctypes.c_void_p(vote.data_ptr()),stream));assert vote.item()==count
            del vote;mark(name+'.return')
        torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info(rank);bank=(26,28,24)[STAGE]*512*1024**2
        report['after_trim_and_tiny_CCL']={'free':free,'total':total,'device_total_minus_free':total-free,'Torch_allocated':torch.cuda.memory_allocated(rank),'Torch_reserved':torch.cuda.memory_reserved(rank),'native_allocator_allocated_reserved':'UNKNOWN_no_exported_counter','is_peak':False}
        report['planned_resident_estimate']=bank;report['resident_actually_allocated']=False;report['free_minus_planned_resident']=free-bank;report['reserve']=6*1024**3;report['estimated_resident_plus_reserve_screen']=free-bank>=6*1024**3+64*1024**2;report['prefill_scratch_peak']='UNKNOWN_notrun';report['completed']=True;mark('capacity.snapshot.saved')
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        try:
            if ctx:
                raw=ctx;ctx=None;mark('native_cleanup.enter');check(lib,lib.modern_glm_constructor_only_destroy_r8790(raw,str(journal).encode()));report['normal_destroy']=True;mark('native_cleanup.return')
            mark('Runtime_shutdown.enter');check(lib,lib.modern_glm_model_shutdown());report['normal_shutdown']=True;mark('Runtime_shutdown.return')
        except BaseException as exc:report.update(cleanup_error=repr(exc),cleanup_traceback=traceback.format_exc());raise
        finally:mark('worker.finally.saved')
    print('GLM_CONSTRUCTOR_ONLY_NORMAL_R8790',STAGE,rank,flush=True)
if __name__=='__main__':
    path=ROOT/'control'/f'GLM-constructor-only-r8790-{RUN}-ids.json'
    if STAGE==0:
        lib=api()
        def unique():
            b=ctypes.create_string_buffer(lib.modern_glm_model_id_size());check(lib,lib.modern_glm_model_id(b));return base64.b64encode(b.raw).decode()
        data={'world':unique(),'tp':[unique() for _ in range(3)],'control':unique()};tmp=path.with_suffix('.tmp');assert not path.exists()
        with tmp.open('x') as f:json.dump(data,f)
        tmp.replace(path)
    begin=time.monotonic()
    while not path.exists():
        if time.monotonic()-begin>75:raise TimeoutError('same-run comm IDs missing')
        time.sleep(.1)
    mp.spawn(worker,args=(json.loads(path.read_text()),),nprocs=8,join=True)
    print('GLM_CONSTRUCTOR_ONLY_NODE_NORMAL_R8790',STAGE,flush=True)
