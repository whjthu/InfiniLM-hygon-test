"""Exact original comm sequence, before any TextModel construction."""
import base64,ctypes,hashlib,json,os,pathlib,time,traceback
import torch
import torch.multiprocessing as mp
ROOT=pathlib.Path('/refactor');STAGE=int(os.environ['SLURM_PROCID']);RUN=os.environ['GLM_RUN_R5040']
BUILD=json.loads((ROOT/'candidates/glm-comm-only-r8784/CPU-build.json').read_text())
LIBRARY=pathlib.Path(BUILD['library']);SHA='0a5c8f885932b210ce2cad0a227a74e564219d92dbedfeb4c44f78e7357c34d8'
assert BUILD['completed'] and BUILD['library_SHA256']==SHA and hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==SHA
assert STAGE in range(3) and os.environ['GLM_PP_SIZE_R3603']=='3'
def api():
    lib=ctypes.CDLL(str(LIBRARY));P=ctypes.c_void_p;S=ctypes.c_size_t
    lib.modern_glm_model_error.restype=ctypes.c_char_p;lib.modern_glm_model_id_size.restype=S;lib.modern_glm_model_id.argtypes=[P]
    lib.modern_glm_comm_only_create_r8783.argtypes=[ctypes.c_int]*2+[S]*2+[ctypes.c_char_p]+[P]*3+[ctypes.c_char_p];lib.modern_glm_comm_only_create_r8783.restype=P
    lib.modern_glm_comm_only_destroy_r8783.argtypes=[P,ctypes.c_char_p];lib.modern_glm_model_shutdown.argtypes=[]
    for name in ['modern_glm_model_control','modern_glm_model_tp_control']:getattr(lib,name).argtypes=[P,S,P,P]
    return lib
def check(lib,code):
    if code:raise RuntimeError(lib.modern_glm_model_error().decode())
def worker(rank,payload):
    lib=api();ctx=None;torch.set_num_threads(1);torch.cuda.set_device(rank)
    path=ROOT/'results'/f'GLM-comm-only-r8783-{RUN}-stage{STAGE}-rank{rank}.json'
    journal=path.with_suffix('.jsonl');assert not path.exists() and not journal.exists()
    journal.touch(exist_ok=False)
    report={'run':RUN,'stage':STAGE,'rank':rank,'participant':STAGE*8+rank,'PID':os.getpid(),'library_SHA256':SHA,
        'scope':'world24->TP8->originalsharedcontrol; stopbeforeTextModel; tinyCCLcontrol thenoriginaldestroy/shutdown',
        'Model_constructed':False,'weights_IO':False,'finish_calls':0,'resident_calls':0,'forward_calls':0,
        'completed':False,'normal_destroy':False,'normal_shutdown':False,'driver_phases':[],'journal':str(journal)}
    def mark(phase):
        report['driver_phases'].append({'phase':phase,'driver_monotonic_ns':time.monotonic_ns()})
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
        print('GLM_COMM_DRIVER_PHASE',STAGE,rank,phase,flush=True)
    try:
        buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k] if k!='tp' else payload[k][STAGE])) for k in ['world','tp','control']]
        mark('native_comm_create.enter');ctx=lib.modern_glm_comm_only_create_r8783(rank,STAGE,64,640,b'{}',*buffers,str(journal).encode())
        if not ctx:raise RuntimeError(lib.modern_glm_model_error().decode())
        mark('native_comm_create.return.before_TextModel')
        report['actual_free_total_after_comm']=list(torch.cuda.mem_get_info(rank));mark('memory.saved')
        stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for name,count in [('modern_glm_model_tp_control',8),('modern_glm_model_control',24)]:
            mark(name+'.enter');vote=torch.ones(1,device='cuda',dtype=torch.float32)
            check(lib,getattr(lib,name)(ctx,1,ctypes.c_void_p(vote.data_ptr()),stream));assert vote.item()==count
            del vote;mark(name+'.return')
        report['completed']=True
    except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        try:
            if ctx:
                raw=ctx;ctx=None;mark('native_cleanup.enter');check(lib,lib.modern_glm_comm_only_destroy_r8783(raw,str(journal).encode()));report['normal_destroy']=True;mark('native_cleanup.return')
            mark('Runtime_shutdown.enter');check(lib,lib.modern_glm_model_shutdown());report['normal_shutdown']=True;mark('Runtime_shutdown.return')
        except BaseException as exc:report.update(cleanup_error=repr(exc),cleanup_traceback=traceback.format_exc());raise
        finally:mark('worker.finally.saved')
    print('GLM_COMM_ONLY_NORMAL_R8783',STAGE,rank,flush=True)
if __name__=='__main__':
    path=ROOT/'control'/f'GLM-comm-only-r8783-{RUN}-ids.json'
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
    print('GLM_COMM_ONLY_NODE_NORMAL_R8783',STAGE,flush=True)
