"""One fixed-buffer private Runtime graph function qualification, no Model/TP/weights.
Import and execute_session are CPU-safe; only main loads the qualified private DSO.
"""
from pathlib import Path
import argparse,ctypes,datetime,hashlib,json,os,threading,time,traceback
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute_session(b,patterns,validate,observe,deadline):
    """Actual glue used by native bindings and recording CPU leaf; close exactly once."""
    out={'patterns':[],'normal_runtime_close':False,'prepare_ms':None,'capture_ms':None}
    primary=None
    try:
        b.inputs(*patterns[0]);t=time.monotonic();b.prepare();out['prepare_ms']=(time.monotonic()-t)*1000
        t=time.monotonic();b.capture();out['capture_ms']=(time.monotonic()-t)*1000
        for p,values in enumerate(patterns):
            case={'pattern':p,'warm':[],'formal':[]}
            for warm,modes,loops in [(True,[0,1],1),(False,[0,1,1,0],16)]:
                for mode in modes:
                    if time.time()>=deadline:raise TimeoutError('Original r399 GPU deadline, no retry')
                    b.inputs(*values) # Owns host input storage, poisons Y/Z, outside timing.
                    tick=time.monotonic();internal_ms=b.run(mode,loops);ms=(time.monotonic()-tick)*1000/loops
                    b.check();numeric=validate(p,b.read())
                    assert numeric['passed'],'Numeric failure; close original Runtime without reset/replay'
                    case['warm'if warm else'formal'].append(dict(mode=mode,loops=loops,whole_completed_wall_ms_per_iteration=ms,internal_loop_and_terminal_drain_ms_per_iteration=internal_ms,numeric=numeric))
            out['patterns'].append(case);observe(dict(out))
        before=b.report();errors=[]
        def foreign():
            try:b.inputs(*patterns[-1])
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=foreign);thread.start();thread.join()
        assert len(errors)==1 and b.report()==before,'Foreign owner refusal must preserve owner state'
        b.inputs(*patterns[-1]);b.run(1,1);b.check();assert validate(len(patterns)-1,b.read())['passed']
        final=b.report();assert final['prepares']==1 and final['captures']==1 and final['eager_calls']==99 and final['replay_calls']==100 and final['phase']==2
        out.update(counters=final,foreign_thread_refused_without_owner_poison=True,actual_capture_CAPI_requires_2operators_1device_segment_0host_segments_empty_fallback=True)
        return out
    except BaseException as error:
        primary=error;out['session_error']=repr(error);raise
    finally:
        try:b.close();out['normal_runtime_close']=True
        except BaseException as error:
            out['cleanup_error']=repr(error)
            if primary is None:raise
        finally:observe(dict(out))
class Bound:
    def __init__(self,lib):
        self.lib=lib;self.ctx=None
        v=ctypes.c_void_p;i=ctypes.c_int;n=ctypes.c_size_t;f=ctypes.POINTER(ctypes.c_float);u=ctypes.POINTER(ctypes.c_uint16)
        specs={'create':([i,i],v),'inputs':([v,f,n,f,n],i),'prepare':([v],i),'capture':([v],i),'run':([v,i,i,ctypes.POINTER(ctypes.c_double)],i),'check':([v],i),'read':([v,f,u,n],i),'report':([v],ctypes.c_char_p),'destroy':([v],i),'error':([],ctypes.c_char_p)}
        self.f={}
        for name,(args,result)in specs.items():fn=getattr(lib,'deep_fixed_graph_'+name+'_r399');fn.argtypes=args;fn.restype=result;self.f[name]=fn
        assert not self.f['create'](0,0),'Default-OFF refusal before Runtime construction'
        self.ctx=self.f['create'](1,0)
        if not self.ctx:raise RuntimeError(self.error())
    def error(self):return(self.f['error']()or b'unknown').decode(errors='replace')
    def call(self,name,*args):
        if self.f[name](self.ctx,*args):raise RuntimeError(self.error())
    def inputs(self,a,b):self.call('inputs',a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),a.size,b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),b.size)
    def prepare(self):self.call('prepare')
    def capture(self):self.call('capture')
    def check(self):self.call('check')
    def run(self,mode,loops):
        ms=ctypes.c_double();self.call('run',mode,loops,ctypes.byref(ms));return ms.value
    def read(self):
        import numpy as np
        y=np.empty((16,512),dtype=np.float32);z=np.empty((16,512),dtype=np.uint16);self.call('read',y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),z.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),y.size);return y,z
    def report(self):
        text=self.f['report'](self.ctx)
        if not text:raise RuntimeError(self.error())
        return json.loads(text)
    def close(self):
        if self.ctx:
            context=self.ctx;self.ctx=None # Never retry a partially failed destruction.
            if self.f['destroy'](context):raise RuntimeError(self.error())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest-sha',required=True);parser.add_argument('--deadline-unix',required=True,type=float);args=parser.parse_args();D=Path(__file__).resolve().parent;R=Path('/refactor')
    mf=D/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==args.manifest_sha;m=json.loads(mf.read_bytes());assert m['GPU_GO']is True and all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    receipt=Path(m['CPU_receipt']);assert sha(receipt)==m['CPU_receipt_SHA256'];c=json.loads(receipt.read_bytes());assert c['passed']and c['completed']and c['actual_CPU']=='910407.797'and not c['GPU_initialized']and not c['Native_dlopen'];assert sha(c['library'])==m['library_SHA256']==c['library_SHA256']
    for p,h in m['required_runtime_dependencies_SHA256'].items():assert sha(p)==h,p
    if time.time()>=args.deadline_unix:raise TimeoutError('Original launch deadline expired before imports')
    actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=R/'results'/('Deep399-'+actual+'.json');assert not path.exists();q=dict(completed=False,passed=False,actual_GPU=actual,PID=os.getpid(),scope=__doc__,source_manifest_SHA256=args.manifest_sha,CPU_receipt_SHA256=m['CPU_receipt_SHA256'],library_SHA256=m['library_SHA256'],Model_loaded=False,profile=0,GPU_count=1)
    def save(extra=None):
        if extra:q['session']=extra
        q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(path)
    save();bound=None
    try:
        tick=time.monotonic()
        import numpy as np
        import torch
        torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);q['device']=dict(name=torch.cuda.get_device_name(0),visible_count=1,current=0)
        q['Torch_device_setup_ms']=(time.monotonic()-tick)*1000
        tick=time.monotonic();lib=ctypes.CDLL(c['library']);q['DSO_load_ms']=(time.monotonic()-tick)*1000
        class DlInfo(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
        dladdr=ctypes.CDLL(None).dladdr;dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dladdr.restype=ctypes.c_int;q['actual_provider_symbols']={}
        for symbol in ['deep_fixed_graph_create_r399','cublasGemmStridedBatchedEx','cublasSetWorkspace_v2','cublasSetPointerMode_v2']:
            info=DlInfo();assert dladdr(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info));path=Path(info.path.decode()).resolve();h=sha(path)
            expected=m['library_SHA256']if symbol.startswith('deep_fixed_')else m['cublas_SHA256'];assert h==expected,(symbol,str(path),h)
            q['actual_provider_symbols'][symbol]=dict(path=str(path),SHA256=h)
        rng=np.random.default_rng(399);a=np.ascontiguousarray(rng.normal(0,.03125,(16,2048)),dtype=np.float32);b=np.ascontiguousarray(rng.normal(0,.03125,(2048,512)),dtype=np.float32)
        patterns=[(a,b),(np.ascontiguousarray(-a*.5),np.ascontiguousarray(b*.25)),(np.zeros_like(a),b)]
        # This is CPU float64 oracle only. No GPU float64 tensors or operations.
        oracle=[x.astype(np.float64)@y.astype(np.float64)for x,y in patterns];q['input_SHA256']=[dict(A=hashlib.sha256(x.tobytes()).hexdigest(),B=hashlib.sha256(y.tobytes()).hexdigest())for x,y in patterns]
        def validate(index,values):
            y,z=values;zf=(z.astype(np.uint32)<<16).view(np.float32);ref=oracle[index];refnorm=float(np.linalg.norm(ref));peak=max(float(np.max(np.abs(ref))),1e-30);out={}
            for key,v in [('F32',y),('BF16',zf)]:
                finite=bool(np.isfinite(v).all());relative=float(np.linalg.norm(v-ref)/max(refnorm,1e-30));error=float(np.max(np.abs(v-ref)))/peak
                out[key]=dict(finite=finite,relative_RMS=relative,maxabs_over_reference_max=error);out[key]['passed']=finite and relative<=1/64 and error<=1/8
            out['passed']=all(out[n]['passed']for n in ['F32','BF16']);return out
        tick=time.monotonic();bound=Bound(lib);q['Context_init_ms']=(time.monotonic()-tick)*1000
        q['session']=execute_session(bound,patterns,validate,save,args.deadline_unix);bound=None
        t=torch.ones(32,device='cuda',dtype=torch.float32);assert(t+1).sum().item()==64;torch.cuda.synchronize();del t;torch.cuda.empty_cache();q.update(completed=True,passed=True,normal_runtime_close=True,Torch_after_Runtime_close=True,qualification='Actual single Runtime/provider graph capability only. Samepreparedhandle/workspaceeager vs graph. Privateeager setsWorkspace eachcall; productionGeneric onlySetStream/sharedhandle, so no directproductionSDK/attentiongain. Syntheticfixed2D input;ModelbindingandamortizationUNKNOWN.',Model_gain=False);save()
    except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if bound is not None:
            try:bound.close();q['normal_runtime_close']=True
            except BaseException as e:q['cleanup_error']=repr(e)
        q['completed']=True;save()
if __name__=='__main__':main()
