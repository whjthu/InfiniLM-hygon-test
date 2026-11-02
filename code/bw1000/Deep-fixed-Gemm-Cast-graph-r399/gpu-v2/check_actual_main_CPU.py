"""Run the actual main/Bound/session with recording CAPI+dladdr and real temp files.
No native library, GPU, Runtime or model is loaded. CPU numpy supplies the recording leaf.
"""
from pathlib import Path
import ast,contextlib,ctypes,hashlib,importlib.util,io,json,os,sys,tempfile,threading,time,types
import numpy as np
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
class Fn:
    serial=0
    def __init__(self,func,path):self.func=func;self.path=path;Fn.serial+=1;self.address=65536+16*Fn.serial
    def __call__(self,*args):return self.func(*args)
class Leaf:
    def __init__(self,private,vendor,fail_capture):
        self.private=private;self.vendor=vendor;self.fail_capture=fail_capture;self.owner=threading.get_ident();self.err=b'recording error';self.closed=0;self.constructed=0
        self.c={'prepares':0,'captures':0,'eager_calls':0,'replay_calls':0,'phase':2,'workspace_bytes':8*1024*1024+512}
        names=['create','inputs','prepare','capture','run','check','read','report','destroy','error'];self.functions={'deep_fixed_graph_'+n+'_r399':Fn(getattr(self,n),private)for n in names}
        for name in ['cublasGemmStridedBatchedEx','cublasSetWorkspace_v2','cublasSetPointerMode_v2']:self.functions[name]=Fn(lambda*args:0,vendor)
    def __getattr__(self,name):return self.functions[name]
    def create(self,opt,device):
        if not opt:return None
        assert opt==1 and device==0;self.constructed+=1;return 42
    def inputs(self,ctx,a,an,b,bn):
        if threading.get_ident()!=self.owner:self.err=b'recording wrong owner';return 1
        assert ctx==42 and an==16*2048 and bn==2048*512
        self.a=np.ctypeslib.as_array(a,shape=(an,)).reshape(16,2048).copy();self.b=np.ctypeslib.as_array(b,shape=(bn,)).reshape(2048,512).copy();self.y=np.full((16,512),np.nan,dtype=np.float32);self.z=np.full((16,512),0x7fc0,dtype=np.uint16);return 0
    def calculate(self):
        self.y=self.a@self.b;u=self.y.view(np.uint32);self.z=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
    def prepare(self,ctx):self.c['prepares']+=1;self.calculate();return 0
    def capture(self,ctx):
        if self.fail_capture:self.err=b'recording original capture failure';return 1
        self.c['captures']+=1;return 0
    def run(self,ctx,mode,loops,ms):self.c['replay_calls'if mode else'eager_calls']+=loops;self.calculate();ms._obj.value=.01;return 0
    def check(self,ctx):return 0
    def read(self,ctx,y,z,count):
        assert count==16*512;np.ctypeslib.as_array(y,shape=(count,))[:]=self.y.ravel();np.ctypeslib.as_array(z,shape=(count,))[:]=self.z.ravel();return 0
    def report(self,ctx):return json.dumps(self.c).encode()
    def destroy(self,ctx):assert ctx==42;self.closed+=1;return 0
    def error(self):return self.err
def case(name,fail_capture=False,fail_writes=False):
    source=(D/'GPU_driver.py').read_bytes();tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');assigned={n.id for n in ast.walk(main)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Store)};assert'report_path'in assigned and'vendor_path'in assigned and'path'not in assigned
    spec=importlib.util.spec_from_file_location('r399_actualmain_'+name,D/'GPU_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory()as tmp:
        root=Path(tmp);package=root/'package';package.mkdir();runtime=root/'refactor';(runtime/'results').mkdir(parents=True);vendor=root/'readonly-vendor';vendor.mkdir();v=vendor/'libcublas.so.12.6.3';v.write_bytes(b'recording vendor original bytes');library=vendor/'private.so';library.write_bytes(b'recording private original bytes');originals={p:sha(p.read_bytes())for p in [v,library]}
        (package/'GPU_driver.py').write_bytes(source);cp=package/'CPU-fixture.json';cp.write_text(json.dumps({'passed':True,'completed':True,'actual_CPU':'910407.797','GPU_initialized':False,'Native_dlopen':False,'library':str(library),'library_SHA256':originals[library]}));mf=package/'GPU_SOURCE_MANIFEST.json';mf.write_text(json.dumps({'GPU_GO':True,'files_SHA256':{'GPU_driver.py':sha(source)},'CPU_receipt':str(cp),'CPU_receipt_SHA256':sha(cp.read_bytes()),'library_SHA256':originals[library],'cublas_SHA256':originals[v],'required_runtime_dependencies_SHA256':{}}))
        leaf=Leaf(str(library),str(v),fail_capture);addresses={x.address:x for x in leaf.functions.values()}
        def dladdr(address,info):info._obj.path=addresses[address.value].path.encode();return 1
        def CDLL(path,*args,**kwargs):return types.SimpleNamespace(dladdr=Fn(dladdr,str(v)))if path is None else leaf
        old_cast=ctypes.cast
        def cast(value,kind):return ctypes.c_void_p(value.address)if isinstance(value,Fn)else old_cast(value,kind)
        real_path=m.Path
        m.Path=lambda value:runtime if str(value)=='/refactor'else real_path(value)
        m.__file__=str(package/'GPU_driver.py');old_argv=sys.argv;old_torch=sys.modules.get('torch');old_cdll=ctypes.CDLL;old_write=Path.write_text;writes=[];snapshots=[];stderr=io.StringIO();raised=None
        def write(path,data,*args,**kwargs):
            writes.append(str(path));snapshots.append(json.loads(data));assert path.parent==runtime/'results','Vendor path must never receive a write attempt'
            if fail_writes and len(writes)>=2:raise OSError('recording observe write failure'if len(writes)==2 else'recording final save failure')
            return old_write(path,data,*args,**kwargs)
        old_env={n:os.environ.get(n)for n in ['SLURM_JOB_ID','SLURM_STEP_ID']};os.environ.update(SLURM_JOB_ID='910374',SLURM_STEP_ID='999999')
        fake_torch=types.SimpleNamespace(set_num_threads=lambda n:None,float32='recording_float32',ones=lambda n,**kw:np.ones(n,dtype=np.float32),cuda=types.SimpleNamespace(device_count=lambda:1,set_device=lambda n:None,get_device_name=lambda n:'recording Hygon',synchronize=lambda:None,empty_cache=lambda:None))
        try:
            sys.argv=[m.__file__,'--manifest-sha',sha(mf.read_bytes()),'--deadline-unix',str(time.time()+300)];sys.modules['torch']=fake_torch;ctypes.CDLL=CDLL;ctypes.cast=cast;Path.write_text=write
            with contextlib.redirect_stderr(stderr):
                try:m.main()
                except BaseException as error:raised=error
        finally:
            sys.argv=old_argv;ctypes.CDLL=old_cdll;ctypes.cast=old_cast;Path.write_text=old_write
            if old_torch is None:sys.modules.pop('torch',None)
            else:sys.modules['torch']=old_torch
            for n,value in old_env.items():
                if value is None:os.environ.pop(n,None)
                else:os.environ[n]=value
        report=runtime/'results/Deep399-910374.999999.json';assert report.exists()and leaf.closed==1 and leaf.constructed==1 and all(sha(p.read_bytes())==h for p,h in originals.items())and{p.name for p in vendor.iterdir()}=={'private.so','libcublas.so.12.6.3'}
        final=json.loads(report.read_bytes())
        if not fail_capture and not fail_writes:
            assert raised is None and final['completed']and final['passed']and len(final['session']['patterns'])==3 and final['session']['normal_runtime_close']and len(writes)==7
            assert leaf.c['eager_calls']==99 and leaf.c['replay_calls']==100
        elif fail_capture:
            assert isinstance(raised,RuntimeError)and'recording original capture failure'in str(raised)
            if not fail_writes:assert final['completed']and not final['passed']and'recording original capture failure'in final['error']
            else:assert'session_observe'in stderr.getvalue()and'main_final_report'in stderr.getvalue()
        else:assert isinstance(raised,OSError)and str(raised)=='recording observe write failure'and'main_final_report'in stderr.getvalue()
        return dict(case=name,passed=True,actual_main=True,actual_Bound_and_session=True,write_attempts=len(writes),all_writes_designated_report_path=True,vendor_write_attempts=0,vendor_original_bytes_unchanged=True,primary_error=None if raised is None else repr(raised),recording_CAPI_dladdr_Torch=True,Native_dlopen=False,GPU=False,Runtime=False,three_pattern_observe_and_final_verified=not fail_capture and not fail_writes,close_attempts=leaf.closed,error_events=[json.loads(line)for line in stderr.getvalue().splitlines()if line.strip()])
def main():
    cases=[case('three_patterns_observe_final'),case('primary_capture_failure_final',True,False),case('primary_capture_failure_and_save_failure',True,True),case('observe_failure_not_masked_by_final_save_failure',False,True)]
    print(json.dumps(dict(completed=True,passed=True,cases=cases,scope='Actualmain/Bound/controlglue and actualtemporaryfilesystem, recordingCAPI/dladdr/Torch only. No compiledNative/GPU qualification.')))
if __name__=='__main__':main()
