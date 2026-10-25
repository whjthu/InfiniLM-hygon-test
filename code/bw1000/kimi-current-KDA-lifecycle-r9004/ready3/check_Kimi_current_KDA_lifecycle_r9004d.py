"""Math0 projected-input component; CPU reference is diagnostic, never Model proof."""
import argparse, copy, ctypes, hashlib, json, pathlib, traceback
import numpy as np

def bf(x):
    x=np.asarray(x,dtype=np.float32).copy(); u=x.view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

class CPU:
    def __init__(self, seed=9004):
        r=np.random.default_rng(seed);self.H,self.D=6,128;C=2304
        self.x=bf(r.normal(0,.2,(280,C)));self.g=bf(r.normal(0,.2,(280,6,128)))
        self.b=bf(r.normal(0,.2,(280,6)));self.w=r.normal(0,.2,(C,4)).astype(np.float32)
        self.a=np.zeros(6,np.float32);self.bias=np.zeros((6,128),np.float32);self.calls=0;self.tokens=0
    def initial(self):
        return [np.zeros((3,2304,3),np.float32),np.zeros((3,6,128,128),np.float32)]
    def run(self,state,start,width,read,write):
        self.calls+=1;self.tokens+=width
        conv,ssm=state;hist=np.concatenate((conv[read].T,self.x[start:start+width]),axis=0)
        ys=bf(np.stack([(hist[t:t+4].T*self.w).sum(-1,dtype=np.float32) for t in range(width)]))
        conv[write]=hist[width:width+3].T
        qkv=bf(ys/(1+np.exp(-ys))).reshape(width,3,6,128);s=ssm[read].copy();out=[]
        for j in range(width):
            q,k,v=qkv[j];q=q/np.sqrt((q*q).sum(-1,keepdims=True)+np.float32(1e-6))/np.float32(np.sqrt(128))
            k=k/np.sqrt((k*k).sum(-1,keepdims=True)+np.float32(1e-6))
            decay=np.exp(-5/(1+np.exp(-np.exp(self.a)[:,None]*(self.g[start+j]+self.bias))))
            old=s*decay[:,None,:];kv=(old*k[:,None,:]).sum(-1)
            delta=(v-kv)/(1+np.exp(-self.b[start+j]))[:,None]
            hq=(old*q[:,None,:]).sum(-1);kq=(k*q).sum(-1)
            out.append(bf(hq+delta*kq[:,None]));s=old+delta[:,:,None]*k[:,None,:]
        ssm[write]=bf(s)
        return np.stack(out)
    def close(self): pass
    def counts(self):return [self.calls]*3+[self.tokens]

class Native(CPU):
    def __init__(self,path,device):
        super().__init__();import torch
        self.t=torch;torch.set_num_threads(1);torch.cuda.set_device(device);self.device=device
        self.lib=ctypes.CDLL(str(path));self.lib.kda_conv_error.restype=ctypes.c_char_p
        self.lib.kda_conv_create.argtypes=[ctypes.c_int];self.lib.kda_conv_create.restype=ctypes.c_void_p
        self.lib.kda_conv_run.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p)]
        self.lib.kda_conv_close.argtypes=[ctypes.c_void_p]
        self.lib.kda_conv_counts.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
        self.ctx=self.lib.kda_conv_create(device);assert self.ctx,self.lib.kda_conv_error()
        self.binding={}
        class Info(ctypes.Structure):
            _fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
        dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)]
        for name in ['kda_conv_create','kda_conv_run','kda_conv_counts','kda_conv_close','kda_conv_error']:
            info=Info();assert dl.dladdr(ctypes.cast(getattr(self.lib,name),ctypes.c_void_p),ctypes.byref(info))
            actual=pathlib.Path(info.file.decode());assert actual.resolve()==path.resolve()
            self.binding[name]={'file':str(actual),'SHA256':hashlib.sha256(actual.read_bytes()).hexdigest()}
    def run(self,state,start,width,read,write):
        t=self.t
        def gpu(x,dt): return t.tensor(x,device=self.device,dtype=dt)
        conv,ssm=state;F,B,I=t.float32,t.bfloat16,t.int32
        args=[gpu(self.x[start:start+width][None],B),gpu(self.w[:,None],F),gpu(conv,B),gpu(self.g[start:start+width][None],B),gpu(self.b[start:start+width][None],B),gpu(self.a,F),gpu(self.bias,F),gpu(ssm,B),gpu([0,width],I),gpu([read],I),gpu([write],I),t.empty((1,width,2304),device=self.device,dtype=B),t.empty((1,width,6,128),device=self.device,dtype=B)]
        t.cuda.synchronize();ptrs=(ctypes.c_void_p*13)(*[x.data_ptr() for x in args])
        assert self.lib.kda_conv_run(self.ctx,width,3,ptrs)==0,self.lib.kda_conv_error()
        state[0][:]=args[2].float().cpu().numpy();state[1][:]=args[7].float().cpu().numpy()
        return args[12][0].float().cpu().numpy()
    def close(self): assert self.lib.kda_conv_close(self.ctx)==0,self.lib.kda_conv_error()
    def counts(self):
        a=(ctypes.c_uint64*4)();assert self.lib.kda_conv_counts(self.ctx,a)==0
        return list(a)

def serial(backend,state,start,n,active=1,shadow=2):
    outputs=[]
    for p in range(start,start+n):
        outputs.append(backend.run(state,p,1,active,shadow)[0]);active,shadow=shadow,active
    return np.stack(outputs),active,shadow

def distance(a,b):
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return {'max_abs':float(np.max(np.abs(a-b))),'different_elements':int(np.count_nonzero(a!=b))}

def exercise(backend):
    rows=[];stale_detected=0
    repeat_a=backend.initial();repeat_b=backend.initial()
    ya=backend.run(repeat_a,0,8,1,2);yb=backend.run(repeat_b,0,8,1,2)
    assert np.array_equal(ya,yb) and all(np.array_equal(a,b) for a,b in zip(repeat_a,repeat_b)), 'same-path ownrepeat changed'
    for past in [0,3,252,255,256]:
        base=backend.initial();active,shadow=1,2
        # Build the complete recurrent prefix, including 255 -> 256 histories.
        # The KDA/conv component has no Model KV page table.
        if past:_,active,shadow=serial(backend,base,0,past)
        gold=copy.deepcopy(base);goldout,ga,gs=serial(backend,gold,past,8,active,shadow)
        verified=copy.deepcopy(base);chunk=backend.run(verified,past,8,active,shadow)
        assert all(np.array_equal(a[active],b[active]) for a,b in zip(verified,base)), 'verify overwrote active'
        for accepted in range(8):
            retained=accepted+1
            for terminal in [False,True]:
                candidate=copy.deepcopy(verified);before=copy.deepcopy(candidate)
                call_before=backend.counts()
                if terminal:
                    # EOS/cap leaves no consumer; discarded suffix is never committed.
                    assert all(np.array_equal(a,b) for a,b in zip(candidate,before))
                    assert backend.counts()==call_before, 'terminal must not replay'
                    rows.append({'past':past,'accepted':accepted,'EOS_or_limit':True,'continuation':False,'replay_calls':0});continue
                if retained<8:backend.run(candidate,past,retained,active,shadow)
                call_after=backend.counts();assert call_after[0]-call_before[0]==int(retained<8)
                assert all(np.array_equal(a[active],b[active]) for a,b in zip(candidate,base)), 'replay overwrote active'
                reference=copy.deepcopy(base);_,ra,rs=serial(backend,reference,past,retained,active,shadow)
                assert np.array_equal(candidate[0][shadow],reference[0][ra]), 'conv prefix commit failed'
                diag=distance(candidate[1][shadow],reference[1][ra])
                # Correction remains unconsumed until this next one-step call.
                cy=backend.run(candidate,past+retained,1,shadow,active)
                ry=backend.run(reference,past+retained,1,ra,rs)
                assert all(np.array_equal(a[0],b[0]) for a,b in zip(candidate,base)), 'unused state slot0 changed'
                if retained<8:
                    wrong=copy.deepcopy(verified);backend.run(wrong,past+retained,1,shadow,active)
                    assert not np.array_equal(wrong[0][active],candidate[0][active]) or not np.array_equal(wrong[1][active],candidate[1][active]), 'fixture did not distinguish missing replay'
                    stale_detected+=1
                rows.append({'past':past,'accepted':accepted,'EOS_or_limit':False,'continuation':True,'replay_calls':int(retained<8),'native_or_reference_replay_counter_delta':[b-a for a,b in zip(call_before,call_after)],'KDA_state_diagnostic':diag,'continuation_output_diagnostic':distance(cy,ry),'continuation_state_diagnostic':distance(candidate[1][active],reference[1][rs]),'verify_allposition_diagnostic':distance(chunk,goldout)})
    return {'status':'COMPONENT_LIFECYCLE_EXECUTED_NOT_MODEL_QUALIFIED','backend':type(backend).__name__,'cases':rows,'ownrepeat_same_path_exact_diagnostic':True,'unused_pool_slot0_canary_untouched':True,'missing_replay_negative_controls_detected':stale_detected,'Model_final_IDs_EOS':'NOT_TESTED','Model_TP16PP2':'NOT_TESTED','pagecrossing':'full recurrent prefixes0/3/252/255/256, no KV pages in KDA/conv component','dtype':'BF16 operands/state + F32 weights/gate bias; CPU FP32 reductions are diagnostic, no GPU F64','metric':'no speed or acceptance claim'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--gpu-lib',type=pathlib.Path);p.add_argument('--device',type=int,default=0);p.add_argument('--root-go',action='store_true');a=p.parse_args()
    assert not a.out.exists()
    if a.gpu_lib and not a.root_go:p.error('native GPU requires explicit root GO; do not submit from this script')
    backend=None;failure=None;result={'status':'FAILED','source_SHA256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'native_or_reference_closed_normally':False}
    try:
        backend=Native(a.gpu_lib,a.device) if a.gpu_lib else CPU()
        result=exercise(backend);result['operation_counts_conv_silu_kda_tokens']=backend.counts();result['source_SHA256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        assert result['operation_counts_conv_silu_kda_tokens']==[1143,1143,1143,1297]
        if a.gpu_lib:
            result['native_SHA256']=hashlib.sha256(a.gpu_lib.read_bytes()).hexdigest();result['actual_symbol_bindings']=backend.binding
            maps=pathlib.Path('/proc/self/maps').read_text();libs=sorted({line.split()[-1] for line in maps.splitlines() if any(n in line for n in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so'])})
            result['actual_runtime_dependencies_SHA256']={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in libs}
    except BaseException as e:
        failure=e;result.update(status='FAILED',exception=repr(e),traceback=traceback.format_exc())
        if backend:
            try:result['operation_counts_conv_silu_kda_tokens']=backend.counts()
            except BaseException as counter_error:result['counter_error']=repr(counter_error)
    finally:
        if backend:
            try:backend.close();result['native_or_reference_closed_normally']=True
            except BaseException as close_error:
                result['close_error']=repr(close_error)
                if failure is None:failure=close_error
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if failure is not None:raise failure
