"""Future single-owner real routed-operator adjudication, NOT a launcher.

Existing914 correctednative only. Actualr4674 M8 duplicatedtoT16 is explicit;
no random latent/routes, no Model/CCL, no FP64. No execution without root GO.
"""
import pathlib,json,hashlib,time,ctypes,contextlib
import numpy as np
E,T,H,I,N,R=896,16,3584,192,384,256
NATIVE_SHA='a08a228b6367a5046758be0fa242d075de5e23f96b6b403417ef5fa952c08d7d'
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
def fixture(refs):
    values={}
    for name,z in refs.items():
        p=pathlib.Path(z['file']);raw=p.read_bytes();assert len(raw)==z['bytes']and hashlib.sha256(raw).hexdigest()==z['SHA256']
        values[name]=np.frombuffer(raw,{'BF16':np.uint16,'I32':np.int32,'F32':np.float32}[z['dtype']]).reshape(z['shape'])
    assert values['routed_input'].shape==(8,H)
    x,ids,rw=[np.tile(values[k],(2,1)).copy()for k in ['routed_input','routing_ids','routing_weights']]
    assert x.shape==(T,H)and ids.shape==rw.shape==(T,16)
    assert ((ids>=0)&(ids<E)).all()and all(len(np.unique(row))==16 for row in ids)
    assert len(np.unique(ids))==21 and np.isfinite(fp(x)).all()and np.isfinite(rw).all()
    return x,ids,rw

def decode(packed,scale):
    """Independent logical-coordinate E2M1/E8M0 CPU FP32, not integerINT4.

    255 (nonfinite E8M0) is rejected rather than inventing a replacement.
    Endpoint0 uses exponent-127 (same ldexp endpoint in existing decoder).
    """
    assert packed.dtype==scale.dtype==np.uint8 and not np.any(scale==255)
    codes=np.empty((packed.shape[0],packed.shape[1]*2),np.uint8)
    codes[:,::2]=packed&15;codes[:,1::2]=packed>>4
    table=np.asarray([0,.5,1,1.5,2,3,4,6],np.float32)
    values=table[codes&7]*np.where(codes&8,np.float32(-1),np.float32(1))
    return np.ldexp(values,np.repeat(scale.astype(np.int32)-127,32,axis=1)).astype(np.float32)

def errors(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32);delta=a-b
    return dict(finite=bool(np.isfinite(a).all()and np.isfinite(b).all()),changed=int(np.count_nonzero(a!=b)),maxabs=float(np.abs(delta).max()),RMSE=float(np.sqrt(np.mean(delta*delta,dtype=np.float32))),relativeRMS=float(np.sqrt(np.mean(delta*delta,dtype=np.float32))/max(1e-30,float(np.sqrt(np.mean(b*b,dtype=np.float32))))))

def run(manifest,output):
    import torch
    from safetensors import safe_open
    from kimi_wfp4_repack_r8669zu import _permute
    torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    m=json.loads(pathlib.Path(manifest).read_text());assert digest(m['native'])==NATIVE_SHA
    xbits,ids_host,rw_host=fixture(m['fixture_refs']);active=np.unique(ids_host);flat=ids_host.reshape(-1)
    root=pathlib.Path('/shared/Kimi-K3');mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.1.block_sparse_moe.experts.'
    # Full E896 tensor extents; only actual activeexpert pages populated/read.
    # No weight outside actualrouteIDs is accessed. Keep allcounts/routes E896.
    shapes={'w13':(E,N,H//2),'s13':(E,N,H//32),'w2':(E,H,I//2),'s2':(E,H,I//32)}
    weights={k:torch.empty(sh,device='cuda',dtype=torch.uint8)for k,sh in shapes.items()};repacked=torch.empty_like(weights['w13'])
    oracle=np.empty((R,N),np.float32);oracle_y=np.zeros((T,H),np.float32);host={};hashes={}
    with contextlib.ExitStack()as stack:
        files={}
        def get(e,name,axis,width):
            key=prefix+str(int(e))+'.'+name;fn=mapping[key]
            if fn not in files:files[fn]=stack.enter_context(safe_open(root/fn,framework='pt',device='cpu'))
            view=files[fn].get_slice(key);sl=[slice(None)]*len(view.get_shape());sl[axis]=slice(0,width) # TP16globalrank0 exactslice
            v=view[tuple(sl)].contiguous();assert v.dtype==torch.uint8
            hashes[key]=hashlib.sha256(v.numpy().tobytes()).hexdigest();return v.numpy().copy()
        for e in active:
            w13=np.concatenate([get(e,'w1.weight_packed',0,I),get(e,'w3.weight_packed',0,I)])
            s13=np.concatenate([get(e,'w1.weight_scale',0,I),get(e,'w3.weight_scale',0,I)])
            w2=get(e,'w2.weight_packed',1,I//2);s2=get(e,'w2.weight_scale',1,I//32)
            assert w13.shape==(N,H//2)and s13.shape==(N,H//32)and w2.shape==(H,I//2)and s2.shape==(H,I//32)
            for k,v in [('w13',w13),('s13',s13),('w2',w2),('s2',s2)]:weights[k][int(e)].copy_(torch.from_numpy(v))
            candidate=w13.copy();_permute(candidate);repacked[int(e)].copy_(torch.from_numpy(candidate))
            route=np.flatnonzero(flat==e);oracle[route]=fp(xbits)[route//16]@decode(w13,s13).T
            host[int(e)]=(w2,s2)
    # CPU fullroutedoutput reference: FP32matmul + originalSITU + BF16act + FP32W2.
    gate,up=oracle[:,:I],oracle[:,I:]
    activation_ref=(np.float32(4)*np.tanh(gate/np.float32(4))/(1+np.exp(-gate)))*(np.float32(25)*np.tanh(up/np.float32(25)))
    activation_ref=torch.from_numpy(activation_ref).to(torch.bfloat16).float().numpy()
    for e in active:
        route=np.flatnonzero(flat==e);w2,s2=host[int(e)];partial=activation_ref[route]@decode(w2,s2).T
        for j,rid in enumerate(route):oracle_y[rid//16]+=np.float32(rw_host.reshape(-1)[rid])*partial[j]
    host.clear()
    lib=ctypes.CDLL(m['native']);base=lib.modern_kimi_grouped_I192_r8669y;base.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*14+[ctypes.c_size_t]*2+[ctypes.c_void_p]
    new=lib.modern_kimi_official_WFP4_r8669zu;new.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*13
    x=torch.from_numpy(xbits).view(torch.bfloat16).cuda();ids=torch.from_numpy(ids_host).cuda();rw=torch.from_numpy(rw_host).cuda();y=torch.empty((T,H),device='cuda',dtype=torch.bfloat16);act=torch.empty((R,I),device='cuda',dtype=torch.bfloat16)
    counts=torch.empty(E,device='cuda',dtype=torch.int32);routes=torch.empty((E,R),device='cuda',dtype=torch.int32);jobs=[torch.empty(256,device='cuda',dtype=torch.int32)for _ in range(2)]+[torch.empty(1,device='cuda',dtype=torch.int32)];extra=torch.full((410880+256,),0x5a,device='cuda',dtype=torch.uint8);stream=torch.cuda.current_stream().cuda_stream
    def call(on):
        args=[y,act,x,ids,rw,repacked if on else weights['w13'],weights['s13'],weights['w2'],weights['s2'],counts,routes]
        rc=new(0,*[v.data_ptr()for v in args],extra.data_ptr(),stream)if on else base(0,*[v.data_ptr()for v in args],*[v.data_ptr()for v in jobs],E,T,stream)
        assert rc==0,('actualproviderstatus',on,rc)
    result={'completed':False,'scope':m['scope'],'fixture_SHA256':{k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in [('input',xbits),('ids',ids_host),('routing_weights',rw_host)]},'active_experts':active.tolist(),'weight_slices_SHA256':hashes,'samples':[],'ordinary_kernel_scope':'Events spancompletepack/W13/SITU/W2 compoundprovider oncurrentstream, notexclusiveW13kernel; hostincludeslaunch+originalcompletion. CPUoracleoutsidealltiming.'}
    def save():pathlib.Path(output).write_bytes((json.dumps(result,indent=2)+'\n').encode())
    save();own={}
    for on in (False,True):
        call(on);torch.cuda.synchronize();yy=y.float().cpu().numpy();aa=act.float().cpu().numpy();own[on]=(yy.copy(),aa.copy());result['candidate'if on else'current']={'output_vs_CPU_FP32':errors(yy,oracle_y),'activation_vs_CPU_FP32':errors(aa,activation_ref)}
        if on:result['raw_W13_FP32_vs_CPU_FP32']=errors(extra[:R*N*4].view(torch.float32).cpu().numpy().reshape(R,N),oracle)
        save() # Persistactualfullcomparisonbeforeanyquality/timingclassification.
    result['candidate_vs_current']={'output':errors(own[True][0],own[False][0]),'activation':errors(own[True][1],own[False][1])};assert all(errors(v[0],v[0])['finite']and errors(v[1],v[1])['finite']for v in own.values())
    assert result['raw_W13_FP32_vs_CPU_FP32']['finite'],'raw_F32_nonfinite'
    save()
    for _ in range(3):call(False);call(True)
    torch.cuda.synchronize();a,b=[torch.cuda.Event(enable_timing=True)for _ in range(2)]
    for pair in range(8):
        for on in ([False,True]if pair%2==0 else[True,False]):
            tick=time.perf_counter();a.record();call(on);b.record();b.synchronize();result['samples'].append({'pair':pair,'candidate':on,'compound_provider_event_ms':a.elapsed_time(b),'host_complete_ms':(time.perf_counter()-tick)*1000});save()
    for on in (False,True):
        call(on);torch.cuda.synchronize();assert np.array_equal(y.float().cpu().numpy(),own[on][0])and np.array_equal(act.float().cpu().numpy(),own[on][1]),'ownrepeat_mismatch'
    assert torch.all(extra[410880:]==0x5a);result.update(completed=True,ownrepeat_both=True,canary=True,actual_TP_slice='globalTP16rank0/I192/activationsH3584, noTPcommunication',no_Model_quality_claim=True)
    del weights,repacked,x,ids,rw,y,act,counts,routes,jobs,extra,a,b
    torch.cuda.synchronize();torch.cuda.empty_cache();result['normal_owner_close']=True;save()

if __name__=='__main__':
    import sys
    assert len(sys.argv)==3,'Notanautolauncher: rootGPU GO +manifest/outputrequired';run(*sys.argv[1:])
