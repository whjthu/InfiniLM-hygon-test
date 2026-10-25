"""Future single-owner real routed-operator adjudication, NOT a launcher.

New W13 A8 candidate + unchanged original I192 baseline. Actualr4674 M8 duplicatedtoT16 is explicit;
no random latent/routes, no Model/CCL, no FP64. No execution without root GO.
"""
import pathlib,json,hashlib,time,ctypes,contextlib
import numpy as np
E,T,H,I,N,R=896,16,3584,192,384,256
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
    m=json.loads(pathlib.Path(manifest).read_text());assert digest(m['native'])==m['native_SHA256']
    receipt=pathlib.Path(m['cpu_receipt_path']);assert digest(receipt)==m['cpu_receipt_SHA256']
    cpu=json.loads(receipt.read_text());assert cpu['completed'] and cpu['CPU_only'] and not cpu['GPU'] and not cpu['dlopen']
    assert cpu['native_SHA256']==m['native_SHA256']
    assert digest(m['table_native'])==m['table_native_SHA256']
    table_cpu=json.loads(pathlib.Path(m['table_cpu_receipt_path']).read_text());assert digest(m['table_cpu_receipt_path'])==m['table_cpu_receipt_SHA256']
    assert table_cpu['completed']and table_cpu['CPU_only']and not table_cpu['GPU']and not table_cpu['dlopen']
    assert table_cpu['native_SHA256']==m['table_native_SHA256']
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    assert digest(m['native'])==m['native_SHA256']
    tablelib=ctypes.CDLL(m['table_native'],mode=ctypes.RTLD_LOCAL);bitlib=ctypes.CDLL(m['native'],mode=ctypes.RTLD_LOCAL)
    base=tablelib.modern_kimi_grouped_I192_r8669y;base.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*14+[ctypes.c_size_t]*2+[ctypes.c_void_p]
    table=tablelib.modern_kimi_A8_W13_r8993;bit=bitlib.modern_kimi_A8_W13_r8993
    for fn in [table,bit]:fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*13;fn.restype=ctypes.c_int
    class DlInfo(ctypes.Structure):
        _fields_=[('dli_fname',ctypes.c_char_p),('dli_fbase',ctypes.c_void_p),('dli_sname',ctypes.c_char_p),('dli_saddr',ctypes.c_void_p)]
    dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int
    binding={}
    for name,lib,expected in [('B_957_table',tablelib,m['table_native']),('C_bitfield',bitlib,m['native'])]:
        binding[name]={}
        for symbol in ['modern_kimi_A8_W13_r8993','kimi_A8_HIP_matrix_r8993']:
            info=DlInfo();address=ctypes.cast(getattr(lib,symbol),ctypes.c_void_p).value;assert dl.dladdr(address,ctypes.byref(info))!=0
            origin=str(pathlib.Path(info.dli_fname.decode()).resolve());assert origin==str(pathlib.Path(expected).resolve()),('DSO_origin_mismatch',name,symbol,origin,expected)
            binding[name][symbol]={'origin':origin,'base':int(info.dli_fbase),'address':address}
        assert len({v['base']for v in binding[name].values()})==1
    assert binding['B_957_table']['modern_kimi_A8_W13_r8993']['base']!=binding['C_bitfield']['modern_kimi_A8_W13_r8993']['base'],'DSO_base_collision'
    xbits,ids_host,rw_host=fixture(m['fixture_refs']);active=np.unique(ids_host);flat=ids_host.reshape(-1)
    root=pathlib.Path('/shared/Kimi-K3');mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.1.block_sparse_moe.experts.'
    # Full E896 tensor extents; only actual activeexpert pages populated/read.
    # No weight outside actualrouteIDs is accessed. Keep allcounts/routes E896.
    shapes={'w13':(E,N,H//2),'s13':(E,N,H//32),'w2':(E,H,I//2),'s2':(E,H,I//32)}
    weights={k:torch.empty(sh,device='cuda',dtype=torch.uint8)for k,sh in shapes.items()}
    oracle=np.empty((R,N),np.float32);oracle_a8=np.zeros((R,N),np.float32);oracle_y=np.zeros((T,H),np.float32);host={};hashes={}
    xf=fp(xbits).reshape(T,H//32,32);sx=np.max(np.abs(xf),axis=-1).astype(np.float32)/np.float32(127);sx=np.where(sx>0,sx,np.float32(1)).astype(np.float32);qx=np.clip(np.rint(xf/sx[:,:,None]),-127,127).astype(np.int8).reshape(T,H)
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
            route=np.flatnonzero(flat==e);oracle[route]=fp(xbits)[route//16]@decode(w13,s13).T
            nib=np.empty((N,H),np.uint8);nib[:,::2]=w13&15;nib[:,1::2]=w13>>4
            wi=np.asarray([0,1,2,3,4,6,8,12],np.int8)[nib&7]*np.where(nib&8,np.int8(-1),np.int8(1));assert not np.any(s13==255)
            for kg in range(H//32):
                partial=qx[route//16,kg*32:(kg+1)*32].astype(np.int32)@wi[:,kg*32:(kg+1)*32].astype(np.int32).T
                ws=np.ldexp(np.ones(N,np.float32),s13[:,kg].astype(np.int32)-128)
                oracle_a8[route]+=(partial.astype(np.float32)*sx[route//16,kg,None])*ws[None,:]
            host[int(e)]=(w2,s2)
    # CPU fullroutedoutput reference: FP32matmul + originalSITU + BF16act + FP32W2.
    gate,up=oracle[:,:I],oracle[:,I:]
    activation_ref=(np.float32(4)*np.tanh(gate/np.float32(4))/(1+np.exp(-gate)))*(np.float32(25)*np.tanh(up/np.float32(25)))
    activation_ref=torch.from_numpy(activation_ref).to(torch.bfloat16).float().numpy()
    for e in active:
        route=np.flatnonzero(flat==e);w2,s2=host[int(e)];partial=activation_ref[route]@decode(w2,s2).T
        for j,rid in enumerate(route):oracle_y[rid//16]+=np.float32(rw_host.reshape(-1)[rid])*partial[j]
    host.clear()
    x=torch.from_numpy(xbits).view(torch.bfloat16).cuda();ids=torch.from_numpy(ids_host).cuda();rw=torch.from_numpy(rw_host).cuda();y=torch.empty((T,H),device='cuda',dtype=torch.bfloat16);act=torch.empty((R,I),device='cuda',dtype=torch.bfloat16)
    counts=torch.empty(E,device='cuda',dtype=torch.int32);routes=torch.empty((E,R),device='cuda',dtype=torch.int32);jobs=[torch.empty(256,device='cuda',dtype=torch.int32)for _ in range(2)]+[torch.empty(1,device='cuda',dtype=torch.int32)];extra=torch.full((475648+256,),0x5a,device='cuda',dtype=torch.uint8);stream=torch.cuda.current_stream().cuda_stream;assert stream==0,'default_stream_only_ABI_scope'
    arms=['A_original','B_957_table','C_bitfield']
    def call(arm):
        args=[y,act,x,ids,rw,weights['w13'],weights['s13'],weights['w2'],weights['s2'],counts,routes]
        rc=base(0,*[v.data_ptr()for v in args],*[v.data_ptr()for v in jobs],E,T,stream)if arm==0 else[None,table,bit][arm](0,*[v.data_ptr()for v in args],extra.data_ptr(),stream)
        assert rc==0,('actualproviderstatus',arms[arm],rc)
    result={'completed':False,'scope':m['scope'],'fixture_SHA256':{k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in [('input',xbits),('ids',ids_host),('routing_weights',rw_host)]},'active_experts':active.tolist(),'weight_slices_SHA256':hashes,'samples':[],'qualification':{},'actual_DSO_origins':binding,'stream_ABI_scope':'defaultstream0 only, ctypesRTLD_LOCAL twoDSOs independenthandles; noModel claim','timed_scope':'Aquant/routepack/grouping/Wdecode/W13/SiTU/originalW2/drain completeprovider; CPUoracle outsidealltiming','raw_A8_CPU_relativeRMS_limit':1e-3,'fixed_calls':{'qualification':3,'warm':9,'measure':24,'ownrepeat':3,'total':39}}
    def save():pathlib.Path(output).write_bytes((json.dumps(result,indent=2)+'\n').encode())
    save();own={}
    for arm in range(3):
        call(arm);torch.cuda.synchronize();yy=y.float().cpu().numpy();aa=act.float().cpu().numpy();own[arm]=(yy.copy(),aa.copy())
        d={'output_vs_CPU_FP32':errors(yy,oracle_y),'activation_vs_CPU_FP32':errors(aa,activation_ref)};result['qualification'][arms[arm]]=d;save()
        assert d['output_vs_CPU_FP32']['finite']and d['activation_vs_CPU_FP32']['finite'],'nonfinite_output'
        if arm:
            d['raw_W13_vs_A8_CPU']=errors(extra[:R*N*4].view(torch.float32).cpu().numpy().reshape(R,N),oracle_a8)
            d['device_error_flag']=int(extra[475392:475396].view(torch.int32).item());d['canary']=bool(torch.all(extra[475648:]==0x5a));save()
            assert d['canary'],'first_canary_failure'
            assert d['device_error_flag']==0,'device_error'
            assert d['raw_W13_vs_A8_CPU']['finite']and d['raw_W13_vs_A8_CPU']['relativeRMS']<=1e-3,'STOP_raw_A8_CPU_disagreement_beforetiming'
            q_actual=extra[410880:468224].view(torch.int8).cpu().numpy().reshape(T,H);sx_actual=extra[468224:475392].view(torch.float32).cpu().numpy().reshape(T,H//32)
            d['quant_diagnostic']={'q_changed':int(np.count_nonzero(q_actual!=qx)),'scale_maxabs':float(np.max(np.abs(sx_actual-sx)))}
        save()
    result['crossarm_diagnostic']={arms[a]+'_vs_'+arms[b]:{'output':errors(own[a][0],own[b][0]),'activation':errors(own[a][1],own[b][1])}for a,b in [(1,0),(2,0),(2,1)]};save()
    for _ in range(3):
        for arm in range(3):call(arm)
    torch.cuda.synchronize();a,b=[torch.cuda.Event(enable_timing=True)for _ in range(2)]
    for pair in range(8):
        for arm in ([0,1,2]if pair%2==0 else[2,1,0]):
            tick=time.perf_counter();a.record();call(arm);b.record();b.synchronize();result['samples'].append({'pair':pair,'arm':arms[arm],'compound_provider_event_ms':a.elapsed_time(b),'host_complete_ms':(time.perf_counter()-tick)*1000});save()
    for arm in range(3):
        call(arm);torch.cuda.synchronize();assert np.array_equal(y.float().cpu().numpy(),own[arm][0])and np.array_equal(act.float().cpu().numpy(),own[arm][1]),'ownrepeat_mismatch'
        if arm:assert int(extra[475392:475396].view(torch.int32).item())==0
    assert torch.all(extra[475648:]==0x5a);result.update(completed=True,ownrepeat_all=True,canary=True,actual_TP_slice='TP16rank0/I192/H3584,noTPcommunication',no_Model_quality_claim=True)
    summary={}
    for arm in arms:
        summary[arm]={key:float(np.mean([v[key]for v in result['samples']if v['arm']==arm]))for key in ['compound_provider_event_ms','host_complete_ms']}
    result['timing_summary']=summary
    del weights,x,ids,rw,y,act,counts,routes,jobs,extra,a,b
    torch.cuda.synchronize();torch.cuda.empty_cache();result['normal_owner_close']=True;save()

if __name__=='__main__':
    import sys
    assert len(sys.argv)==3,'rootGPU GO+manifest/output required';run(*sys.argv[1:])
