import pathlib,ast
r=pathlib.Path.cwd();p=r/'Kimi-W13-A8-bitfield-private-r8996';s=(r/'Kimi-W13-A8-ABI-private-r8993/Kimi_A8_ABI_real_r8993.py').read_text();s=s[:s.index("    lib=ctypes.CDLL(m['native'])")]
s=s.replace("    import torch","    assert digest(m['table_native'])==m['table_native_SHA256']\n    table_cpu=json.loads(pathlib.Path(m['table_cpu_receipt_path']).read_text());assert digest(m['table_cpu_receipt_path'])==m['table_cpu_receipt_SHA256']\n    assert table_cpu['completed']and table_cpu['CPU_only']and not table_cpu['GPU']and not table_cpu['dlopen']\n    assert table_cpu['native_SHA256']==m['table_native_SHA256']\n    import torch",1)
s+='''    tablelib=ctypes.CDLL(m['table_native'],mode=ctypes.RTLD_LOCAL);bitlib=ctypes.CDLL(m['native'],mode=ctypes.RTLD_LOCAL)
    base=tablelib.modern_kimi_grouped_I192_r8669y;base.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*14+[ctypes.c_size_t]*2+[ctypes.c_void_p]
    table=tablelib.modern_kimi_A8_W13_r8993;bit=bitlib.modern_kimi_A8_W13_r8993
    for fn in [table,bit]:fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*13;fn.restype=ctypes.c_int
    x=torch.from_numpy(xbits).view(torch.bfloat16).cuda();ids=torch.from_numpy(ids_host).cuda();rw=torch.from_numpy(rw_host).cuda();y=torch.empty((T,H),device='cuda',dtype=torch.bfloat16);act=torch.empty((R,I),device='cuda',dtype=torch.bfloat16)
    counts=torch.empty(E,device='cuda',dtype=torch.int32);routes=torch.empty((E,R),device='cuda',dtype=torch.int32);jobs=[torch.empty(256,device='cuda',dtype=torch.int32)for _ in range(2)]+[torch.empty(1,device='cuda',dtype=torch.int32)];extra=torch.full((475648+256,),0x5a,device='cuda',dtype=torch.uint8);stream=torch.cuda.current_stream().cuda_stream;assert stream==0,'default_stream_only_ABI_scope'
    arms=['A_original','B_957_table','C_bitfield']
    def call(arm):
        args=[y,act,x,ids,rw,weights['w13'],weights['s13'],weights['w2'],weights['s2'],counts,routes]
        rc=base(0,*[v.data_ptr()for v in args],*[v.data_ptr()for v in jobs],E,T,stream)if arm==0 else[None,table,bit][arm](0,*[v.data_ptr()for v in args],extra.data_ptr(),stream)
        assert rc==0,('actualproviderstatus',arms[arm],rc)
    result={'completed':False,'scope':m['scope'],'fixture_SHA256':{k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in [('input',xbits),('ids',ids_host),('routing_weights',rw_host)]},'active_experts':active.tolist(),'weight_slices_SHA256':hashes,'samples':[],'qualification':{},'stream_ABI_scope':'defaultstream0 only, ctypesRTLD_LOCAL twoDSOs independenthandles; noModel claim','timed_scope':'Aquant/routepack/grouping/Wdecode/W13/SiTU/originalW2/drain completeprovider; CPUoracle outsidealltiming','raw_A8_CPU_relativeRMS_limit':1e-3,'fixed_calls':{'qualification':3,'warm':9,'measure':24,'ownrepeat':3,'total':39}}
    def save():pathlib.Path(output).write_bytes((json.dumps(result,indent=2)+'\\n').encode())
    save();own={}
    for arm in range(3):
        call(arm);torch.cuda.synchronize();yy=y.float().cpu().numpy();aa=act.float().cpu().numpy();own[arm]=(yy.copy(),aa.copy())
        d={'output_vs_CPU_FP32':errors(yy,oracle_y),'activation_vs_CPU_FP32':errors(aa,activation_ref)};result['qualification'][arms[arm]]=d
        assert d['output_vs_CPU_FP32']['finite']and d['activation_vs_CPU_FP32']['finite'],'nonfinite_output'
        if arm:
            d['raw_W13_vs_A8_CPU']=errors(extra[:R*N*4].view(torch.float32).cpu().numpy().reshape(R,N),oracle_a8)
            d['device_error_flag']=int(extra[475392:475396].view(torch.int32).item());save()
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
'''
ast.parse(s);(p/'Kimi_A8_ABC_real_r8996.py').write_bytes(s.encode());print('ABC source ready, noGPUcontroller/dispatch')
