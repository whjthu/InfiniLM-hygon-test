"""One-layer current Delta C1T1 OFF/ON/OFF diagnostic; no Model TPS claim."""
from pathlib import Path
import json,os,hashlib,ctypes,threading,time,copy,concurrent.futures,sys
R=Path('/refactor');D=Path(__file__).resolve().parent;S=D
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np,torch,infinilm
    assert sha(R/'incoming/Kimi-latest-service-r275/runtime/stage_control_r2386.py')=='269cc1a5097e0264052d1c929a18ac0bbfe20ade0b44947455dd21bf9e6b804d'
    sys.path.insert(0,str(R/'incoming/Kimi-latest-service-r275/runtime'))
    from stage_control_r2386 import StageControl
    m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert sha(D/'GPU_MANIFEST.json')==os.environ['KIMI_R345_GPU_MANIFEST_SHA']
    assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    assert sha(S/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256']
    sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
    cp=R/'candidates/Kimi-KDA-C1-layout-Delta-r345/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256']
    c=json.loads(cp.read_bytes());assert c['completed'] and c['passed'] and c['CPU_only'] and not c['Native_dlopen']
    for key in ['native','Main']:assert sha(c[key])==c[key+'_SHA256']
    for v in c['actual_dependency_library_SHA256'].values():assert sha(v['path'])==v['SHA256']
    fixture=json.loads((S/'fixture_manifest.json').read_bytes());assert sha(fixture['norm']['path'])==fixture['norm']['SHA256']
    assert fixture['norm']['shape']==[544,7168] and fixture['nodes']==['a14r2n08','a14r2n09']
    norm=np.fromfile(fixture['norm']['path'],dtype='<u2').reshape(544,7168)
    cfg=Path(fixture['config']).read_text();node=int(os.environ['SLURM_PROCID']);assert node in [0,1]
    actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];tag='Kimi345-C1-layout-Delta'
    output=R/'results'/(tag+'-'+actual+'-node'+str(node)+'.json');assert not output.exists()
    torch.set_num_threads(1);local=threading.local();lock=threading.Lock()
    report={'completed':False,'node':node,'actual_GPU':actual,'source_manifest_SHA256':sha(D/'GPU_MANIFEST.json'),
            'CPU_receipt_SHA256':sha(cp),'Native_SHA256':c['native_SHA256'],'Main_SHA256':c['Main_SHA256'],
            'scope':fixture['scope'],'C':1,'T':1,'pool':3,'Math':0,'capture':False,'GPU_F64':False,
            'raw_tensors_saved':False,'workers':[{}for _ in range(8)]}
    def save():
        snap=copy.deepcopy(local.own)if hasattr(local,'own')else None
        with lock:
            if snap is not None:report['workers'][local.device]=snap
            tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(output)
    save();ctl=StageControl(node,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,os.environ['KIMI_ENGINE_NONCE'],2,360)
    ctl.collect('C1_frontend',{'node':node,'Native_SHA256':c['native_SHA256']});lib=ctypes.CDLL(c['native'])
    class DlInfo(ctypes.Structure):
        _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
    dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)]
    actual_bindings={}
    for symbol,key in [('layer0_create_r9038','native'),('modern_kimi_KDA_input_counts_r4305','Main')]:
        info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info))
        actual_path=Path(info.filename.decode()).resolve();assert actual_path==Path(c[key]).resolve() and sha(actual_path)==c[key+'_SHA256']
        actual_bindings[symbol]={'DSO_SHA256':sha(actual_path),'exact_expected_path':True}
    for name,meta in c['actual_dependency_library_SHA256'].items():
        paths={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if line.split()[-1].startswith('/') and Path(line.split()[-1]).name==name}
        assert len(paths)==1 and sha(next(iter(paths)))==meta['SHA256'],'Actual loaded dependency differs: '+name
        actual_bindings[name]={'actual_loaded_SHA256':meta['SHA256']}
    report['actual_DSO_bindings']=actual_bindings;save()
    lib.layer0_error_r9038.restype=ctypes.c_char_p;lib.layer0_id_size_r9038.restype=ctypes.c_size_t
    lib.layer0_id_r9038.argtypes=[ctypes.c_void_p];lib.layer0_create_r9038.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.layer0_create_r9038.restype=ctypes.c_void_p
    lib.layer0_load_local_r9038.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
    for name in ['finish','reset','close']:getattr(lib,'layer0_'+name+'_r9038').argtypes=[ctypes.c_void_p]
    lib.layer0_batch_run_r9122.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]+[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_states_r9113.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_restore_r309.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_profile_mode_r9140.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.layer0_fused_mode_r343.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.layer0_fused_counts_r343.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_fused_diag_r345.argtypes=[ctypes.c_void_p];lib.layer0_fused_diag_r345.restype=ctypes.c_char_p
    lib.layer0_profile_last_r9140.argtypes=[ctypes.c_void_p];lib.layer0_profile_last_r9140.restype=ctypes.c_char_p
    lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];assert lib.modern_kimi_reduce_policy_r3226(0)==0
    def check(code):
        if code:raise RuntimeError(lib.layer0_error_r9038().decode())
    uid=(ctypes.c_ubyte*lib.layer0_id_size_r9038())()
    if node==0:check(lib.layer0_id_r9038(uid))
    v=ctl.collect('C1_UID',{'id':bytes(uid).hex()if node==0 else None});ctypes.memmove(uid,bytes.fromhex(v[0]['id']),len(uid))
    barrier=threading.Barrier(8,timeout=360)
    def worker(device):
        rank=node*8+device;own={'rank':rank,'normal_close':False,'samples':[]};local.device=device;local.own=own;ctx=None
        try:
            torch.cuda.set_device(device);ctx=lib.layer0_create_r9038(cfg.encode(),rank,device,uid);assert ctx,lib.layer0_error_r9038().decode()
            for key,meta in sorted(fixture['weights'][str(rank)].items()):
                assert sha(meta['path'])==meta['SHA256'];a=np.fromfile(meta['path'],dtype='<u2'if meta['dtype']=='BF16'else'<f4').reshape(meta['shape'])
                shape=(ctypes.c_size_t*a.ndim)(*a.shape);check(lib.layer0_load_local_r9038(ctx,key.encode(),a.ctypes.data,a.ndim,shape,1 if meta['dtype']=='BF16'else 0))
            check(lib.layer0_finish_r9038(ctx));barrier.wait()
            if device==0:ctl.collect('C1_allweights',{'node':node,'owners':8})
            barrier.wait();measurement_started=time.monotonic()
            def states():
                conv=np.empty((3,2304,3),np.uint16);ssm=np.empty((3,6,128,128),np.uint16)
                check(lib.layer0_states_r9113(ctx,conv.ctypes.data,ssm.ctypes.data));return conv,ssm
            def run(x,read,write):
                x=np.ascontiguousarray(x);n=len(x);arrays=[np.asarray(v,np.int32)for v in [[1],[512 if n==1 else 0],[0,n],[read],[write]]]
                guard=np.full(n*7168+64,0x7fc1,np.uint16);out=guard[32:-32];digest=hashlib.sha256(x.tobytes()).hexdigest()
                t=time.perf_counter();check(lib.layer0_batch_run_r9122(ctx,n,1,*[a.ctypes.data for a in arrays],0,x.ctypes.data,out.ctypes.data));elapsed=time.perf_counter()-t
                assert np.all(guard[:32]==0x7fc1) and np.all(guard[-32:]==0x7fc1) and not np.any((out&0x7f80)==0x7f80)
                assert hashlib.sha256(x.tobytes()).hexdigest()==digest;return out.copy(),elapsed
            check(lib.layer0_reset_r9038(ctx));run(norm[:512],1,1);baseline=states()
            assert np.any(baseline[0][1]) and np.any(baseline[1][1])
            reference=None;counter=0
            schedule=[(r,on)for r in range(8)for on in ([False,True,True,False]if r%2==0 else[True,False,False,True])]
            for block,(round_id,on) in enumerate(schedule):
                arm='ON'if on else'OFF'
                check(lib.layer0_profile_mode_r9140(ctx,0));check(lib.layer0_fused_mode_r343(ctx,int(on)))
                for trial in range(10):
                    assert time.monotonic()-measurement_started<300,'Bounded component measurement expired'
                    check(lib.layer0_restore_r309(ctx,baseline[0].ctypes.data,baseline[1].ctypes.data))
                    barrier.wait()
                    if device==0:ctl.collect('C1_start:'+str(counter),{'node':node})
                    counter+=1;barrier.wait()
                    output_value,elapsed=run(norm[512:513],1,2);profile=None
                    if on and 'first_ON_dispatch'not in own:
                        counts=(ctypes.c_uint64*2)();check(lib.layer0_fused_counts_r343(ctx,counts));diag=lib.layer0_fused_diag_r345(ctx);own['first_ON_dispatch']={'counts':list(counts),'metadata':json.loads(diag.decode())if diag else None};save()
                        assert list(counts)==[1,0],('firstON eligibility failed',own['first_ON_dispatch'])
                    updated=states()
                    assert all(np.array_equal(a[[0,1]],b[[0,1]])for a,b in zip(baseline,updated)),'Untouched source/canary state rows changed'
                    if reference is None:reference=(output_value,updated)
                    # An instrumentation-only experiment: off/on use the same math,
                    # inputs and starting state. Equality diagnoses observer effects,
                    # not a blanket requirement for future model optimizations.
                    equal=np.array_equal(reference[0],output_value) and all(np.array_equal(a,b)for a,b in zip(reference[1],updated))
                    def f32(v):return (v.astype(np.uint32)<<16).view(np.float32)
                    diag={}
                    for name,old_value,new_value in [('output',reference[0],output_value),('conv',reference[1][0],updated[0]),('state',reference[1][1],updated[1])]:
                        aa,bb=f32(old_value),f32(new_value);assert np.isfinite(aa).all()and np.isfinite(bb).all()
                        diag[name]={'max_abs':float(np.max(np.abs(aa-bb))),'rms':float(np.sqrt(np.mean((aa-bb)**2)))}
                        assert np.allclose(aa,bb,atol=.05,rtol=.02),(name,diag[name])
                    if not on:assert equal,'baseline self drift in output/state'

                    own['samples'].append({'arm':arm,'block':block,'round':round_id,'numerical':diag,'trial':trial,'warmup':trial<2,'API_wall_ms':1000*elapsed,
                                          'output_state_equal_to_OFF':equal,'source_rows_unchanged':True,'profile':profile})
                check(lib.layer0_profile_mode_r9140(ctx,0));save()
            counts=(ctypes.c_uint64*2)();check(lib.layer0_fused_counts_r343(ctx,counts));own['fused_counts']=list(counts);assert counts[0]==160 and counts[1]==0,('fused eligibility failed',list(counts))
            own.update(completed=True,measurement_elapsed_s=time.monotonic()-measurement_started,layer_calls=321);save();barrier.wait()
        except BaseException as e:
            own['error']=repr(e);save()
            try:barrier.abort()
            except Exception:pass
            raise
        finally:
            if ctx is not None:own['normal_close']=lib.layer0_close_r9038(ctx)==0
            save()
        return own
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:workers=list(pool.map(worker,range(8)))
    assert all(w['completed'] and w['normal_close'] and len(w['samples'])==320 and 'error'not in w for w in workers)
    report.update(completed=True,normal_close=True,owners=8,Model_gain=False);save();ctl.collect('C1_normal',{'node':node,'normal':True})
    print('KIMI345_NORMAL '+str(node),flush=True)

if __name__=='__main__':main()
