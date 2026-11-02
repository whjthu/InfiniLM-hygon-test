"""One-layer current Delta C1T1 OFF/ON/OFF diagnostic; no Model TPS claim."""
from pathlib import Path
import json,os,hashlib,ctypes,threading,time,copy,concurrent.futures,sys
R=Path('/refactor');D=Path(__file__).resolve().parent;S=D.parent/'cpu-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    assert sha(R/'incoming/Kimi-latest-service-r275/runtime/stage_control_r2386.py')=='269cc1a5097e0264052d1c929a18ac0bbfe20ade0b44947455dd21bf9e6b804d'
    sys.path.insert(0,str(R/'incoming/Kimi-latest-service-r275/runtime'))
    from stage_control_r2386 import StageControl
    m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert sha(D/'GPU_MANIFEST.json')==os.environ['KIMI405_GPU_MANIFEST_SHA']
    assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    assert sha(S/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256']
    sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
    cp=R/'candidates/Kimi-C1-Delta-projection-fusion-r405/cpu-v2/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256']
    c=json.loads(cp.read_bytes());assert c['completed'] and c['passed'] and c['CPU_only'] and not c['Native_dlopen']
    for key in ['native']:assert sha(c[key])==c[key+'_SHA256']
    for p,h in c['actual_resolved_dependency_library_SHA256'].items():assert sha(p)==h
    fixture=json.loads((S/'fixture_manifest.json').read_bytes());assert sha(fixture['norm']['path'])==fixture['norm']['SHA256']
    assert fixture['norm']['shape']==[544,7168]
    for rank,items in fixture['weights'].items():
        for key,meta in items.items():
            expected=2 if meta['dtype']=='BF16'else 4
            for n in meta['shape']:expected*=n
            assert Path(meta['path']).stat().st_size==expected,(rank,key)
    norm=np.fromfile(fixture['norm']['path'],dtype='<u2').reshape(544,7168)
    cfg=Path(fixture['config']).read_text();import torch,infinilm
    node=int(os.environ['SLURM_PROCID']);assert node in [0,1]
    actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];tag='Kimi405-C1-Delta-projection'
    output=R/'results'/(tag+'-'+actual+'-node'+str(node)+'.json');assert not output.exists()
    torch.set_num_threads(1);local=threading.local();lock=threading.Lock()
    report={'completed':False,'node':node,'actual_GPU':actual,'source_manifest_SHA256':sha(D/'GPU_MANIFEST.json'),
            'CPU_receipt_SHA256':sha(cp),'Native_SHA256':c['native_SHA256'],'current_Native_SHA256':c['current_Native_SHA256'],
            'scope':fixture['scope'],'C':1,'T':1,'pool':3,'Math':0,'capture':False,'GPU_F64':False,
            'raw_tensors_saved':'node0rank0 first OFF/ON two fixtures only, R raw refs not Git','workers':[{}for _ in range(8)]}
    def save():
        snap=copy.deepcopy(local.own)if hasattr(local,'own')else None
        with lock:
            if snap is not None:report['workers'][local.device]=snap
            tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(output)
    save();ctl=StageControl(node,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,os.environ['KIMI_ENGINE_NONCE'],2,360)
    ctl.collect('C1_frontend',{'node':node,'Native_SHA256':c['native_SHA256']});lib=ctypes.CDLL(c['native'])
    class DlInfo(ctypes.Structure):
        _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
    dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)]
    actual_bindings={}
    for symbol,key in [('layer0_create_r9038','native'),('layer0_projection_stats_r405','native')]:
        info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info))
        actual_path=Path(info.filename.decode()).resolve();assert actual_path==Path(c[key]).resolve() and sha(actual_path)==c[key+'_SHA256']
        actual_bindings[symbol]={'DSO_SHA256':sha(actual_path),'exact_expected_path':True}
    for expected,h in c['actual_resolved_dependency_library_SHA256'].items():
        name=Path(expected).name
        paths={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if line.split()[-1].startswith('/') and Path(line.split()[-1]).name==name}
        # ldd loader itself may not appear under its resolved symlink name; all actual loaded basename matches must agree.
        if paths:assert len(paths)==1 and sha(next(iter(paths)))==h,('loaded dependency differs',name,paths)
        actual_bindings[name]={'expected_SHA256':h,'matching_loaded_paths':sorted(paths)}
    report['actual_DSO_bindings']=actual_bindings;save()
    lib.layer0_error_r9038.restype=ctypes.c_char_p;lib.layer0_id_size_r9038.restype=ctypes.c_size_t
    lib.layer0_id_r9038.argtypes=[ctypes.c_void_p];lib.layer0_create_r9038.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.layer0_create_r9038.restype=ctypes.c_void_p
    lib.layer0_load_local_r9038.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
    for name in ['finish','reset','close']:getattr(lib,'layer0_'+name+'_r9038').argtypes=[ctypes.c_void_p]
    lib.layer0_batch_run_r9122.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]+[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_states_r9113.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_restore_r309.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    lib.layer0_projection_mode_r405.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.layer0_projection_stats_r405.argtypes=[ctypes.c_void_p];lib.layer0_projection_stats_r405.restype=ctypes.c_char_p
    lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];assert lib.modern_kimi_reduce_policy_r3226(0)==0
    def check(code):
        if code:raise RuntimeError(lib.layer0_error_r9038().decode())
    uid=(ctypes.c_ubyte*lib.layer0_id_size_r9038())()
    if node==0:check(lib.layer0_id_r9038(uid))
    v=ctl.collect('C1_UID',{'id':bytes(uid).hex()if node==0 else None});ctypes.memmove(uid,bytes.fromhex(v[0]['id']),len(uid))
    barrier=threading.Barrier(8,timeout=360)
    def worker(device):
        rank=node*8+device;own={'rank':rank,'PID':os.getpid(),'TID':threading.get_native_id(),'normal_close':False,'samples':[],'restore_wall_ms':[]};local.device=device;local.own=own;ctx=None
        try:
            torch.cuda.set_device(device);ctx=lib.layer0_create_r9038(cfg.encode(),rank,device,uid);assert ctx,lib.layer0_error_r9038().decode()
            for key,meta in sorted(fixture['weights'][str(rank)].items()):
                assert sha(meta['path'])==meta['SHA256'];a=np.fromfile(meta['path'],dtype='<u2'if meta['dtype']=='BF16'else'<f4').reshape(meta['shape'])
                shape=(ctypes.c_size_t*a.ndim)(*a.shape);check(lib.layer0_load_local_r9038(ctx,key.encode(),a.ctypes.data,a.ndim,shape,1 if meta['dtype']=='BF16'else 0))
            t=time.perf_counter();check(lib.layer0_finish_r9038(ctx));own['finish_setup_wall_ms']=(time.perf_counter()-t)*1000;own['pack_stats']=json.loads(lib.layer0_projection_stats_r405(ctx));assert own['pack_stats']['packs']==1 and own['pack_stats']['packed_bytes']==45961216;barrier.wait()
            if device==0:ctl.collect('C1_allweights',{'node':node,'owners':8})
            barrier.wait();measurement_started=time.monotonic()
            def states():
                conv=np.empty((3,2304,3),np.uint16);ssm=np.empty((3,6,128,128),np.uint16)
                check(lib.layer0_states_r9113(ctx,conv.ctypes.data,ssm.ctypes.data));return conv,ssm
            def run(x,read,write):
                x=np.ascontiguousarray(x);n=len(x);arrays=[np.asarray(v,np.int32)for v in [[1],[512 if n==1 else 0],[0,n],[read],[write]]]
                guard=np.full(n*7168+64,0x7fc1,np.uint16);out=guard[32:-32];digest=hashlib.sha256(x.tobytes()).hexdigest()
                own['attempted']=own.get('attempted',0)+1;t=time.perf_counter();rc=lib.layer0_batch_run_r9122(ctx,n,1,*[a.ctypes.data for a in arrays],0,x.ctypes.data,out.ctypes.data);elapsed=time.perf_counter()-t;own['CAPI_returned']=own.get('CAPI_returned',0)+1
                if rc:own['failed_CAPI']=dict(rc=rc,error=lib.layer0_error_r9038().decode(),API_wall_ms=elapsed*1000);save();check(rc)
                own['last_output_canary']=bool(np.all(guard[:32]==0x7fc1) and np.all(guard[-32:]==0x7fc1))
                own['last_input_readonly']=hashlib.sha256(x.tobytes()).hexdigest()==digest;return out.copy(),elapsed
            check(lib.layer0_reset_r9038(ctx));run(norm[:512],1,1);baseline=states()
            assert np.any(baseline[0][1]) and np.any(baseline[1][1])
            from consumer import execute
            references={};counter=0
            class Backend:
                def restore(self):
                    assert time.monotonic()-measurement_started<300,'bounded measurement expired'
                    t=time.perf_counter();check(lib.layer0_restore_r309(ctx,baseline[0].ctypes.data,baseline[1].ctypes.data));own['restore_wall_ms'].append(1000*(time.perf_counter()-t))
                def mode(self,on):self.on=on;check(lib.layer0_projection_mode_r405(ctx,int(on)))
                def barrier(self,index):
                    barrier.wait()
                    if device==0:ctl.collect('C1_start:'+str(index),{'node':node})
                    barrier.wait()
                def stats(self):return json.loads(lib.layer0_projection_stats_r405(ctx))
                def run(self,fi):
                    # Two different existing normalized teacher rows; full Delta call once.
                    self.value,self.ms=run(norm[512+fi:513+fi],1,2);self.updated=states();self.fi=fi
                    return self.value
                def inspect(self,raw,fi):
                    source_ok=own.get('last_output_canary',False)and own.get('last_input_readonly',False)and all(np.array_equal(a[[0,1]],b[[0,1]])for a,b in zip(baseline,self.updated))
                    if fi not in references:
                        assert not self.on,'reference must be original OFF'
                        references[fi]=(self.value.copy(),tuple(v.copy()for v in self.updated))
                    ref=references[fi];diag={};all_good=source_ok
                    def f32(v):return(v.astype(np.uint32)<<16).view(np.float32)
                    for name,aa,bb in [('output',ref[0],self.value),('conv_written_row2',ref[1][0][2],self.updated[0][2]),('ssm_written_row2',ref[1][1][2],self.updated[1][2])]:
                        aa,bb=f32(aa),f32(bb);finite=bool(np.isfinite(aa).all()and np.isfinite(bb).all())
                        maxabs=float(np.max(np.abs(aa-bb)))if finite else None
                        nrms=float(np.sqrt(np.mean((aa-bb)**2))/(np.sqrt(np.mean(aa**2))+1e-20))if finite else None
                        scale=float(np.max(np.abs(aa)))if finite else None
                        passed=bool(finite and nrms<=.02 and maxabs<=max(.05,.1*scale))
                        diag[name]=dict(finite=finite,max_abs=maxabs,NRMS=nrms,reference_max_abs=scale,passed=passed);all_good &= passed
                    diag.update(passed=bool(all_good),source_state_rows_unchanged=source_ok,API_wall_ms=1000*self.ms,fixture=fi)
                    if node==0 and device==0 and (fi,self.on)not in raw_saved:
                        raw_saved.add((fi,self.on));p=R/'results'/('Kimi405-'+actual+'-warm-fi'+str(fi)+'-'+('ON'if self.on else'OFF')+'.npz')
                        np.savez_compressed(p,output=self.value,conv=self.updated[0],state=self.updated[1]);own.setdefault('warm_raw_refs',[]).append(dict(path=str(p),SHA256=sha(p),fixture=fi,arm='ON'if self.on else'OFF'))
                    return diag
            raw_saved=set()
            def completed_save(row):
                own['samples'].append(row)
                if row['arm']=='ON'and 'first_ON_dispatch'not in own:own['first_ON_dispatch']=row['counters_after']
                save()
            execute(Backend(),completed_save,lambda:time.monotonic()-measurement_started>=300)
            counts=json.loads(lib.layer0_projection_stats_r405(ctx));own['projection_counts']=counts
            assert counts['chosen']==160 and counts['consumer_counts']==[160]*4 and counts['fallback']==0 and counts['epochs']==321
            own.update(completed=True,measurement_elapsed_s=time.monotonic()-measurement_started,layer_calls=321);save();barrier.wait()
        except BaseException as e:
            own['error']=repr(e)
            try:save()
            except BaseException as s:own['save_error']=repr(s)
            try:barrier.abort()
            except Exception:pass
            raise
        finally:
            if ctx is not None:
                t=time.perf_counter();own['normal_close']=lib.layer0_close_r9038(ctx)==0;own['destroy_wall_ms']=1000*(time.perf_counter()-t)
            try:save()
            except BaseException as e:
                if 'error'not in own:raise
                print('FINAL_SAVE_ERROR '+repr(e),flush=True)
        return own
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:workers=list(pool.map(worker,range(8)))
    assert all(w['completed'] and w['normal_close'] and len(w['samples'])==320 and 'error'not in w for w in workers)
    report.update(completed=True,normal_close=True,owners=8,Model_gain=False);save();ctl.collect('C1_normal',{'node':node,'normal':True})
    print('KIMI405_NORMAL '+str(node),flush=True)

if __name__=='__main__':main()
