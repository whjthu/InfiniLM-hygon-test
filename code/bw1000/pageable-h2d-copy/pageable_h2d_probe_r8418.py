"""Finite synthetic loader-copy comparison, no model or runtime installation change."""
import pathlib,os,json,hashlib,time,statistics,subprocess,sys,datetime

def worker(device,arm):
    # Fresh process and environment per arm, before importing or initializing HIP.
    flag=None if arm.startswith('O') else '4194304'
    if flag is None:os.environ.pop('GPU_PINNED_MIN_XFER_SIZE',None)
    else:os.environ['GPU_PINNED_MIN_XFER_SIZE']=flag
    import torch,ctypes,mmap,resource
    torch.set_num_threads(1);torch.cuda.set_device(device)
    cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32
    os.sched_setaffinity(0,cpus[device*4:(device+1)*4])
    root=pathlib.Path('/refactor/results');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    output=root/f'pageable-h2d-r8418-{tag}-{arm}-device{device}.json'
    assert not output.exists()
    scratch=pathlib.Path('/local')/f'pageable-h2d-r8418-{tag}-{arm}-{device}-{os.getpid()}'
    scratch.mkdir(exist_ok=False)
    libpath=pathlib.Path('/opt/dtk/hip/lib/libgalaxyhip.so.5')
    lib=ctypes.CDLL(str(libpath))
    copy=lib.hipMemcpyAsync;copy.restype=ctypes.c_int
    copy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p]
    report=dict(completed=False,step=tag,device=device,arm=arm,flag=flag,flag_units='KB in upstream ROCclr; installedDTK acceptance/mechanism not proved by throughput alone',
      source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
      HIP_library_SHA256=hashlib.sha256(libpath.read_bytes()).hexdigest(),cases=[],
      scope='Synthetic warm file-cache mmap -> HIP H2D -> unmap, freshprocess ABBA, no Model weights/full-load speed or observed USERPTR cause claim.')
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(output)
    save();stream=torch.cuda.Stream(device=device)
    try:
        for size in [1024**2,32*1024**2,128*1024**2]:
            file=scratch/f'source-{size}.bin';block=bytes(((i+device*17)%256 for i in range(65536)))
            with file.open('xb')as f:
                for _ in range(size//len(block)):f.write(block)
            # Hash and warm the exact same immutable source before each arm's copies.
            expected=hashlib.sha256(file.read_bytes()).hexdigest()
            guard=torch.full((size+512,),0xa5,dtype=torch.uint8,device=device);dst=guard[256:-256]
            torch.cuda.synchronize(device)
            def verify():
                stream.synchronize();raw=dst.cpu().numpy().tobytes()
                actual=hashlib.sha256(raw).hexdigest();assert actual==expected
                assert bool((guard[:256]==0xa5).all()and(guard[-256:]==0xa5).all())
                return actual
            timings=[];hashes=[]
            with file.open('rb')as f:
                for iteration in range(19):
                    begin=time.perf_counter();fault0=resource.getrusage(resource.RUSAGE_SELF)
                    mm=mmap.mmap(f.fileno(),size,access=mmap.ACCESS_COPY)
                    buf=(ctypes.c_ubyte*size).from_buffer(mm)
                    copy_start=time.perf_counter();cpu_start=time.thread_time()
                    code=copy(dst.data_ptr(),ctypes.addressof(buf),size,1,stream.cuda_stream)
                    assert code==0,('hipMemcpyAsync H2D',code)
                    stream.synchronize();copy_end=time.perf_counter();cpu_end=time.thread_time()
                    del buf;unmap_start=time.perf_counter();mm.close();end=time.perf_counter()
                    fault1=resource.getrusage(resource.RUSAGE_SELF)
                    if iteration>=3:
                        timings.append(dict(H2D_completed_s=copy_end-copy_start,unmap_s=end-unmap_start,
                          full_map_copy_unmap_s=end-begin,copy_thread_CPU_s=cpu_end-cpu_start,
                          minorfaults=fault1.ru_minflt-fault0.ru_minflt,majorfaults=fault1.ru_majflt-fault0.ru_majflt))
                    if iteration in [2,18]:hashes.append(verify())
            report['cases'].append(dict(bytes=size,warmup_copies=3,measured_copies=16,
                source_SHA256=expected,GPU_before_after_SHA256=hashes,full_bytes_and_guards_passed=True,samples=timings,
                median_H2D_completed_GB_s=size/statistics.median(x['H2D_completed_s']for x in timings)/1e9,
                median_map_copy_unmap_GB_s=size/statistics.median(x['full_map_copy_unmap_s']for x in timings)/1e9))
            save();del dst,guard;torch.cuda.synchronize(device);torch.cuda.empty_cache()
            assert file.parent.resolve()==scratch.resolve();file.unlink()
        report.update(completed=True,normal_tensor_cleanup=True)
    except BaseException as exc:
        report.update(error=repr(exc));raise
    finally:
        save()
        # Only the three exact synthetic files created here may be removed.
        for size in [1024**2,32*1024**2,128*1024**2]:
            p=scratch/f'source-{size}.bin'
            if p.exists():assert p.parent.resolve()==scratch.resolve();p.unlink()
        scratch.rmdir()

if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[1]),sys.argv[2]);raise SystemExit
    tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    for arm in ['O1','S1','S2','O2']:
        children=[subprocess.Popen([sys.executable,'-B',__file__,str(device),arm])for device in range(8)]
        codes=[p.wait()for p in children];assert codes==[0]*8,(arm,codes)
    owners=[]
    for arm in ['O1','S1','S2','O2']:
        for device in range(8):
            p=pathlib.Path('/refactor/results')/f'pageable-h2d-r8418-{tag}-{arm}-device{device}.json'
            q=json.loads(p.read_text());assert q['completed']and q['normal_tensor_cleanup']
            owners.append(dict(arm=arm,device=device,file=p.name,SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),cases=q['cases']))
    p=pathlib.Path('/refactor/results')/f'pageable-h2d-r8418-{tag}.json'
    assert not p.exists();p.write_text(json.dumps(dict(completed=True,step=tag,owners=owners),indent=2))
    print(json.dumps(dict(completed=True,file=str(p))))
