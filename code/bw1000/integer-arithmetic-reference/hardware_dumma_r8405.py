"""Eight-owner integer MMAC arithmetic reference; no model/quantizer claim."""
import pathlib,os,json,ctypes,hashlib,time,statistics,datetime,socket,subprocess,threading
import numpy as np
import torch,torch.multiprocessing as mp
from integer_probe_inputs_r8405 import patterns,expected_output
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
base=r/'candidates/hardware-dumma-r8403';root=r/'results'/('hardware-dumma-r8405-'+tag)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clocks():
 a=['/opt/hyhal/bin/hy-smi','--showclocks','--showproductname','--showpower','--json'];begin=time.time()
 p=subprocess.run(a,capture_output=True,text=True,timeout=20)
 return dict(start_epoch=begin,end_epoch=time.time(),code=p.returncode,stdout=p.stdout,stderr=p.stderr)
def worker(device,barrier):
 torch.set_num_threads(1);torch.cuda.set_device(device)
 cpus=sorted(os.sched_getaffinity(0));mask=cpus[device*4:(device+1)*4];assert len(mask)==4;os.sched_setaffinity(0,mask)
 props=torch.cuda.get_device_properties(device);assert 'gfx936'in getattr(props,'gcnArchName','')and props.multi_processor_count==80
 build=json.loads((base/'CPU-build.json').read_text());assert build['completed']and build['CPU_only']
 libpath=pathlib.Path(build['library']);assert sha(libpath)==build['library_SHA256']
 assert sha(r/'incoming/hardware_dumma_r8403.hip')==build['source_SHA256']
 assert any('16x16x32_i8'in s for s in build['mmac_instruction_lines'])and any('16x16x64_i4'in s for s in build['mmac_instruction_lines'])
 lib=ctypes.CDLL(str(libpath));fn=lib.hardware_dumma_r8403
 fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
 p=root.with_name(root.name+'-device'+str(device)+'.json')
 report=dict(completed=False,device=device,node=socket.gethostname(),step=tag,CPU_affinity=mask,
  hardware=dict(name=props.name,arch=getattr(props,'gcnArchName',''),CU=props.multi_processor_count,VRAM_bytes=props.total_memory),
  library_SHA256=sha(libpath),CPU_build_SHA256=sha(base/'CPU-build.json'),cases=[],numpy_version=np.__version__,torch_version=torch.__version__,
  source_SHA256={n:sha(r/'incoming'/n)for n in ['hardware_dumma_r8403.hip','hardware_dumma_r8405.py','hardware_dumma_GPU_r8405_inside.sh','integer_probe_inputs_r8405.py']},
  scope='Register-resident16x16 integer MMA repeated accumulation. Includes kernel input/output and launch; excludes CPUoracle/H2D/D2H. Not a physical peak, GEMM/quantizer, HBM-rate or ModelTPS test.')
 def save():
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
 save();stream=torch.cuda.Stream(device=device);torch.cuda.set_stream(stream)
 assert torch.cuda.mem_get_info(device)[0]>58*1024**3
 # Invalid metadata must reject before launch or dereferencing any device pointer.
 assert fn(2,1,None,None,None,1,1,stream.cuda_stream)==999
 assert fn(0,2,None,None,None,1,1,stream.cuda_stream)==999
 report['invalid_metadata_rejected']=True
 for op in [0,1]:
  K=32 if op==0 else 64;aa,bt,pa,pb,gold,value_range=patterns(device,op)
  for blocks,chains in [(1,1),(80,1),(640,1),(2560,1),(640,4),(2560,4)]:
   groups=blocks*chains;indices=np.arange(groups)%16;rawa=np.ascontiguousarray(pa[indices].reshape(-1));rawb=np.ascontiguousarray(pb[indices].reshape(-1))
   ag=torch.full((rawa.size+512,),0xa5,dtype=torch.uint8,device=device);bg=torch.full_like(ag,0x5a)
   a,b=ag[256:-256],bg[256:-256];a.copy_(torch.from_numpy(rawa));b.copy_(torch.from_numpy(rawb))
   ob=torch.full((groups*256+512,),123456789,dtype=torch.int32,device=device);out=ob[256:-256]
   loops=8192
   expected=expected_output(gold,blocks,chains,loops)
   def launch():assert fn(op,chains,a.data_ptr(),b.data_ptr(),out.data_ptr(),blocks,loops,stream.cuda_stream)==0
   def verify():
    torch.cuda.synchronize(device)
    got=out.cpu().numpy();diff=np.flatnonzero(got!=expected)
    if len(diff):raise RuntimeError(dict(op=op,blocks=blocks,chains=chains,first=int(diff[0]),actual=int(got[diff[0]]),expected=int(expected[diff[0]])))
    assert np.array_equal(a.cpu().numpy(),rawa)and np.array_equal(b.cpu().numpy(),rawb)
    assert bool((ag[:256]==0xa5).all()and(ag[-256:]==0xa5).all()and(bg[:256]==0x5a).all()and(bg[-256:]==0x5a).all())
    assert bool((ob[:256]==123456789).all()and(ob[-256:]==123456789).all())
    return hashlib.sha256(got.tobytes()).hexdigest()
   launch();actual_before=verify()
   for _ in range(3):launch()
   torch.cuda.synchronize(device);barrier.wait()
   telemetry=clocks()if device==0 else None;barrier.wait();samples=[];hosts=[];bounds=[]
   for trial in range(7):
    e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin=time.perf_counter();epoch=time.time();e0.record()
    for _ in range(8):launch()
    e1.record();e1.synchronize();samples.append(e0.elapsed_time(e1)/1000/8);hosts.append((time.perf_counter()-begin)/8);bounds.append([epoch,time.time()])
   actual_after=verify();barrier.wait()
   operations=2*16*16*K*loops*blocks*chains
   report['cases'].append(dict(dtype='signedINT8'if op==0 else'signedINT4',tile=[16,16,K],blocks=blocks,independent_chains_per_wave=chains,
    loops=loops,useful_integer_operations=operations,MAC_counts_as_2=True,warmup_kernels=3,trials=7,kernels_per_trial=8,
    event_seconds_per_kernel=samples,host_seconds_per_kernel=hosts,event_TOPS_median=operations/statistics.median(samples)/1e12,
    host_TOPS_median=operations/statistics.median(hosts)/1e12,event_CV=statistics.pstdev(samples)/statistics.mean(samples),
    full_CPU_INT64_oracle_and_all_guards_before_after=True,CPU_expected_SHA256=hashlib.sha256(expected.tobytes()).hexdigest(),
    GPU_actual_before_SHA256=actual_before,GPU_actual_after_SHA256=actual_after,operand_range=list(value_range),
    trial_epoch_boundaries=bounds,clock_observation_before_timing=telemetry))
   save();del a,b,ag,bg,out,ob;torch.cuda.synchronize(device);torch.cuda.empty_cache()
 report.update(completed=True,normal_tensor_cleanup=True);save()
if __name__=='__main__':
 assert not root.with_name(root.name+'.json').exists()
 ctx=mp.get_context('spawn');barrier=ctx.Barrier(8);mp.spawn(worker,args=(barrier,),nprocs=8,join=True)
 owners=[]
 for d in range(8):
  p=root.with_name(root.name+'-device'+str(d)+'.json');q=json.loads(p.read_text());assert q['completed']and q['normal_tensor_cleanup']
  owners.append(dict(file=p.name,SHA256=sha(p),device=d,cases=q['cases']))
 aggregate=dict(completed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=tag,node=socket.gethostname(),owners=owners,
  scope='Achieved arithmetic-only microbenchmark references, not certified physical peaks or Model gains.')
 p=root.with_name(root.name+'.json');assert not p.exists();p.write_text(json.dumps(aggregate,indent=2));print(json.dumps(dict(completed=True,path=str(p))))
