"""Finite arithmetic-only FP32 vector reference, never a physical ceiling."""
import os,pathlib,json,hashlib,ctypes,time,statistics,socket
import numpy as np
import torch,torch.multiprocessing as mp
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];root=r/'results'/('hardware-fp32-fma-r8802-'+tag);base=r/'candidates/hardware-fp32-fma-r8800'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def worker(dev,barrier):
 torch.set_num_threads(1);torch.cuda.set_device(dev);cpus=sorted(os.sched_getaffinity(0));mask=cpus[4*dev:4*dev+4];assert len(mask)==4;os.sched_setaffinity(0,mask)
 props=torch.cuda.get_device_properties(dev);assert 'gfx936'in props.gcnArchName and props.multi_processor_count==80 and props.total_memory>=63*1024**3
 assert torch.cuda.mem_get_info(dev)[0]>=6*1024**3
 b=json.loads((base/'CPU-ISA-qualified-r8804.json').read_text());assert b['completed']and b['CPU_only']and len(b['fma_instruction_lines'])==4 and b['independent_FP32_chains']==8 and not b['matrix_instruction_lines'];assert sha(b['library'])==b['library_SHA256']and sha(r/'incoming/hardware_fp32_fma_r8800.hip')==b['source_SHA256']
 lib=ctypes.CDLL(b['library']);fn=lib.hardware_fp32_fma_r8800;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
 blocks=props.multi_processor_count*16;threads=256;loops=65536;n=blocks*threads;guard=256;sentinel=12345.;ops=2*8*n*loops
 aa=((np.arange(n,dtype=np.int32)[:,None]%8+np.arange(8)[None,:]+1)/256).astype(np.float32).reshape(-1)
 ag=torch.full((len(aa)+2*guard,),sentinel,device=dev);og=torch.full_like(ag,sentinel);a=ag[guard:-guard];out=og[guard:-guard];a.copy_(torch.from_numpy(aa));stream=torch.cuda.Stream(device=dev);torch.cuda.set_stream(stream);torch.cuda.synchronize(dev)
 select=np.linspace(0,n-1,64,dtype=np.int64);gold=(np.arange(1,9)[None,:]/8+loops*aa.reshape(n,8)[select].astype(np.float64)/64).astype(np.float32)
 def launch():assert fn(a.data_ptr(),out.data_ptr(),blocks,loops,stream.cuda_stream)==0
 def verify():
  torch.cuda.synchronize(dev);got=out.cpu().numpy();assert np.isfinite(got).all();assert np.array_equal(got.reshape(n,8)[select],gold)
  assert np.array_equal(a.cpu().numpy(),aa)
  assert bool((ag[:guard]==sentinel).all()and(ag[-guard:]==sentinel).all()and(og[:guard]==sentinel).all()and(og[-guard:]==sentinel).all())
  return hashlib.sha256(got.tobytes()).hexdigest()
 launch();before=verify()
 for _ in range(3):launch()
 torch.cuda.synchronize(dev);barrier.wait(timeout=90);events=[];walls=[];bounds=[]
 for trial in range(5):
  barrier.wait(timeout=90);e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);t=time.perf_counter();epoch=time.time();e0.record();launch();e1.record();e1.synchronize();events.append(e0.elapsed_time(e1)/1000);walls.append(time.perf_counter()-t);bounds.append([epoch,time.time()])
 after=verify();barrier.wait(timeout=90)
 q={'completed':True,'step':tag,'device':dev,'node':socket.gethostname(),'hardware':{'arch':props.gcnArchName,'CU':props.multi_processor_count,'VRAM':props.total_memory},'CPU_affinity':mask,'library_SHA256':b['library_SHA256'],'CPU_build_SHA256':sha(base/'CPU-ISA-qualified-r8804.json'),'source_SHA256':b['source_SHA256'],'assembly_SHA256':b['assembly_SHA256'],'blocks':blocks,'threads':threads,'independent_chains':8,'loops':loops,'FMA_MAC_counts_as_2':True,'operations_per_kernel':ops,'warmup_kernels':3,'samples':5,'event_seconds':events,'wall_seconds':walls,'event_TFLOPS_median':ops/statistics.median(events)/1e12,'wall_TFLOPS_median':ops/statistics.median(walls)/1e12,'trial_epoch_bounds':bounds,'full_output_finite':True,'all_guards_and_input_unchanged':True,'CPU_FP64_oracle_selected_outputs':len(select)*8,'selected_CPU_oracle_exact':True,'before_output_SHA256':before,'after_output_SHA256':after,'scope':'Achievedpacked-vector4xv_pk_fma_f32/8chainFP32reference; eventincludeskernelinput/output, notpure_scalar/physicalceiling/HBM/depack/GEMM/ModelTPS. CPUoracleonlyFP64;GPUFP32only.'}
 del a,out,ag,og;torch.cuda.synchronize(dev);torch.cuda.empty_cache();q['normal_tensor_cleanup']=True
 p=root.with_name(root.name+'-device'+str(dev)+'.json');p.write_text(json.dumps(q,indent=2))
if __name__=='__main__':
 assert torch.cuda.device_count()==8;assert not root.with_name(root.name+'.json').exists();ctx=mp.get_context('spawn');barrier=ctx.Barrier(8);mp.spawn(worker,args=(barrier,),nprocs=8,join=True)
 owners=[json.loads(root.with_name(root.name+'-device'+str(d)+'.json').read_text())for d in range(8)];assert all(o['completed']and o['normal_tensor_cleanup']for o in owners)
 root.with_name(root.name+'.json').write_text(json.dumps({'completed':True,'step':tag,'owners':owners,'hardware':'BW1000','scope':'Packed-vectorFP32arithmeticreferenceonly,notpure_scalar/physicalceilingorModelclaim'},indent=2));print('FP32_FMA_COMPLETE',tag,flush=True)
