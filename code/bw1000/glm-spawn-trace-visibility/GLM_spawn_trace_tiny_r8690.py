"""Eight spawned Torch owners: profiler visibility only, no Model or CCL."""
import ctypes,json,os,pathlib,time
import torch,torch.multiprocessing as mp
R=pathlib.Path('/refactor')
def worker(rank):
 torch.set_num_threads(1);torch.cuda.set_device(rank)
 ctypes.CDLL('/opt/dtk/hip/lib/libgalaxyhip.so.5',mode=ctypes.RTLD_GLOBAL)
 api=ctypes.CDLL(None);api.hipProfilerStart.restype=ctypes.c_int;api.hipProfilerStop.restype=ctypes.c_int
 x=torch.arange(16,dtype=torch.float32,device='cuda');torch.cuda.synchronize(rank)
 start=time.time_ns();a=api.hipProfilerStart();assert a==0
 y=x.sum();torch.cuda.synchronize(rank)
 b=api.hipProfilerStop();end=time.time_ns();assert b==0
 value=float(y.item());assert value==120.
 del y,x;torch.cuda.synchronize(rank)
 q={'completed':True,'rank':rank,'pid':os.getpid(),'node':os.uname().nodename,'start_code':a,'stop_code':b,'sum':value,'window_realtime_ns':[start,end],'normal_Torch_owner_exit':True,'GPU_performance_claim':False,'no_Model_no_CCL':True}
 p=R/'results'/f'GLM-spawn-trace-r8690-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json'
 with p.open('x')as f:json.dump(q,f,indent=2)
if __name__=='__main__':mp.spawn(worker,nprocs=8,join=True)
