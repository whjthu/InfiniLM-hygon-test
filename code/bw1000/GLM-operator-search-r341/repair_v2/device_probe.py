import torch,ctypes,os,pathlib,json,gc
rank=int(os.environ['SLURM_LOCALID']);variants=json.loads(os.environ['GLM341_VARIANTS']);assert 0<=rank<len(variants);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
x=torch.arange(16,device='cuda');assert int(x.sum().item())==120
lib=ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');lib.cudaDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];bus=ctypes.create_string_buffer(64);assert lib.cudaDeviceGetPCIBusId(bus,64,0)==0
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];q={'actual_step':step,'node':os.uname().nodename,'rank':rank,'physical_PCI':bus.value.decode(),'device_count':1,'current_device':torch.cuda.current_device(),'variant':variants[rank],'visible_env':{k:os.environ.get(k) for k in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']},'weight_load':False}
del x;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();q['normal']=True
p=pathlib.Path('/refactor/results')/('GLM341c-map-'+step+'-rank'+str(rank)+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
