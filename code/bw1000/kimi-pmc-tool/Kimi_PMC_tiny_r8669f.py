"""One existing roofline torch.add pure-elementwise dispatch; PMC tool qualification only."""
import torch,numpy as np,json,pathlib,os,sys,time
torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
start=time.perf_counter();p=torch.cuda.get_device_properties(0);arch=getattr(p,'gcnArchName','');assert 'gfx936' in arch
# CPU init + H2D, no GPU fill/reduction/atomic kernel before/after selected add.
x=torch.from_numpy(np.full(1024*1024,2.0,dtype=np.float32)).cuda();y=torch.empty_like(x)
torch.cuda.synchronize();torch.add(x,x,out=y);torch.cuda.synchronize()
a=y.cpu().numpy();assert np.isfinite(a).all()and(a==4.0).all()
del x,y;torch.cuda.empty_cache();torch.cuda.synchronize()
out={'completed':True,'normal_Torch_cleanup':True,'device_count':1,'arch':arch,'name':p.name,'pid':os.getpid(),'selected_operation':'torch.add(x,x,out=y) once, noatomics; sameop as existingbw_roofline_probe.py','elements':1024*1024,'input_F32':2.0,'all_output_F32':4.0,'kernel_dispatch_expected':1,'seconds_including_initialization':time.perf_counter()-start,'Model_or_performance_claim':False,'PMC_counter_qualification':'PENDING_OFFLINE_ACTUAL_COUNTERS'}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out,indent=2))
