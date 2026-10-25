import pathlib,json,hashlib,os,time,resource,gc,traceback,signal
signal.alarm(145)
start=time.perf_counter();import torch
from mapped_layer_weights_r9125 import mapped_layer_weights
torch.set_num_threads(1)
r=pathlib.Path('/refactor');node=os.environ['SLURM_PROCID'];producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('Kimi-MoE-lazyCPU-r9125-'+producer+'-node'+node+'.json');assert not out.exists()
def snap():
 p=pathlib.Path('/proc/self');s=(p/'status').read_text();io=(p/'io').read_text();u=resource.getrusage(resource.RUSAGE_SELF);return {'VmRSS_kB':int(next(v.split()[1]for v in s.splitlines()if v.startswith('VmRSS:'))),'io':{a:int(b)for a,b in(v.split(':')for v in io.splitlines())},'minor_faults':u.ru_minflt,'major_faults':u.ru_majflt,'maxRSS_kB':u.ru_maxrss,'CPU_seconds':u.ru_utime+u.ru_stime}
report={'producer':producer,'node':os.uname().nodename,'CPU_only':True,'GPU_initialized_before':torch.cuda.is_initialized(),'import_seconds':time.perf_counter()-start,'threading':{'intraop':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'affinity':sorted(os.sched_getaffinity(0))},'passed':False};before=snap();model=pathlib.Path('/shared/Kimi-K3');expected=json.loads(pathlib.Path(__file__).with_name('expected-headers-r9125.json').read_text());prefix='language_model.model.layers.1.block_sparse_moe.';tick=time.perf_counter()
try:
 with mapped_layer_weights(model,prefix,expected)as(held,desc,proof):
  report.update(map_metadata_seconds=time.perf_counter()-tick,after_metadata=snap(),proof=proof)
  mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];samples=[]
  for i in sorted(set([0,len(held)-1,*range(0,len(held),896)])):
   v=held[i];key=prefix+desc[i]['name'];path=model/mapping[key]
   with path.open('rb')as f:n=int.from_bytes(f.read(8),'little');h=json.loads(f.read(n));f.seek(8+n+h[key]['data_offsets'][0]);gold=f.read(1)[0]
   actual=int(v.view(torch.uint8).reshape(-1)[0]);assert actual==gold;samples.append({'key':key,'first_byte_equal':True})
  report['sample_readonly_checks']=samples;report['after_bounded_samples']=snap();report['passed']=proof['tensor_count']==5384 and not torch.cuda.is_initialized();report['current_full_tensor_SHA_verified']=False
 gc.collect();report['after_release']=snap();report['GPU_initialized_after']=torch.cuda.is_initialized();report['elapsed_total_seconds']=time.perf_counter()-start;report['map_plus_bounded_samples_seconds']=time.perf_counter()-tick;report['before_map']=before;report['interpretation']='Warm/current cache after654, not cold A/B and not proven speedup vs old41/540. Metadata mapping plus boundedbyte samples only; no nativeModule/GPUload.'
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(report,indent=2));print(json.dumps({'producer':producer,'node':report['node'],'passed':report['passed'],'path':str(out),'SHA256':hashlib.sha256(out.read_bytes()).hexdigest(),'GPU_initialized':report.get('GPU_initialized_after')}));assert report['passed']
