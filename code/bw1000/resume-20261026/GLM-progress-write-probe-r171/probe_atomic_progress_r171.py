"""CPU-only synthetic atomic-file write timing. Never reads model/progress/trace files."""
import pathlib,json,time,os,uuid,statistics,socket
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results').resolve();assert R.is_dir()
stem='GLM-CPU-progress-probe-r171-'+str(os.getpid())+'-'+uuid.uuid4().hex
tmp=R/(stem+'.tmp');target=R/(stem+'.json');assert tmp.parent==target.parent==R and not tmp.exists()and not target.exists()
times=[];written=[];replace_times=[];owned_tmp=False;owned_target=False;result={'completed':False,'CPU_only':True,'GPU':False,'Torch_imported':False,'Native_dlopen':False,'scope':'64 synthetic small JSON atomic writes in same shared results directory; no original files read/modified','node':socket.gethostname(),'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'results_directory':str(R),'samples_requested':64,'synthetic_payload_only':True,'reads_original_progress':False};start=time.monotonic()
def describe(xs):
 x=sorted(xs)
 def quant(p):
  f=(len(x)-1)*p;i=int(f);j=min(i+1,len(x)-1);return x[i]*(j-f)+x[j]*(f-i)if j!=i else x[i]
 return {'n':len(x),'min':x[0],'median':statistics.median(x),'mean':statistics.mean(x),'p95':quant(.95),'p99':quant(.99),'max':x[-1]}
try:
 for i in range(64):
  assert time.monotonic()-start<120,'bounded_CPU_probe'
  payload={'run':'CPU-SYNTHETIC-PROGRESS-NO-MODEL-000000','stage':0,'rank':0,'phase':'CPU-DECODE-PROBE','seq':i+1,'op':'DECODE','warm_active':False,'stall_guard_active':True,'UTC_epoch':0.0,'stall_limit_seconds':60,'actual_step':'CPU-only','native_calls':i}
  text=json.dumps(payload);tick=time.perf_counter()
  with tmp.open('x')as f:owned_tmp=True;f.write(text)
  mid=time.perf_counter();tmp.replace(target);owned_tmp=False;owned_target=True;end=time.perf_counter()
  times.append(1000*(end-tick));replace_times.append(1000*(end-mid));written.append(len(text.encode()))
 assert json.loads(target.read_bytes())['seq']==64
 result.update(completed=True,samples_completed=64,total_atomic_write_ms=describe(times),replace_only_ms=describe(replace_times),payload_bytes_min=min(written),payload_bytes_max=max(written),total_payload_bytes=sum(written),seconds=time.monotonic()-start,fsync_added=False,performance_claim='filesystem timing only; no causal model-speed claim')
except BaseException as error:result.update(error_class=type(error).__name__,samples_completed=len(times))
finally:
 cleaned=[]
 for path,owned in [(tmp,owned_tmp),(target,owned_target)]:
  assert path.parent.resolve()==R and path.name.startswith(stem)
  if owned and path.exists():path.unlink();cleaned.append(path.suffix)
 result['own_files_cleanup_passed']=not tmp.exists()and not target.exists();result['removed_own_file_count']=len(cleaned);result['recursive_delete']=False
print(json.dumps(result));assert result['completed']and result['own_files_cleanup_passed']
