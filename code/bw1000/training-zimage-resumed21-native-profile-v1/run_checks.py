"""Qualify CUDA profiler then profile one real warmed/restored native update."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
c=json.loads((S/'CLAIM.json').read_bytes());b=json.loads((S/'BINDING.json').read_bytes());UUID=c['GPU_UUID'];PCI=c['GPU_PCI']
epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=float(os.environ['CONTRACT_GPU_DEADLINE']);anchor=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=1700
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original nativeprofile epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def proc(pid):
 f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
def idle():
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 selected=[[v.strip() for v in row.split(',')] for row in cards if UUID in row]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 assert len(selected)==1 and selected[0][1]==PCI and int(selected[0][2])<=16 and not any(UUID in row for row in apps)
 return dict(cards=selected,owned_compute_apps=[])
owner=proc(os.getpid());dump('GPU-owner.json',owner);phases=[];stage='source'
def child(label,args,limit):
 before=idle()
 with (W/(label+'.private.log')).open('xb') as log:
  p=subprocess.Popen(args,cwd=F,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT);owned=proc(p.pid);assert owned['PGID']==owner['PGID'];dump(label+'-owner.json',owned)
  rc=p.wait(timeout=min(limit,deadline-time.time(),remaining-(time.monotonic()-anchor)))
 row=dict(label=label,owner=owned,returncode=rc,before_GPU=before);phases.append(row);dump(label+'-returned.json',row);return row
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==374
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 stage='CUDA-profiler-qualification'
 row=child('profiler-qualification',[sys.executable,'-B','-u',str(F/'tools/hygon_zimage_profile_smoke.py'),'--output',str(W/'profiler-qualification')],min(180,epoch+180-time.time()))
 p=W/'profiler-qualification/profile-summary.json';smoke=json.loads(p.read_bytes()) if p.is_file() else None
 assert row['returncode']==0 and smoke and smoke['passed'] and smoke['CUDA_device_events']>0 and smoke['native_models_loaded']==0
 dump('profiler-qualification-after-GPU.json',idle());stage='native-profile'
 cp=Path(b['diagnostic_checkpoint20']['path']);assert cp.stat().st_size==b['diagnostic_checkpoint20']['bytes'] and sha(cp)==b['diagnostic_checkpoint20']['SHA256']
 args=[sys.executable,'-B','-u',str(F/'tools/hygon_zimage_native_record.py'),'--binding',str(S/'BINDING.json'),'--output',str(W/'native-profile'),
  '--mode','resume20','--checkpoint',str(cp),'--first-resumed-update-only','--warmup-before-restored-update','--profile-resumed-update','--epoch',str(epoch),'--deadline',str(deadline)]
 row=child('native-profile',args,deadline-time.time());p=W/'native-profile/result.json';native=json.loads(p.read_bytes()) if p.is_file() else None
 assert row['returncode']==0 and native and native['passed'] and native['completed_global_steps']==[21] and native['actual_optimizer_updates']==2
 assert native['diagnostic_warmup']['restored20_after_warmup'] and not native['first_update_capture_complete'] and native['profile_summary']['passed']
 profile=native['profile_summary'];raw=W/'native-profile/operator-profile-private/events.private.jsonl'
 assert raw.stat().st_size==profile['raw_event_bytes'] and sha(raw)==profile['raw_event_SHA256'] and profile['CUDA_device_events']>0 and profile['routes']
 finalpost=idle();dump('final-after-GPU.json',finalpost)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=374,phases=phases,
  profiler_qualification=smoke,native=native,native_result_SHA256=sha(p),actual_GPU_worker_processes=2,actual_native_model_loads=1,actual_native_optimizer_updates=2,
  profiled_native_updates=1,extra_warmup_updates=1,profile=profile,GPU_cards=1,wall_seconds=time.monotonic()-anchor,
  formal_performance=False,original_native25_resume5_acceptance=False,private_values_exported=False,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(stage=stage,error_type=type(error).__name__,traceback=traceback.format_exc()[-10000:],phases=phases))
 result=dict(passed=False,stage=stage,error_type=type(error).__name__,owner=owner,epoch=epoch,deadline=deadline,phases=phases,GPU_cards=1,wall_seconds=time.monotonic()-anchor,Model_GO=False)
dump('GPU-result.json',result);print(json.dumps(dict(passed=result['passed'],stage=stage,error_type=result.get('error_type'),GPU_cards=1)))
if not result['passed']:raise SystemExit(1)
