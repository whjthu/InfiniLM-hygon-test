"""One finite isolated microbenchmark; ready watcher always observes cleanup."""
import pathlib,json,hashlib,subprocess,time,datetime,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';m=json.loads((r/'incoming/hardware-dumma-manifest-r8405.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in m['sources'].items():assert sha(r/'incoming'/n)==h,n
lib=r/'candidates/hardware-dumma-r8403/libhardware_dumma_r8403.so';assert sha(lib)==m['library_SHA256']
assert sha(r/'candidates/hardware-dumma-r8403/CPU-build.json')==m['CPU_build_SHA256']
intent=r/'results/hardware-dumma-GPU-r8405-intent.json';watch=r/'results/hardware-dumma-GPU-r8405-watch.json';name='hardware-dumma-GPU-r8405'
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def accounting():
 a=['sacct','-S','2026-10-23','-j','910374','-n','-P','-o','JobID,JobName%90,State%40,ExitCode,Elapsed,NodeList%80']
 return[l.split('|')for l in subprocess.check_output(a,text=True,timeout=40).splitlines()]
def idle(label):
 a=['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n04','--gres=none','--mem=512M','-t','00:02:00','--job-name='+label,
    'python3',str(r/'incoming/probe_Deep_idle_r4847.py')]
 z=subprocess.run(a,capture_output=True,text=True,timeout=155);assert z.returncode==0,(z.stdout,z.stderr)
 q=json.loads(next(s for s in z.stdout.splitlines()if s.startswith('{')));assert q['passed']and q['node']=='a02r4n04';return q
if len(sys.argv)>1:
 assert sys.argv[1]=='watch'and intent.exists()and not watch.exists()
 w=dict(ready=True,utc=now(),actual_handle=None,terminal=None,postKFD_passed=False,observation_errors=[]);write(watch,w)
 while not w['terminal']:
  try:
   rows=[v for v in accounting()if v[1]==name];assert len(rows)<=1
   if rows:
    row=rows[0];assert w['actual_handle']in(None,row[0]);w['actual_handle']=row[0];w['last_observation']=row
    if row[2].split()[0].split('+')[0]in terminal:w['terminal']=row
  except BaseException as exc:w['observation_errors'].append(dict(utc=now(),error=repr(exc)))
  w['utc']=now();write(watch,w)
  if not w['terminal']:time.sleep(15)
 try:w['postKFD']=idle('hardware-dumma-postidle-r8405');w['postKFD_passed']=True
 except BaseException as exc:w['postKFD_error']=repr(exc)
 w['utc']=now();write(watch,w);print(json.dumps(w));raise SystemExit
assert not intent.exists()and not watch.exists()
for v in accounting():
 if v[0].startswith('910374.')and v[0].split('.')[-1].isdigit()and v[2].split()[0].split('+')[0]not in terminal:
  assert'a02r4n04'not in subprocess.check_output(['scontrol','show','hostnames',v[5]],text=True,timeout=20).splitlines(),v
pre=idle('hardware-dumma-before-r8405')
d=dict(utc=now(),state='intent',no_resubmit=True,CPU_build_SHA256=m['CPU_build_SHA256'],library_SHA256=m['library_SHA256'],fresh_idle=pre)
with intent.open('x')as f:json.dump(d,f,indent=2)
with(r/'logs/hardware-dumma-r8405-watcher.log').open('x')as f:w=subprocess.Popen([sys.executable,__file__,'watch'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
end=time.monotonic()+30
while not watch.exists():assert time.monotonic()<end and w.poll()is None;time.sleep(.1)
ready=json.loads(watch.read_text());assert ready['ready']and ready['actual_handle']is None and ready['terminal']is None and w.poll()is None
a=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w','a02r4n04','--gres=dcu:8','--mem=16000M','-t','00:12:00','--job-name='+name,
 '--output='+str(r/'logs/hardware-dumma-GPU-r8405-%J.log'),'env','WAVE_INSIDE_R2323=hardware_dumma_GPU_r8405_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
assert w.poll()is None
with(r/'logs/hardware-dumma-r8405-controller.log').open('x')as f:p=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d.update(state='submitted_follow_samehandle',srun_PID=p.pid,watcher_PID=w.pid,ready_before_numeric=True,args=a);write(intent,d)
print(json.dumps(dict(utc=now(),srun_PID=p.pid,watcher_PID=w.pid,ready_before_numeric=True,intent=str(intent))))
