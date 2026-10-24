import sys,pathlib,json,hashlib,subprocess,time,uuid,datetime
assert len(sys.argv)==2
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'GLM-comm-only-r8783-launch.json';assert not out.exists()
m=r/'incoming/GLM-comm-only-manifest-r8783.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()==sys.argv[1];manifest=json.loads(m.read_text())
for n,h in manifest['source_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
for n,h in manifest['DSO_SHA256'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
nodes=['a02r4n12','a02r4n13','a02r4n14']
allocation=subprocess.check_output(['scontrol','show','job','910376','-o'],text=True,timeout=25)
snapshot=subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%100,State,NodeList%100'],text=True,timeout=25)
for line in snapshot.splitlines():
    v=line.split('|')
    if '.' in v[0] and v[0].split('.')[-1].isdigit() and v[2] in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:
        hosts=subprocess.check_output(['scontrol','show','hostnames',v[3]],text=True,timeout=20).splitlines()
        assert not set(nodes)&set(hosts),('busyfail_NO_GPU',v)
idleargs=['srun','--jobid=910376','--overlap','--exact','-N3','-n3','--ntasks-per-node=1','-c1','-w',','.join(nodes),'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-comm-only-idle-r8783','python3',str(r/'incoming/verify_gpu_idle_r5738.py')]
p=subprocess.run(idleargs,text=True,capture_output=True,timeout=100);assert p.returncode==0,(p.stdout,p.stderr)
idle=[json.loads(v) for v in p.stdout.splitlines() if v.startswith('{')]
assert sorted(v['node'] for v in idle)==nodes and all(v['kfd_pids']==[] for v in idle)
run='comm-only-8783-'+uuid.uuid4().hex[:20]
intent={'state':'launch_intent_no_GPU','run':run,'manifest_SHA256':sys.argv[1],'fresh_idle':idle,'fresh_sacct':snapshot,'allocation':allocation}
with out.open('x') as f:json.dump(intent,f,indent=2)
with (r/'logs/GLM-comm-only-watch-r8783.log').open('x') as log:
    watch=subprocess.Popen(['python3',str(r/'incoming/watch_GLM_comm_only_r8783.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ready=None;deadline=time.monotonic()+45;watchpath=r/'results/GLM-comm-only-watch-r8783.json'
while time.monotonic()<deadline and watch.poll() is None:
    if watchpath.exists():
        state=json.loads(watchpath.read_text())
        if state['state']=='watching' and state['run']==run and state['manifest_SHA256']==sys.argv[1]:ready=state;break
    time.sleep(.2)
assert ready is not None and watch.poll() is None,'watch_not_ready_NO_GPU'
args=['srun','--jobid=910376','--overlap','--exact','--kill-on-bad-exit=1','-N3','-n3','--ntasks-per-node=1','-c64','-w',','.join(nodes),'--gres=dcu:8','--mem=240000M','-t','00:05:00','--job-name=GLM-comm-only-r8783','--output='+str(r/'logs/GLM-comm-only-r8783-%J-%t.log'),'env','GLM_RUN_R5040='+run,'INFINILM_OWNER_CPU_WIDTH_R2392=4','WAVE_INSIDE_R2323=GLM_comm_only_r8783_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with (r/'logs/GLM-comm-only-r8783-controller.log').open('x') as log:
    controller=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
intent.update(state='submitted_unique_constructor',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=controller.pid,watch_pid=watch.pid,watch_ready=ready,args=args)
out.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
