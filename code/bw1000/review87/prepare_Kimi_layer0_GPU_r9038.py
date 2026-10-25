import pathlib
b=pathlib.Path('Kimi-layer0-TP16-replay-private-r9038');p=b/'layer0_driver.py';s=p.read_text();needle='torch.cuda.set_device(device);ctx=lib.layer0_create_r9038';assert s.count(needle)==1;s=s.replace(needle,"affinity=sorted(os.sched_getaffinity(0));assert len(affinity)>=32;chosen=affinity[device*4:device*4+4];os.sched_setaffinity(threading.get_native_id(),chosen);own['owner_affinity']=sorted(os.sched_getaffinity(0));assert own['owner_affinity']==chosen;torch.cuda.set_device(device);ctx=lib.layer0_create_r9038");compile(s,'driver','exec');p.write_bytes(s.encode())
controller=r'''import pathlib,json,hashlib,subprocess,datetime,time,traceback
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';tag='Kimi-layer0-real-r9038';v=json.loads((r/(tag+'-intent.json')).read_text());out=r/'results'/(tag+'-terminal.json');assert not out.exists();m=json.loads((i/'Kimi-layer0-manifest-r9038.json').read_text());q={'state':'watchready','watch_ready_before_numeric':True,'numeric_attempts':0,'post_attempts':0,'exceptions':[],'Model_qualified':False}
def save():p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(out)
save()
try:
 with(r/'logs'/(tag+'-controller.log')).open('x')as f:
  p=subprocess.Popen(v['args'],stdout=f,stderr=subprocess.STDOUT);q.update(state='numeric_running',numeric_attempts=1,numeric_pid=p.pid);save();q['numeric_returncode']=p.wait(timeout=650);save()
 for _ in range(60):
  snap=subprocess.check_output(['sacct','-S','2026-10-25','-j','910375','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=20);rows=[s.split('|')for s in snap.splitlines()if len(s.split('|'))>5 and s.split('|')[1]==tag];assert len(rows)<=1;q['Slurm_rows']=rows
  if rows:
   if 'bound_handle'in q:assert q['bound_handle']==rows[0][0]
   else:q['bound_handle']=rows[0][0]
   if rows[0][2].split()[0].rstrip('+')not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:q['actual_terminal']=True;break
  if not rows and q['numeric_returncode']!=0:break
  save();time.sleep(2)
 assert q.get('actual_terminal');q['receipts']=[]
 for node in range(2):
  path=r/'results'/f'{tag}-{q["bound_handle"]}-node{node}.json';d=json.loads(path.read_text());q['receipts'].append({'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':d['status'],'normal_close':d.get('normal_close'),'native_calls_thisnode':d.get('native_calls_thisnode')});assert d['status']=='LAYER0_REPLAY_COMPONENT_DIAGNOSTIC_COMPLETE'and d['normal_close']and d['native_calls_thisnode']==472;assert all(z['normal_close']and 'error'not in z for z in d['workers']);assert all(d[k]for k in ['finite','canary','host_input_readonly','trace_finite'])
  for suffix,h in m['runtime_dependencies_SHA256'].items():assert d['actual_runtime_dependencies_SHA256'].get(suffix)==h,(suffix,d['actual_runtime_dependencies_SHA256'])
 q['component_audit_pass']=q['numeric_returncode']==0 and rows[0][2:4]==['COMPLETED','0:0'];save()
except BaseException as e:q['exceptions'].append({'phase':'numeric_or_audit','error':repr(e),'traceback':traceback.format_exc()});save()
finally:
 try:
  q['post_attempts']=1;save();args=['srun','--jobid=910375','--overlap','--exact','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(m['nodes']),'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-layer0-post-r9038','python3',str(i/'Kimi_Q0_idle_r8066.py')];p=subprocess.run(args,text=True,capture_output=True,timeout=160);q['post']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};records=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];q['post_pass']=p.returncode==0 and len(records)==2 and {x['node']for x in records}==set(m['nodes'])and all(x['passed']and not x['kfd_pids']for x in records)
 except BaseException as e:q['exceptions'].append({'phase':'post','error':repr(e)})
 q.update(state='terminal',completed=True,component_diagnostic_complete=bool(q.get('component_audit_pass')and q.get('post_pass')),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(q))
'''
launcher=r'''import pathlib,json,hashlib,subprocess,datetime,os
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';tag='Kimi-layer0-real-r9038';m=json.loads((i/'Kimi-layer0-manifest-r9038.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();nodes=m['nodes'];assert nodes==['a02r4n08','a02r4n09'];assert m['root_continuous_GO']and m['TP']==16 and m['PP']==1
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
for n,h in m['existing_dependency_SHA256'].items():assert sha(i/n)==h,n
assert sha(r/m['native'].removeprefix('/refactor/'))==m['native_SHA256'];allocation=subprocess.check_output(['scontrol','show','job','910375','-o'],text=True,timeout=20);assert 'JobState=RUNNING'in allocation;pool=subprocess.check_output(['scontrol','show','hostnames',next(a.split('=',1)[1]for a in allocation.split()if a.startswith('NodeList='))],text=True,timeout=20).splitlines();assert set(nodes)<=set(pool)
snapshot=subprocess.check_output(['sacct','-S','2026-10-25','-j','910375','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=20)
for line in snapshot.splitlines():
 f=line.split('|')
 if len(f)>5 and '.'in f[0]and f[0].split('.')[-1].isdigit()and f[2].split()[0].rstrip('+')in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:
  occupied=subprocess.check_output(['scontrol','show','hostnames',f[5]],text=True,timeout=20).splitlines();assert not set(nodes)&set(occupied),('node busy',f)
intent=r/(tag+'-intent.json');assert not intent.exists();idle=subprocess.run(['srun','--jobid=910375','--overlap','--exact','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(nodes),'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-layer0-idle-r9038','python3',str(i/'Kimi_Q0_idle_r8066.py')],capture_output=True,text=True,timeout=160);records=[json.loads(x)for x in idle.stdout.splitlines()if x.startswith('{')];assert idle.returncode==0 and len(records)==2 and {x['node']for x in records}==set(nodes)and all(x['passed']and not x['kfd_pids']for x in records),(idle.stdout,idle.stderr);head=next(x['ib0_ipv4'][0]for x in records if x['node']==nodes[0]);nonce=os.urandom(16).hex();port=19000+int(nonce[:5],16)%500;wave='Kimi_layer0_GPU_r9038_inside.sh';assert wave==m['WAVE_INSIDE']and sha(i/wave)==m['sources_SHA256'][wave]
env='ALL,KIMI_USE_RAMCACHE_R3679=0,KIMI_REDUCE_POLICY=0,INFINILM_OWNER_CPU_WIDTH_R2392=4,INFINILM_DENSE_PREFILL_F32_R2475=1,INFINILM_MXFP4_TILE_TP_R2559=0,GLOBAL_OFFSET_R2354=0,GLOBAL_RUN_R2354=layer0-r9038,GLOBAL_PORT_R2354='+str(port)+',KIMI_ENGINE_HEAD='+head+',KIMI_ENGINE_NONCE='+nonce+',KIMI_TP_NODES=2,KIMI_PP_SIZE=1,GLOBAL_MODEL_MODE_R2354=shca,WAVE_INSIDE_R2323='+wave
args=['srun','--export='+env,'--jobid=910375','--overlap','--exact','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(nodes),'--gres=dcu:8','--mem=128000M','-t','00:10:00','--job-name='+tag,'--output='+str(r/'logs'/(tag+'-%J-%t.log')),'bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe'];v={'nodes':nodes,'args':args,'fresh_idle':records,'manifest_SHA256':sha(i/'Kimi-layer0-manifest-r9038.json'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with intent.open('x')as f:json.dump(v,f,indent=2)
with(r/'logs'/(tag+'-watch-launch.log')).open('x')as f:p=subprocess.Popen(['python3',str(i/'controller_Kimi_layer0_r9038.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'controller_pid':p.pid,'watch':str(r/'results'/(tag+'-terminal.json')),'intent':v,'numeric_attempts_max':1,'post_attempts_max':1}))
'''
for n,s in [('controller_gpu.py',controller),('launch_gpu.py',launcher)]:compile(s,n,'exec');(b/n).write_bytes(s.encode())
inside=pathlib.Path('Kimi-crossover-Model-private-r9026/Kimi_crossover_Model_r9026_inside.sh').read_text();needle='export INFINILM_KIMI_TARGET_TAPS_DIR_R9022="/refactor/results/Kimi-crossover-private-r9026-${SLURM_JOB_ID}.${SLURM_STEP_ID}"\ntimeout --kill-after=30s 2640 python -u /refactor/incoming/modern_Kimi_crossover_Model_r9026.py';assert inside.count(needle)==1;inside=inside.replace(needle,'unset INFINILM_KIMI_TARGET_TAPS_DIR_R9022 INFINILM_KIMI_KDA_SERIAL_R9027\npython /refactor/incoming/Kimi_layer0_driver_r9038.py --cpu-contract /refactor/incoming/Kimi-layer0-manifest-r9038.json\ntimeout --kill-after=15s 540 python -u /refactor/incoming/Kimi_layer0_driver_r9038.py /refactor/incoming/Kimi-layer0-manifest-r9038.json');(b/'GPU_inside.sh').write_bytes(inside.encode());print('original675env/two-node argv/one-shot watch sources ready')
