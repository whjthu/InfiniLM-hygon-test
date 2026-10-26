import common,os,json,subprocess,socket,time,datetime
r=common.ROOT;src=common.SRC;env,_=common.clean_env();hp=r/'results/Kimi-adaptive-Model-AB-CPUclosure-r9225.json';assert not hp.exists();q={'CPU_only':True,'GPU_submitted':False,'actualCPUstep':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'PID':os.getpid(),'groups':[]}
def save():q['heartbeatUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(hp,q)
save();jobs=[]
try:
 for i,g in enumerate(common.RES['groups']):
  common.verify_parent(g,env);name='Kimi-AB-CPU-r9225-'+str(i);args=['srun','--jobid='+g['parent'],'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=none','--mem=2048M','-w',','.join(g['nodes']),'-t','00:05:00','--job-name='+name,'--export=ALL,GLOBAL_OFFSET_R2354='+str(g['offset'])+',GLOBAL_MODEL_MODE_R2354=shca,WAVE_INSIDE_R2323=Kimi-adaptive-Model-AB-r9225/runtime-'+g['tag']+'/preflight_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];out=r/'logs'/(name+'.out');err=r/'logs'/(name+'.err');fo=out.open('x');fe=err.open('x');z=subprocess.Popen(args,env=env,stdout=fo,stderr=fe);jobs.append((g,name,z,out,err,fo,fe));q['groups'].append({'name':name,'args':args,'tag':g['tag'],'client_PID':z.pid});save()
 for i,(g,name,z,out,err,fo,fe)in enumerate(jobs):
  deadline=time.monotonic()+350
  while z.poll()is None:assert time.monotonic()<deadline;save();time.sleep(3)
  fo.close();fe.close();rows=[json.loads(line)for line in out.read_text().splitlines()if line.startswith('{')];v=q['groups'][i];v.update(returncode=z.returncode,rows=rows,stderr=err.read_text());save();assert z.returncode==0 and len(rows)==4 and {x['host']for x in rows}==set(g['nodes'])and all(x['passed']and not x['GPU']and not x['dlopen']and x['affinity64']>=64 for x in rows),v;steps={x['actual_step']for x in rows};assert len(steps)==1;v['terminal']=common.terminal(g['parent'],name,steps.pop(),env);assert v['terminal'][2:4]==['COMPLETED','0:0'];save()
 q.update(completed=True,passed=True);save()
except BaseException as e:q['error']=repr(e);save();raise
