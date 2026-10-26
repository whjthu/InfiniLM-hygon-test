import common,os,json,subprocess,socket,time,datetime
r=common.ROOT;src=common.SRC;env,_=common.clean_env();p=r/'results/Kimi-PP4-cache-CPUclosure-r9206.json';assert not p.exists();q={'CPU_only':True,'GPU_submitted':False,'actualCPUstep':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'PID':os.getpid(),'groups':[]}
def save():q['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(p,q)
save();jobs=[]
for i,g in enumerate(common.RES['target_groups']):
 name='Kimi-PP4-cache-CPUclosure-r9206-g'+str(i);args=['srun','--jobid='+g['parent'],'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=none','--mem=1024M','-w',','.join(g['nodes']),'-t','00:06:00','--job-name='+name,'--export=ALL,GLOBAL_OFFSET_R2354='+str(g['offset'])+',GLOBAL_MODEL_MODE_R2354=shca,WAVE_INSIDE_R2323=Kimi-shard-PP4-cache-Model-r9206/runtime/preflight_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];out=r/'logs'/(name+'.out');err=r/'logs'/(name+'.err');fo=out.open('x');fe=err.open('x');z=subprocess.Popen(args,env=env,stdout=fo,stderr=fe);jobs.append((g,name,z,out,err,fo,fe));q['groups'].append({'name':name,'args':args,'client_PID':z.pid});save()
for i,(g,name,z,out,err,fo,fe)in enumerate(jobs):
 deadline=time.monotonic()+390
 while z.poll()is None:assert time.monotonic()<deadline;save();time.sleep(5)
 fo.close();fe.close();rows=[json.loads(s)for s in out.read_text().splitlines()if s.startswith('{')];v=q['groups'][i];v.update(client_returncode=z.returncode,packets=rows,stderr=err.read_text());save();assert z.returncode==0 and len(rows)==4 and {x['host']for x in rows}==set(g['nodes']) and all(x['all_source_modelview_DSO_object_closure']and x['affinity64']>=64 and not x['ldd_notfound']for x in rows),(v,z.returncode);hs={x['actual_step']for x in rows};assert len(hs)==1;h=hs.pop();v['actual_terminal']=common.terminal(g['parent'],name,h,env);assert v['actual_terminal'][2:4]==['COMPLETED','0:0'];save()
q.update(completed=True,passed=True);save();print(json.dumps(q))
