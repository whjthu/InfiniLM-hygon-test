import common,os,json,subprocess,pathlib,time,datetime,socket,traceback
r=common.ROOT;src=common.SRC;health=r/'results/Kimi-PP4-map-r9154.json';assert not health.exists();q={'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_hostname':socket.gethostname(),'PID':os.getpid(),'completed':False,'GPU_submitted':False,'groups':[]};q['run']='pp4-r9154-'+os.urandom(8).hex();q['nonce']=os.urandom(16).hex();q['base_port']=19000+int(q['nonce'][:6],16)%500;env,removed=common.clean_env();q['sanitized_removed_names']=removed
def save():q['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(health,q)
save()
try:
 jobs=[]
 for k,g in enumerate(common.RES['target_groups']):
  name='Kimi-PP4-map-r9154-g'+str(k);q['groups'].append({'group':k,'parent':g['parent'],'nodes':g['nodes'],'name':name,'parent_validation':common.verify_parent(g,env)});args=['srun','--jobid='+g['parent'],'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=none','--mem=512M','-w',','.join(g['nodes']),'-t','00:03:00','--job-name='+name,'--export=ALL,GLOBAL_RUN_R2354='+q['run']+',KIMI_ENGINE_NONCE='+q['nonce']+',PP4_GROUP_R9154='+str(k)+',GLOBAL_OFFSET_R2354='+str(g['offset'])+',WAVE_INSIDE_R2323=Kimi-PP4-crossalloc-r9154/map_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];out=r/'logs'/(name+'.stdout');err=r/'logs'/(name+'.stderr');fo=out.open('x');fe=err.open('x');p=subprocess.Popen(args,env=env,stdout=fo,stderr=fe);jobs.append((k,p,fo,fe,out,err));q['groups'][k].update(args=args,client_PID=p.pid);save()
 for k,p,fo,fe,out,err in jobs:
  deadline=time.monotonic()+230
  while p.poll()is None:
   assert time.monotonic()<deadline;save();time.sleep(5)
  rc=p.returncode;fo.close();fe.close();lines=out.read_text().splitlines();rows=[json.loads(s)for s in lines if s.startswith('{')];v=q['groups'][k];v.update(client_returncode=rc,probe_rows=rows,stderr=err.read_text());save();assert rc==0 and len(rows)==4 and all(x['passed']for x in rows),(rc,rows,v['stderr']);handles={x['actual_step']for x in rows};assert len(handles)==1;handle=handles.pop();v['actual_handle']=handle;assert handle.startswith(v['parent']+'.')and handle.split('.')[1].isdigit();actual=common.terminal(v['parent'],v['name'],handle,env);v['actual_terminal']=actual;assert actual[2:4]==['COMPLETED','0:0'];save()
 rows=[x for v in q['groups']for x in v['probe_rows']];assert sorted(x['global_node']for x in rows)==list(range(8));assert sorted(y for x in rows for y in x['world_GPU_ranks'])==list(range(64));assert all(x['affinity_count']>=64 for x in rows);q['verified_global_mapping']=sorted(rows,key=lambda x:x['global_node']);q['head_ib0_IPv4']=next(x['ib0_IPv4']for x in rows if x['global_node']==0);save()
 # Fresh KFD only after both authoritative CPU mapping steps are COMPLETED0.
 for k,g in enumerate(common.RES['target_groups']):
  name='Kimi-PP4-fresh-r9154-g'+str(k);a=['srun','--jobid='+g['parent'],'--overlap','--exact','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(g['nodes']),'-t','00:02:00','--job-name='+name,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=160);rs=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];q['groups'][k]['fresh_KFD']={'returncode':z.returncode,'rows':rs,'stderr':z.stderr};save();assert z.returncode==0 and len(rs)==4 and all(x['passed']and not x['kfd_pids']for x in rs);hs={x['step']for x in rs};assert len(hs)==1;handle=hs.pop();row=common.terminal(g['parent'],name,handle,env);assert row[2:4]==['COMPLETED','0:0'];q['groups'][k]['fresh_actual_terminal']=row;save()
 q.update(completed=True,passed=True,GPU_submitted=False,CPU_mapping_and_fresh_only=True);save()
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),completed=True,passed=False);save();raise
print(json.dumps(q))
