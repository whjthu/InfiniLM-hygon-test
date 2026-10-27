import pathlib,json,hashlib,subprocess,os,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-ctx2k-Model-r127';L=D/'launcher';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();pin=json.loads((L/'model-source-pins.json').read_bytes());go=json.loads((L/'GPU-GO-after-source-publication.json').read_bytes());assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==pin['manifest_SHA256']and len(go['Git_publication_SHA'])==40;context=json.loads((L/'run-context-public.json').read_bytes());run=context['run'];label='Kimi-ctx2k-phase-keeper-'+run
record=r/'results/Kimi-ctx2k-phase-r127-launch.json';intent=r/'results/Kimi-ctx2k-phase-r127-intent.json';assert not record.exists()and not intent.exists(),'existing unknownstate: inspect same label/handles, never resubmit'
for p,h in pin['files'].items():assert sha(pathlib.Path(p.replace('/refactor',str(r),1)))==h,p
for parent in ['910406','910379','910371']:
 rows=subprocess.check_output(['sacct','-S','2026-10-25','-j',parent,'-n','-P','-o','JobIDRaw,JobName%110,State,ExitCode'],text=True,timeout=25).splitlines();assert not any(label in row or run in row for row in rows),'existing actual same run: never resubmit'
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k,None)
prior=subprocess.check_output(['squeue','--steps','-j','910406','-h','-o','%i|%j|%N'],text=True,env=env,timeout=25).splitlines();numeric=[x for x in prior if x.split('|')[0].split('.')[-1].isdigit()];assert len(numeric)<4,numeric
cmd=['srun','--jobid=910406','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=4096M','-w','a14r2n19','-t','02:10:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(L/'preflight_then_run.py')]
q={'run':run,'label':label,'GPU_submitted_by_this_helper':False,'manifest_SHA256':pin['manifest_SHA256'],'source_publication_SHA':go['Git_publication_SHA'],'intent_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with intent.open('xb')as f:f.write((json.dumps(q,indent=2)+'\n').encode())
f=(r/'logs'/(label+'-client.log')).open('x');p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);q['CPU_client_PID']=p.pid;record.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
