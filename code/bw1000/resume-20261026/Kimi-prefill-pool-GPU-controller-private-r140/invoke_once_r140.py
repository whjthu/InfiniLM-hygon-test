import pathlib,json,hashlib,os,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-prefill-pool-controller-r140';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();go=json.loads((D/'GPU-GO-after-source-publication.json').read_bytes());assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==sha(D/'GPU-manifest-source.json')and len(go['Git_publication_SHA'])==40
record=r/'results/Kimi-prefill-pool-r140-launch.json';intent=r/'results/Kimi-prefill-pool-r140-intent.json';health=r/'results/Kimi-prefill-pool-r140-control.json';assert not record.exists()and not intent.exists()and not health.exists(),'existing unknownstate: read same actual handles, never reinvoke'
pin=json.loads((D/'controller-source-pins.json').read_bytes())
for rel,h in pin['files_SHA256'].items():assert sha(D/rel)==h,rel
for parent in ['910407','910379']:
 rows=subprocess.check_output(['sacct','-S','2026-10-25','-j',parent,'-n','-P','-o','JobIDRaw,JobName%110,State'],text=True,timeout=25).splitlines();assert not any('Kimi-prefill-pool-r140'in row for row in rows),'same label exists; inspect only'
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k,None)
prior=subprocess.check_output(['squeue','--steps','-j','910407','-h','-o','%i|%j|%N'],env=env,text=True,timeout=25).splitlines();assert len([x for x in prior if x.split('|')[0].split('.')[-1].isdigit()])<4
label='Kimi-prefill-pool-r140-keeper';cmd=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=2048M','-w','a14r2n20','-t','00:30:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(D/'keeper_r140.py')]
q={'manifest_SHA256':go['manifest_SHA256'],'Git_publication_SHA':go['Git_publication_SHA'],'CPU_label':label,'GPU_submitted_byhelper':False,'intentUTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with intent.open('xb')as f:f.write((json.dumps(q,indent=2)+'\n').encode())
fo=(r/'logs'/(label+'-client.log')).open('x');p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT,start_new_session=True);q['CPU_client_PID']=p.pid;record.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
