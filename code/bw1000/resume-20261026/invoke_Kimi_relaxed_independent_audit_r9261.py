import pathlib,json,subprocess,os,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-relaxed-hostlogits-r9258b';run='kimi-relaxed-A-r9258b-04e27169'
hp=r/'results'/('Kimi-relaxed-A-model-r9258b-'+run+'.json');w=json.loads(hp.read_bytes())
assert w['audit']['returncode']==1 and"KeyError: 'SLURM_JOB_ID'"in w['audit']['stderr']
assert w['post_KFD_passed']and all(v['terminal'][2:4]==['COMPLETED','0:0']for v in w['groups'])and w['draft']['terminal'][2:4]==['COMPLETED','0:0']
pin=json.loads((D/'launcher/model-source-pins.json').read_bytes());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p,h in pin['files'].items():assert sha(p.replace('/refactor',str(r),1))==h,p
out=r/'results'/('Kimi-relaxed-A-sequential-audit-r9258b-'+run+'.json');assert not out.exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))}
name='Kimi-relaxed-independent-audit-r9261'
a=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=4096M','-w','a14r2n20','-t','00:10:00','--job-name='+name,'--output='+str(r/'logs'/(name+'-%J.log')),'python3',str(D/'launcher/model_audit.py'),str(hp)]
record=r/'results/Kimi-relaxed-independent-audit-r9261-launch.json';assert not record.exists()
fo=(r/'logs'/(name+'-client.log')).open('x');z=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT,start_new_session=True)
q={'schema':1,'CPU_only':True,'GPU':False,'client_PID':z.pid,'args':a,'exact_original_audit_SHA256':sha(D/'launcher/model_audit.py'),'original_failure':{'step':'910407.419','returncode':1,'stderr':w['audit']['stderr']},'original_report_preserved':True,'all_original_checks_unchanged':True,'result':str(out),'new_performance_round':False,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
record.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
