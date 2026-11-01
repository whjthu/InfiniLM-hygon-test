import pathlib,json,subprocess,hashlib,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';mode='measure' if (r/'results/GLM341d-measure-launch.json').exists() else 'probe';q={'scope':'Component only; baseline Torch component chain; synthetic hidden; not Model quality or service performance'}
q['launch']=json.loads((r/('results/GLM341d-'+mode+'-launch.json')).read_text());env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_'))}
q['sacct']=subprocess.check_output(['sacct','-S','2026-10-24','-j','910371,910374','-n','-P','-o','JobID,JobName%80,State,ExitCode,NodeList'],env=env,text=True)
q['sacct']='\n'.join(s for s in q['sacct'].splitlines() if 'GLM341d-'+mode in s)
rows=[s.split('|') for s in q['sacct'].splitlines()];assert len(rows)==2 and all(s[2] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] for s in rows),rows
q['probes']=[json.loads(p.read_text()) for p in (r/'results').glob('GLM341d-map-*-rank*.json')]
q['results']={}
for f in (r/'results').glob('GLM341-*-rank*.json'):q['results'][str(f)]={'SHA256':hashlib.sha256(f.read_bytes()).hexdigest(),'raw':json.loads(f.read_text())}
q['logs']={str(f):{'SHA256':hashlib.sha256(f.read_bytes()).hexdigest(),'tail':f.read_text(errors='replace')[-5000:]} for f in (r/'logs').glob('GLM341d-*.log')}
q['post_KFD']=[]
for parent,node in [('910371','a02r3n16'),('910374','a02r4n07')]:
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM341d-post','python3',str(r/'incoming/verify_gpu_idle_r5738.py')]
 z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=135);assert z.returncode==0,(z.stdout,z.stderr)
 probes=[json.loads(s) for s in z.stdout.splitlines() if s.startswith('{')];assert len(probes)==1;q['post_KFD']+=probes
q['post_clean']=all(not s['kfd_pids'] for s in q['post_KFD']);q['normal_ranks']=sum(z['raw'].get('completed',False) and z['raw'].get('normal',False) for z in q['results'].values())
print(json.dumps(q))


