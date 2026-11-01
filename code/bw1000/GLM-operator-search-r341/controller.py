"""Login-side single bounded batch submitter. Requires root-published source commit."""
import pathlib,json,subprocess,hashlib,os,sys,time,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-operator-search-r341';record=r/'results/GLM341-batch-launch.json'
assert len(sys.argv)==2 and len(sys.argv[1])==40
assert not record.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((inc/'SOURCE_MANIFEST.json').read_text());assert all(sha(inc/n)==h for n,h in m['files_SHA256'].items())
build=json.loads((r/'candidates/GLM-operator-search-r341/CPU_RESULT.json').read_text());assert build['returncode']==0
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_')) and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}
term={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def fresh(parent,node):
 rows=[s.split('|') for s in subprocess.check_output(['sacct','-S','2026-11-01','-j',parent,'-n','-P','-o','JobID,JobName%80,State,ExitCode,NodeList'],env=env,text=True).splitlines()]
 assert not any('GLM341-GPU' in x[1] for x in rows)
 for x in rows:
  if x[0].startswith(parent+'.') and x[0].split('.')[-1].isdigit() and x[2].split()[0].rstrip('+') not in term:
   hosts=subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True).splitlines();assert node not in hosts,x
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM341-idle','python3',str(r/'incoming/verify_gpu_idle_r5738.py')]
 z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=135);assert z.returncode==0,(z.stdout,z.stderr)
 q=[json.loads(s) for s in z.stdout.splitlines() if s.startswith('{')];assert len(q)==1 and q[0]['node']==node and not q[0]['kfd_pids'];return q[0]
q={'published_commit':sys.argv[1],'source_manifest_SHA256':sha(inc/'SOURCE_MANIFEST.json'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_submissions':[],'preflight':[],'CPU_library_SHA256':build['library_SHA256'],'scope':'16 independent one-card component candidates, not TP model'}
for parent,node,offset in [('910371','a02r3n16',0),('910374','a02r4n07',8)]:q['preflight'].append(fresh(parent,node))
record.write_text(json.dumps(q,indent=2))
for parent,node,offset in [('910371','a02r3n16',0),('910374','a02r4n07',8)]:
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n8','-c4','-w',node,'--gres=dcu:8','--mem=65536M','-t','00:30:00','--job-name=GLM341-GPU-'+str(offset),'--output='+str(r/'logs/GLM341-GPU-%J-rank%t.log'),'env','GLM341_VARIANT_OFFSET='+str(offset),'WAVE_INSIDE_R2323=GLM-operator-search-r341/GPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with (r/('logs/GLM341-GPU-'+str(offset)+'-client.log')).open('x') as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q['GPU_submissions'].append({'PID':p.pid,'parent':parent,'node':node,'variants':list(range(offset,offset+8)),'argv':a});record.write_text(json.dumps(q,indent=2))
print(json.dumps(q))
