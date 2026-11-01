import pathlib,json,subprocess,hashlib,os,sys,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-operator-search-r341/repair_v2';commit,mode=sys.argv[1:];assert len(commit)==40 and mode in ['probe','measure']
record=r/('results/GLM341c-'+mode+'-launch.json');assert not record.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((inc/'SOURCE_MANIFEST.json').read_text());assert all(sha(inc/n)==h for n,h in m['files_SHA256'].items())
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_')) and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
if mode=='measure':
 probes=[json.loads(p.read_text()) for p in (r/'results').glob('GLM341c-map-*-rank*.json')];assert len(probes)==11
 for node,n in [('a02r3n16',7),('a02r4n07',4)]:
  pp=[x for x in probes if x['node']==node];assert len(pp)==n and len({x['physical_PCI'] for x in pp})==n and {x['rank'] for x in pp}==set(range(n))
 assert all(x['normal'] and x['device_count']==1 and x['current_device']==0 for x in probes)
term={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def fresh(parent,node):
 rows=[s.split('|') for s in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%80,State,ExitCode,NodeList'],env=env,text=True).splitlines()]
 assert not any('GLM341c-'+mode in x[1] for x in rows)
 for x in rows:
  if x[0].startswith(parent+'.') and x[0].split('.')[-1].isdigit() and x[2].split()[0].rstrip('+') not in term:
   hosts=subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True).splitlines();assert node not in hosts,x
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM341c-idle','python3',str(r/'incoming/verify_gpu_idle_r5738.py')]
 z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=135);assert z.returncode==0,(z.stdout,z.stderr);q=[json.loads(s) for s in z.stdout.splitlines() if s.startswith('{')];assert len(q)==1 and q[0]['node']==node and not q[0]['kfd_pids'];return q[0]
q={'published_commit':commit,'mode':mode,'source_manifest_SHA256':sha(inc/'SOURCE_MANIFEST.json'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_submissions':[],'preflight':[],'prior_terminal_steps':['910371.613','910374.1231'],'prior_post_KFD_clean':True,'scope':'binding diagnosis then only eleven previously unexecuted variants; no kernel rebuild'}
for parent,node,variants in [('910371','a02r3n16',[1,2,3,4,5,6,7]),('910374','a02r4n07',[9,10,12,13])]:q['preflight'].append(fresh(parent,node))
record.write_text(json.dumps(q,indent=2))
for parent,node,variants in [('910371','a02r3n16',[1,2,3,4,5,6,7]),('910374','a02r4n07',[9,10,12,13])]:
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n'+str(len(variants)),'-c4','-w',node,'--gres=dcu:'+str(len(variants)),'--mem='+str(len(variants)*8192)+'M','-t','00:03:00' if mode=='probe' else '00:30:00','--job-name=GLM341c-'+mode+'-'+str(variants[0]),'--output='+str(r/'logs/GLM341c-'+mode+'-%J-rank%t.log'),'env','GLM341_VARIANTS='+json.dumps(variants),'WAVE_INSIDE_R2323=GLM-operator-search-r341/repair_v2/'+('probe_inside.sh' if mode=='probe' else 'GPU_inside.sh'),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with (r/('logs/GLM341c-'+mode+'-'+str(variants[0])+'-client.log')).open('x') as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q['GPU_submissions'].append({'PID':p.pid,'parent':parent,'node':node,'variants':variants,'argv':a});record.write_text(json.dumps(q,indent=2))
print(json.dumps(q))
