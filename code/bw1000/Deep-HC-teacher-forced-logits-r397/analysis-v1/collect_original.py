PROBE="import pathlib,os,json,socket\nroot=pathlib.Path('/sys/class/kfd/kfd/proc');assert root.exists();pids=[int(p.name)for p in root.iterdir()if p.name.isdigit()];matches=[]\nfor p in pathlib.Path('/proc').iterdir():\n if not p.name.isdigit()or int(p.name)==os.getpid():continue\n try:\n  args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0));matched=[v for v in args if v.endswith(('/bootstrap_r206b.py','/worker_r206b.py','/modern_Deep_independent64_r9278.py','/client_chunk_pair.py'))]\n  if matched:matches.append({'PID':int(p.name),'executables':matched})\n except(OSError,PermissionError):pass\ns=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',43882));q={'node':socket.gethostname(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'KFD_available':True,'kfd_pids':pids,'owned_or_old_Deep_processes':matches,'HTTP43882_bindable':True,'GPU_imported':False};print(json.dumps(q));assert q['node']=='a02r4n01'and not pids and not matches\n"

import pathlib,json,hashlib,subprocess,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label='Deep-HC-teacher-forced-logits-r397-numeric-v1';GPU='910374.1325';CPU='910407.795';W=R/'results'/(label+'-watch.json');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
watch=json.loads(W.read_bytes());fields=['JobID','JobName','State','ExitCode','Elapsed','ElapsedRaw','NodeList'];text=subprocess.check_output(['sacct','-S','2026-11-02','-j',GPU+','+CPU,'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed,ElapsedRaw,NodeList%100'],text=True,timeout=25);rows=[dict(zip(fields,l.split('|')))for l in text.splitlines()];byid={x['JobID']:x for x in rows};terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
if not watch.get('completed')or any(v not in byid or byid[v]['State'].split()[0].rstrip('+')not in terminal for v in [GPU,CPU]):
 print(json.dumps({'terminal':False,'watch':watch,'accounting':rows}));raise SystemExit(0)
archive_path=R/'results/Deep397-1325-original-terminal-archive-R-only.json'
if archive_path.exists():
 a=json.loads(archive_path.read_bytes());assert a['actual_GPU']==GPU;print(json.dumps({'archive':a,'remote_archive_path':str(archive_path),'remote_archive_SHA256':sha(archive_path),'existing_original':True}));raise SystemExit(0)
I=R/'incoming/Deep-HC-teacher-forced-logits-r397';report=R/'results'/('Deep-HC-teacher-forced-logits-r397-'+GPU+'.json');q=json.loads(report.read_bytes())
legacy=q.get('legacy_load_receipt',str(R/'results'/('deep-c4-independent64-r9278-n01-'+GPU+'.json')));legacy=R/pathlib.Path(legacy).relative_to('/refactor')if legacy.startswith('/refactor/')else pathlib.Path(legacy)
paths={'report':report,'legacy':legacy,'watch':W,'teacher':I/'private/teacher-prefix.json','published_claim':I/'MODEL_EXECUTION_CLAIM-c2948702.json','runtime_GO':I/'execution-GO-c2948702.json','deployment':I/'deployment-c2948702.json','source_manifest':I/'source-v3/SOURCE_MANIFEST.json','owner_manifest':I/'source-v3/owner/SOURCE_MANIFEST.json','controller_manifest':I/'controller-v1/SOURCE_MANIFEST.json'}
files={}
for role,p in paths.items():
 b=p.read_bytes();files[role]={'path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode('utf-8')}
compile(PROBE,'Deep397_post_PID_probe','exec');assert chr(0)not in PROBE
args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a02r4n01','--gres=none','--mem=512M','-t','00:02:00','--job-name='+label+'-independent-post','python3','-c',PROBE]
p=subprocess.run(args,text=True,capture_output=True,timeout=160);post=[json.loads(l)for l in p.stdout.splitlines()if l.startswith('{')];assert p.returncode==0 and len(post)==1,(p.returncode,p.stdout,p.stderr)
a={'schema':1,'actual_GPU':GPU,'actual_CPU_keeper':CPU,'collected_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_accounting':byid[GPU],'keeper_accounting':byid[CPU],'files':files,'independent_post':post[0],'raw_publication':'R/L only; no full logits stored; teacher prefix and raw reports not for Git','original_watch_preserved':True}
log=R/'logs'/(label+'-'+GPU+'.log')
if log.exists():a['GPU_log']={'path':str(log),'bytes':log.stat().st_size,'SHA256':sha(log)}
with archive_path.open('x')as f:json.dump(a,f,indent=2)
print(json.dumps({'archive':a,'remote_archive_path':str(archive_path),'remote_archive_SHA256':sha(archive_path),'existing_original':False}))
