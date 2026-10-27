"""One owned M15 GPU step only after explicit root GO and published source pins."""
import pathlib,json,subprocess,os,time,datetime,hashlib,socket,sys,traceback,argparse,shutil,re
ap=argparse.ArgumentParser();ap.add_argument('--go-file',required=True);ap.add_argument('--manifest-sha',required=True);opt=ap.parse_args()
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');inc=r/'incoming/GLM-M15-offset8-r127';source=r/'incoming/GLM-M15-component-r121b';label='GLM-M15-offset8-r127';parent='910376';node='a02r4n12';keeper=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');assert keeper.startswith('910407.')and keeper.split('.')[-1].isdigit()and socket.gethostname()=='a14r2n20'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=inc/'manifest-r127.json';assert sha(mp)==opt.manifest_sha;m=json.loads(mp.read_bytes());go_path=pathlib.Path(opt.go_file);go=json.loads(go_path.read_bytes());assert go['GPU_authorized']is True and go['source_published']is True and go['controller_manifest_SHA256']==opt.manifest_sha and re.fullmatch('[0-9a-f]{40}',go['publication_commit']);assert go['publication_branch']=='bw1000-experiments';assert re.fullmatch('[0-9a-f]{64}',go['CPU_receipt_SHA256'])
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
rp=r/(label+'-launch.json');wp=r/(label+'-watch.json');term={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'};assert not rp.exists()and not wp.exists();proc=None;known=None;terminal=None;post_attempted=False;w={'qualified':False,'numeric_positive':False,'actual_handle':None};d={'schema':1,'label':label,'restarts':0,'publication_commit':go['publication_commit'],'root_GO_SHA256':sha(go_path),'controller_manifest_SHA256':sha(mp),'CPU_receipt_SHA256':go['CPU_receipt_SHA256']}
def write(p,q):
 q.update(keeper_step=keeper,keeper_hostname=socket.gethostname(),keeper_PID=os.getpid(),heartbeat_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
def norm(s):return s.split()[0].rstrip('+')
def acct():return [x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=30).splitlines()]
def idle(tag):
 args=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:03:00','--job-name='+label+'-'+tag,'python3',str(r/'incoming/verify_gpu_idle_r5738.py')]
 z=subprocess.run(args,env=env,capture_output=True,text=True,timeout=190);assert z.returncode==0,(tag,z.stdout,z.stderr);a=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert len(a)==1 and a[0]['node']==node and a[0]['kfd_pids']==[] and a[0]['CUDA_not_initialized_by_probe'];return a[0]
write(rp,d);write(wp,w)
try:
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['required_existing_SHA256'].items():assert sha(r/n)==h,n
 cpu_path=r/'candidates/GLM-M15-component-r121b/CPU-build.json';assert sha(cpu_path)==go['CPU_receipt_SHA256']==m['CPU_receipt_SHA256'];cpu=json.loads(cpu_path.read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU_initialized']and not cpu['dlopen']and cpu['no_recompile_or_relink'];assert cpu['repair_manifest_SHA256']==m['CPU_manifest_SHA256'];assert cpu['library_SHA256']==m['DSO_SHA256']==sha(str(cpu['library']).replace('/refactor',str(r),1))
 gm=json.loads((inc/'GPU-manifest-r127.json').read_bytes());assert sha(inc/'GPU-manifest-r127.json')==m['GPU_manifest_SHA256'];assert sha(source/'manifest-r121b.json')==m['CPU_manifest_SHA256']
 for n,h in gm['source_SHA256'].items():assert sha(inc/n)==h,n
 cm=json.loads((source/'manifest-r121b.json').read_bytes())
 for n,h in cm['source_SHA256'].items():assert sha(source/n)==h,n
 old_failure=r/'candidates/GLM-M15-component-r121/CPU-build.json';assert sha(old_failure)==m['original_failed_receipt_SHA256']and not json.loads(old_failure.read_bytes())['completed']
 # Copy only after every published source, CPU receipt and DSO pin has passed.
 pinned=r/'results'/m['pinned_source_directory'];assert not pinned.exists();pinned.mkdir()
 for name in ['GPU-manifest-r127.json','GPU_inside_r127.sh','offset8_harness_r127.py']:
  shutil.copyfile(inc/name,pinned/name);assert sha(pinned/name)==sha(inc/name)
 d['pinned_sources']={p.name:sha(p)for p in pinned.iterdir()};write(rp,d)
 rows=acct();assert any(x[0]==parent and norm(x[2])=='RUNNING'for x in rows);assert not any(x[1]==label for x in rows)
 for x in rows:
  if x[0].startswith(parent+'.')and x[0].split('.')[-1].isdigit()and norm(x[2])not in term:assert node not in subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True,timeout=20).splitlines(),x
 d['fresh_before_KFD']=idle('before');w.update(state='ready_before_uniqueGPU',ready=True);write(rp,d);write(wp,w)
 args=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c8','-w',node,'--gres=dcu:1','--mem=32768M','-t','00:20:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=GLM-M15-offset8-r127/GPU_inside_r127.sh','GLM_M15_CPU_RECEIPT_SHA256_R121B='+go['CPU_receipt_SHA256'],'GLM_M15_PINNED_SOURCES_R127=/refactor/results/'+m['pinned_source_directory'],'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(r/'logs'/(label+'-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='uniqueGPUsubmitted_no_restart',args=args,GPUclientPID=proc.pid);write(rp,d);start=time.monotonic();client_exit_seen=None
 while time.monotonic()-start<1320:
  matches=[x for x in acct()if x[1]==label];assert len(matches)<=1
  if matches:
   row=matches[0];assert known is None or known==row[0];known=row[0];w.update(actual_handle=known,Slurm=row,state='following_same_handle');write(wp,w)
   if norm(row[2])in term:terminal=row;w['terminal']=row;break
  elif proc.poll()is not None:
   if client_exit_seen is None:client_exit_seen=time.monotonic()
   elif time.monotonic()-client_exit_seen>60:raise RuntimeError('srun_exited_accounting_grace_without_named_step_no_retry')
  time.sleep(10)
 else:raise RuntimeError('boundedwatch_samehandle_unknown_no_retry')
 assert known is not None;proc.wait(timeout=30);w['GPU_client_returncode']=proc.returncode
 result=r/'results'/('GLM-M15-offset8-r127-'+known+'.json');w['result_exists']=result.exists()
 if result.exists():
  result_data=json.loads(result.read_bytes());w['result_SHA256']=sha(result);w['result_completed']=bool(result_data.get('completed'));w['normal_close']=bool(result_data.get('normal_close'));w['result_error']=result_data.get('error');records=result_data.get('offsets',[])
  w['offset_summary']=[{k:a.get(k)for k in ['logical_rank','column_offset','median_rate_ratio','positive_pairs','completed','normal_close','fresh_weight_audit','fresh_scale_audit','clear_reuse_audit']}for a in records]
  w['private_cache_closed']=len(records)==8 and all(len(a.get('after_clear_stats',[]))==16 and a['after_clear_stats'][6]==a['after_clear_stats'][7]and a['after_clear_stats'][11]==a['after_clear_stats'][12]==0 for a in records)
  w['CPU_receipt_matches']=result_data.get('CPU_receipt_SHA256')==go['CPU_receipt_SHA256'];w['DSO_matches']=result_data.get('library_SHA256')==m['DSO_SHA256'];w['exact_scope']=result_data.get('M')==15 and result_data.get('N')==6144 and result_data.get('K')==2048 and [a.get('logical_rank')for a in records]==list(range(8))and all(a['column_offset']==a['logical_rank']*2048 and len(a.get('warm',[]))==4 and len(a.get('trials',[]))==8 for a in records)
  w['all_offsets_validated']=bool(result_data.get('all_offsets_validated'));w['all_offsets_qualified']=bool(result_data.get('all_offsets_qualified'));w['numeric_positive']=bool(result_data.get('all_offsets_positive'))
 post_attempted=True;w['post_KFD']=idle('post');w['post_passed']=True
 w['qualified']=bool(terminal[2:4]==['COMPLETED','0:0']and proc.returncode==0 and all(w.get(k)for k in ['result_completed','normal_close','private_cache_closed','CPU_receipt_matches','DSO_matches','exact_scope','all_offsets_validated','post_passed'])and not w.get('result_error'))
 w.update(state='terminal_audited_postchecked',Model_promotion=False,one_component_only=True);d.update(state='completed_keeper',actual_handle=known,qualified=w['qualified'],numeric_positive=w['numeric_positive']);write(wp,w);write(rp,d)
except BaseException as e:
 d.update(error=repr(e),traceback=traceback.format_exc(),state='unknown_GPU_follow_samehandle'if proc else'failed_beforeGPU',actual_handle=known);w.update(error=repr(e),qualified=False,actual_handle=known)
 if proc and terminal and not post_attempted:
  try:w.update(post_KFD=idle('post'),post_passed=True)
  except BaseException as post:w.update(post_passed=False,post_error=repr(post))
 elif proc and not terminal:w['post_not_run']='GPUstateunknown_followrecordedhandle_no_restart_no_otherjob_cancellation'
 elif post_attempted:w.update(post_passed=False,post_error=repr(e))
 write(rp,d);write(wp,w);raise
