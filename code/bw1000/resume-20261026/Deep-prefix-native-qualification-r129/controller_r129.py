"""One future owned TP8 Native qualification. No retry or model promotion."""
import pathlib,json,subprocess,os,time,hashlib,argparse,datetime,socket
ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);ap.add_argument('--go-file',required=True);a=ap.parse_args()
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-prefix-native-qualification-r129';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();label='Deep-prefix-native-once-r129';parent='910374';node='a02r4n05';assert socket.gethostname()=='a14r2n20'and os.environ['SLURM_JOB_ID']=='910407'
assert sha(D/'manifest-r129.json')==a.manifest_sha;m=json.loads((D/'manifest-r129.json').read_bytes());g=json.loads(pathlib.Path(a.go_file).read_bytes());assert g['explicit_native_GO']is True and g['source_published']is True and g['source_manifest_SHA256']==a.manifest_sha and g['parent']==parent and g['node']==node
import re
assert g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and re.fullmatch('[0-9a-f]{64}',g['CPU_receipt_SHA256'])and re.fullmatch('[0-9a-f]{64}',g['DSO_SHA256'])
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
def mapped(p):return pathlib.Path(str(p).replace('/refactor',str(R),1))
cp=mapped(g['CPU_receipt']);assert sha(cp)==g['CPU_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU_initialized']and not cpu['dlopen']and cpu['source_manifest_SHA256']==a.manifest_sha and cpu['library_SHA256']==g['DSO_SHA256']==sha(mapped(cpu['library']))
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
wp=R/(label+'-watch.json');lp=R/(label+'-launch.json');assert not wp.exists()and not lp.exists();proc=None;terminal=None;known=None;posted=False;w={'qualified':False,'GPU_submitted':False,'restarts':0,'scope':'Native fixedcohort/43layerprefixmechanics only; no MTP head/quality/performance claim','publication_commit':g['publication_commit'],'source_manifest_SHA256':a.manifest_sha,'CPU_receipt_SHA256':g['CPU_receipt_SHA256'],'DSO_SHA256':g['DSO_SHA256']}
def save():
 w.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),keeper_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);p=wp.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(wp)
def acct():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-27','-j',parent,'-n','-P','-o','JobID,JobName%110,State,ExitCode,NodeList%110'],env=env,text=True,timeout=25).splitlines()]
TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def idle(tag):
 p=R/'incoming/verify_gpu_idle_r5738.py';assert sha(p)=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53';args=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+label+'-'+tag,'python3',str(p)];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=150);q=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert z.returncode==0 and len(q)==1 and q[0]['node']==node and q[0]['kfd_pids']==[] and q[0]['CUDA_not_initialized_by_probe'];return q[0]
with lp.open('x')as f:json.dump({'source_manifest_SHA256':a.manifest_sha,'GO_SHA256':sha(a.go_file),'publication_commit':g['publication_commit'],'restarts':0},f,indent=2)
save()
try:
 rows=acct();assert any(v[0]==parent and v[2]=='RUNNING'for v in rows)and not any(v[1]==label for v in rows)
 for v in rows:
  if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and v[2].split()[0].rstrip('+')not in TERM:assert node not in subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=20).splitlines()
 w['fresh_before']=idle('before');runtime=R/'results'/('Deep-prefix-native-r129-runtime.json')
 with runtime.open('x')as f:json.dump(g,f,indent=2)
 w['runtime_SHA256']=sha(runtime);save()
 args=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--kill-on-bad-exit=1','--cpu-bind=none','-N1','-n1','-c32','-w',node,'--gres=dcu:8','--mem=65536M','--time=00:40:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=Deep-prefix-native-qualification-r129/GPU_inside_r129.sh','DEEP_PREFIX_RUNTIME_R129=/refactor/results/Deep-prefix-native-r129-runtime.json','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(R/'logs'/(label+'-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,args=args,client_PID=proc.pid,state='following_one_unique_handle');save();tick=time.monotonic();exit_seen=None
 while time.monotonic()-tick<2500:
  matches=[v for v in acct()if v[1]==label];assert len(matches)<=1
  if matches:
   row=matches[0];assert known is None or known==row[0];known=row[0];w.update(actual_handle=known,Slurm=row);save()
   if row[2].split()[0].rstrip('+')in TERM:terminal=row;w['terminal']=row;break
  elif proc.poll()is not None:
   if exit_seen is None:exit_seen=time.monotonic()
   elif time.monotonic()-exit_seen>60:raise RuntimeError('client exited accounting unknown; inspect same label, never restart')
  time.sleep(10)
 else:raise RuntimeError('bounded observation expired; follow same actual handle, no resubmission')
 proc.wait(timeout=30);w['GPU_client_returncode']=proc.returncode;posted=True;w['post']=idle('post');w['post_empty']=True
 result=R/'results'/('Deep-prefix-native-r129-'+known+'.json');q=json.loads(result.read_bytes());w.update(result=str(result),result_SHA256=sha(result),result_summary={k:q.get(k)for k in ['completed','normal_close','all_cases_passed','actual_owners','physical_rows','capacity','Model_gain','MTP_head','error']})
 assert terminal[2:4]==['COMPLETED','0:0']and proc.returncode==0 and q['completed']and q['normal_close']and q['all_cases_passed']and len(q['cases'])==16
 log=R/'logs'/(label+'-'+known+'.log');text=log.read_text(errors='replace');assert text.count('DEEP_PREFIX_R129_NORMAL')==1 and 'Using network Socket'not in text;w['network_IBext_v8_seen']='Using network IBext_v8'in text;w['GDR']='UNKNOWN';w.update(qualified=True,state='terminal_native_mechanics_qualified_no_MTP_speed_claim');save()
except BaseException as e:
 w.update(error=repr(e),actual_handle=known,state='terminal_failure'if terminal else'unknown_follow_same_handle'if proc else'failed_before_GPU')
 if terminal and not posted:
  try:w['error_post']=idle('errorpost')
  except BaseException as error:w['post_error']=repr(error)
 save();raise
