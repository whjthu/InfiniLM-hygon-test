"""Once-only16card residency keeper. CPU recording may replace Slurm leaves only."""
from pathlib import Path
import os,json,time,hashlib,re,subprocess,traceback,importlib.util
from control_wire import atom
from contract import ROLE_NODES,GPU_HARD_S

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def gpu_arguments(root,directory,label):
 wrapper=Path(root)/'incoming/Deep-paired-quality-r375/controller/resource_wave_host_alias_outer.sh'
 return ['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=0','-N2','-n2','--ntasks-per-node=1','-c32','-w','a02r4n12,a02r4n13','--gres=dcu:8','--mem=240000M','-t','01:30:00','--job-name='+label,'--output='+str(Path(root)/'logs'/(label+'-%J-rank%t.log')),'bash',str(wrapper),'probe']

def verify(directory,expected,root=R):
 directory=Path(directory);m=json.loads((directory/'GPU_MANIFEST.json').read_bytes());assert sha(directory/'GPU_MANIFEST.json')==expected and m['GPU_GO']is True
 assert sha(directory/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256']
 source=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(directory/n)==h for n,h in source['files_SHA256'].items())
 def host(p):return Path(root)/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)
 p=host(m['CPU_receipt']);assert sha(p)==m['CPU_receipt_SHA256'];q=json.loads(p.read_bytes())
 assert q['completed']and q['passed']and q['GPU']==0 and not q['Native_dlopen']and q['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']
 assert q['actual_new_Native_binding_derived']and q['actual_fixture_encoder_SHA256_verified']
 assert m['fixture_encoders']==q['phases']['actual_fixture']['result']['fixtures']
 assert sha(host(m['Native_CPU_receipt']))==m['Native_CPU_receipt_SHA256']
 return m

def run(root,directory,run,nonce,expected,publication_commit,*,runner=subprocess.run,spawn=subprocess.Popen,client=None):
 root=Path(root);directory=Path(directory);m=verify(directory,expected,root);label='Deep-PD-r445-'+run
 watch=root/'results'/(label+'-watch.json');assert not watch.exists();cancel=root/'control'/(label+'-cancel.json');assert not cancel.exists()
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 w=dict(completed=False,passed=False,run=run,source_manifest_SHA256=expected,publication_commit=publication_commit,GPU_submitted=False,all_service_cards=16,no_retry=True,actual_GPU=None)
 save=lambda:atom(watch,w);save();process=None;log=None
 def command(args,**kw):return runner(args,env=env,capture_output=True,text=True,timeout=kw.pop('timeout',30),**kw)
 def rows():
  p=command(['sacct','-S','2026-11-01','-j','910376','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,NodeList%100']);assert p.returncode==0;return [s.split('|')for s in p.stdout.splitlines()]
 def exact():
  found=[s for s in rows()if s[1]==label];assert len(found)<=1
  if found:assert re.fullmatch(r'910376\.\d+',found[0][0]);return found[0]
 def idle(phase):
  args=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w','a02r4n12,a02r4n13','--gres=none','--mem=512M','-t','00:02:00','--job-name='+label+'-'+phase,'python3','-B',str(directory/'idle.py')]
  if phase=='post':args+=['--wait-port-release']
  p=command(args,timeout=105);w[phase+'_probe_raw']=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr);save()
  values=[json.loads(s)for s in p.stdout.splitlines()if s.startswith('{')];assert p.returncode==0 and len(values)==2
  assert {v['node']for v in values}==set(ROLE_NODES.values())
  assert all(v['KFD_available']and not v['kfd_pids']and not v['owned_process_PIDs']and all(v['ports_bindable'].values())for v in values)
  return values
 def dedup(ignore=()):
  for s in rows():
   if re.fullmatch(r'910376\.\d+',s[0])and s[0]not in ignore and s[2].split()[0].rstrip('+')not in TERMINAL:
    p=command(['scontrol','show','hostnames',s[4]]);assert p.returncode==0 and not(set(p.stdout.splitlines())&set(ROLE_NODES.values()))
 try:
  parent=command(['scontrol','show','job','910376','-o']);assert parent.returncode==0 and'JobState=RUNNING'in parent.stdout and'UserId=aclsylqidf('in parent.stdout
  nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent.stdout).group(1);p=command(['scontrol','show','hostnames',nodes]);assert set(ROLE_NODES.values())<=set(p.stdout.splitlines())
  assert exact()is None;dedup();w['fresh_before']=idle('before');dedup([v['actual_step']for v in w['fresh_before']]);save()
  namespace_path=root/'incoming/Deep-paired-quality-r375/controller/namespace_binding.py';assert sha(namespace_path)==m['namespace_binding_SHA256']
  spec=importlib.util.spec_from_file_location('pd382_namespace',namespace_path);ns=importlib.util.module_from_spec(spec);spec.loader.exec_module(ns);w['namespace']=ns.verify_namespace(root,namespace_path.parent)
  epoch=time.time();w.update(original_GPU_epoch_unix=epoch,GPU_hard_deadline_unix=epoch+5400)
  env.update(WAVE_INSIDE_R2323='Deep-PD-performance-r445/GPU_inside.sh',DEEP_PD_RUN=run,DEEP_PD_SESSION=nonce,DEEP_PD_EXECUTION_SHA=expected,DEEP_PD_GPU_EPOCH=str(epoch),DEEP_PD_CANCEL=str(cancel))
  log=(root/'logs'/(label+'-srun-client.log')).open('x');process=spawn(gpu_arguments(root,directory,label),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);w.update(GPU_submitted=True,srun_PID=process.pid);save()
  until=epoch+180
  while True:
   markers=[]
   for role in ['P','D']:
    path=root/'control'/('Deep382-'+run+'-'+role+'-listening.json')
    if path.exists():
     value=json.loads(path.read_bytes());assert value['source_manifest_SHA256']==expected and value['run']==run and value['session_nonce_SHA256']==hashlib.sha256(nonce.encode()).hexdigest();markers.append(value)
   if len(markers)==2:break
   assert process.poll()is None and time.time()<until,'Original GPU bootstrap failed/early listeners not visible';time.sleep(.2)
  steps={v['actual_step']for v in markers};assert len(steps)==1 and re.fullmatch(r'910376\.\d+',next(iter(steps)))
  w['actual_GPU']=next(iter(steps));w['early_listener_markers']=markers;save()
  if client is None:
   from support import load
   from controller import execute
   from contract import load_plan,bind_encoders
   def progress(value):w['client_progress']=value;save()
   modules=load(directory);client=lambda:execute(root,directory,run,nonce,expected,epoch,cancel,modules['state_wire'],modules['token_client'].TokenClient,bind_encoders(load_plan(directory/'BENCHMARK_PLAN.json'),m['fixture_encoders']),progress=progress)
  w['client']=client();w['srun_exit_code']=process.wait(timeout=min(90,max(1,epoch+5400-time.time())))
  until=epoch+5520
  while True:
   try:step=exact()
   except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as observation:
    w['observation_error']=repr(observation);save();assert time.time()<until;time.sleep(2);continue
   if step and step[2].split()[0].rstrip('+')in TERMINAL:break
   assert time.time()<until,'Bounded accounting visibility expired after original GPU run';time.sleep(2)
  w.update(actual_GPU=step[0],terminal=step);w['post']=idle('post');w['role_reports']={}
  for role in ['P','D']:
   path=root/'results'/('Deep382-'+run+'-'+role+'.json');value=json.loads(path.read_bytes())
   assert value['completed']and value['passed']and value['normal_control_socket_close']and value['normal_actual8owner_destroy']
   w['role_reports'][role]=dict(path=str(path),SHA256=sha(path),actual_step=value['actual_step'])
  assert w['srun_exit_code']==0 and step[2:4]==['COMPLETED','0:0']and w['client']['all16_actual_owners_destroyed']
  w.update(completed=True,passed=bool(w['client']['completed']and w['client']['quality_all_task_and_natural_EOS']));save();return w
 except BaseException as e:
  w.update(error=repr(e),traceback=traceback.format_exc());save()
  if process is not None:
   atom(cancel,dict(run=run,reason='Owned PD failure; no replay',session_nonce_SHA256=hashlib.sha256(nonce.encode()).hexdigest()))
   step=exact()
   if step:
    w['actual_GPU']=step[0]
    if step[2].split()[0].rstrip('+')not in TERMINAL:w['forced_scancel']=command(['scancel',step[0]]).returncode
   try:w['srun_exit_code']=process.wait(timeout=45)
   except Exception as close:w['srun_cleanup_error']=repr(close)
   try:w['post_after_failure']=idle('post')
   except Exception as close:w['post_failure']=repr(close)
  w['completed']=True;save();raise
 finally:
  if log is not None:log.close()

def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--run',required=True);a=a.parse_args()
 assert os.environ['SLURM_JOB_ID']=='910407'and re.fullmatch('[0-9a-f]{40}',a.publication_commit)and re.fullmatch('[a-z0-9-]+',a.run)
 assert float(os.environ['DEEP_PD_CPU_EPOCH'])+5700-time.time()>5580,'Original CPU budget insufficient before loading any Model'
 run(R,Path(__file__).resolve().parent,a.run,os.environ['DEEP_PD_SESSION'],a.manifest_sha,a.publication_commit)
if __name__=='__main__':main()
